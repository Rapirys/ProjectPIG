from collections import deque, OrderedDict
from dataclasses import dataclass, field
from itertools import chain
from typing import List, Dict, Tuple, Optional, Iterator
import numpy as np
import torch
from malmoenv.world_tracking.utils import SectionBlob, WorldUpdate


@dataclass
class ActiveChunk:
    cx: int
    cz: int
    loaded_on_tick: int
    sections: List[SectionBlob]
    unloaded_on_tick: Optional[int] = None
    updates: List[Tuple[int, int, int, int, int]] = field(default_factory=list)  # tick, x, y, z, state
    tick_lookup: OrderedDict[int, List[int]] = field(default_factory=OrderedDict)

    def append_update(self, tick: int, x: int, y: int, z: int, state: int) -> None:
        self.updates.append((tick, x, y, z, state))
        self.tick_lookup.setdefault(tick, []).append(len(self.updates) - 1)


# ---- World buffer ------------------------------------------------------------
class WorldTrajectory:
    def __init__(self, device: torch.device) -> None:
        self.chunks: List[ActiveChunk] = []
        self.added: List[List[int]] = []
        self.removed: List[List[int]] = []
        self.tick_views: List[List[int]] = []
        self.current_view: Dict[Tuple[int, int], int] = {}
        self.age = -1
        self.camera_positions: List[torch.Tensor] = []  # shape (len, 5): X,Y,Z,yaw,pitch
        self.device = device

    @torch.no_grad()
    def add_update(self, world_update: WorldUpdate, camera_position: np.ndarray) -> None:
        """Append one tick of updates (loads/unloads/single-block edits)."""
        self.age += 1
        tick = self.age

        # unloads
        removed = []
        for event in world_update.chunk_unloads:
            key = (event.cx, event.cz)
            idx = self.current_view.pop(key, None)
            if idx is not None:
                self.chunks[idx].unloaded_on_tick = tick
                removed.append(idx)
        self.removed.append(removed)

        # loads
        added = []
        for event in world_update.chunk_loads:
            chunk = ActiveChunk(cx=event.cx, cz=event.cz, loaded_on_tick=tick, sections=list(event.sections))
            self.chunks.append(chunk)
            idx = len(self.chunks) - 1
            self.current_view[(event.cx, event.cz)] = idx
            added.append(idx)
        self.added.append(added)

        # single-block edits
        if world_update.block_updates is not None and world_update.block_updates.size > 0:
            block_updates = world_update.block_updates  # structured array with fields x,y,z,state
            for x, y, z, state in zip(block_updates["x"], block_updates["y"], block_updates["z"], block_updates["state"]):
                cx = int(x) >> 4
                cz = int(z) >> 4
                idx = self.current_view.get((cx, cz))
                if idx is not None:
                    self.chunks[idx].append_update(tick, int(x), int(y), int(z), int(state))

        # Snapshot the current view for this tick and save camera pose.
        self.tick_views.append(list(self.current_view.values()))
        cam = torch.as_tensor(camera_position, dtype=torch.float32, device=self.device)
        self.camera_positions.append(cam)

    @torch.no_grad()
    def get_world_trajectory(
        self,
        scene_size: Tuple[float, float, float],
        start_tick: int = 0,
        end_tick: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if end_tick is None:
            end_tick = self.age
        if start_tick < 0 or end_tick > self.age or start_tick > end_tick:
            raise IndexError(f"tick range [{start_tick}, {end_tick}] out of bounds [0, {self.age}]")

        sx, sy, sz = map(int, scene_size)
        camera_positions = torch.stack(self.camera_positions[start_tick : end_tick + 1], dim=0)  # (T, 5)
        origins = compute_vox_origin_batch_np(camera_positions.detach().cpu().numpy(), scene_size) #TODO refactor compute_vox_origin_batch_np
        origins = torch.from_numpy(origins).to(self.device)

        chunk_view_iterator = self._get_world_trajectory(start_tick, end_tick)
        vol = torch.zeros((sx, sy, sz), dtype=torch.long, device=self.device)
        for i, view in enumerate(chunk_view_iterator):
            vol.zero_()
            ox = int(origins[i, 0].item())
            oy = int(origins[i, 1].item())
            oz = int(origins[i, 2].item())
            ex, ey, ez = ox + sx, oy + sy, oz + sz

            chunk_size = 16
            for (ccx, ccz), chunk_arr in view.items():
                base_x = ccx * chunk_size
                base_z = ccz * chunk_size

                # chunk AABB in world coords
                x0, x1 = max(ox, base_x), min(ex, base_x + chunk_size)
                z0, z1 = max(oz, base_z), min(ez, base_z + chunk_size)
                y0, y1 = max(oy, 0), min(ey, 256)
                if x0 >= x1 or y0 >= y1 or z0 >= z1:
                    continue

                # region indices
                rx0, rx1 = x0 - ox, x1 - ox
                ry0, ry1 = y0 - oy, y1 - oy
                rz0, rz1 = z0 - oz, z1 - oz

                # chunk indices (chunk_arr is [Y, X, Z] with X,Z local to chunk)
                lx0, lx1 = x0 - base_x, x1 - base_x
                ly0, ly1 = y0, y1
                lz0, lz1 = z0 - base_z, z1 - base_z
                vol[rx0:rx1, ry0:ry1, rz0:rz1] = chunk_arr[lx0:lx1, ly0:ly1, lz0:lz1]

            yield camera_positions[i], origins[i], vol.clone()

    @torch.no_grad()
    def _get_world_trajectory(self, start_tick: int, end_tick: int) -> Iterator[Dict[Tuple[int, int], torch.Tensor]]:
        view: Dict[Tuple[int, int], torch.Tensor] = {}
        view_iter: Dict[Tuple[int, int], Iterator[torch.Tensor]] = {}
        for chunk_idx in self.tick_views[start_tick]:
            chunk = self.chunks[chunk_idx]
            buffer, iterator = self.get_chunk_trajectory(chunk, start_tick=start_tick)
            key = (chunk.cx, chunk.cz)
            view[key] = buffer
            view_iter[key] = iterator

        for it in view_iter.values():
            next(it)
        yield view

        for added_chunks, removed_chunks in zip(self.added[start_tick+1:end_tick+1], self.removed[start_tick+1:end_tick+1]):
            for added_idx in added_chunks:
                added = self.chunks[added_idx]
                buffer, iterator = self.get_chunk_trajectory(added)
                key = (added.cx, added.cz)
                view[key] = buffer
                view_iter[key] = iterator

            for removed_idx in removed_chunks:
                removed = self.chunks[removed_idx]
                view.pop((removed.cx, removed.cz))
                view_iter.pop((removed.cx, removed.cz))

            for it in view_iter.values():
                next(it)
            yield view

    @torch.no_grad()
    def get_chunk_trajectory(self, chunk: ActiveChunk, start_tick: int = None) -> Tuple[torch.Tensor, Iterator[torch.Tensor]]:
        # TODO Trim chunk height early
        chunk_arr = torch.zeros((16, 256, 16), dtype=torch.long, device=self.device)
        start_tick = chunk.loaded_on_tick if start_tick is None else start_tick

        for section in chunk.sections:
            sy = int(section.sy) & 0x0F  # 0..15
            # decode section (np) -> torch.long on device
            sec_arr = torch.as_tensor(section.as_numpy_array(), dtype=torch.long, device=self.device)
            y0 = sy * 16
            chunk_arr[:, y0 : y0 + 16, :] = sec_arr

        def _iter() -> Iterator[torch.Tensor]:
            end = chunk.unloaded_on_tick if chunk.unloaded_on_tick is not None else self.age
            for idx in range(chunk.loaded_on_tick, end + 1):
                updates = [chunk.updates[i] for i in chunk.tick_lookup.get(idx, [])]
                for _, x, y, z, state in updates:
                    chunk_arr[int(x) & 15, int(y), int(z) & 15] = int(state & 0xFFFF)
                if idx >= start_tick:
                    yield chunk_arr

        return chunk_arr, _iter()


@dataclass
class _TrajectoryRecord:
    start_tick: int
    latest_tick: int
    trajectory: WorldTrajectory
    done: bool = False


class WorldBuffer:
    """
    Stores multiple WorldTrajectory objects across episodes.
    One append per environment step, aligned with replay index.
    Prunes finished trajectories once they fall completely outside the replay capacity window.
    """

    # ---- WorldBuffer proper ----
    def __init__(self, capacity: int, scene_size: Tuple[float, float, float], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.scene_size = scene_size
        self._records: deque[_TrajectoryRecord] = deque()
        self._current: Optional[_TrajectoryRecord] = None  # current (open or just-closed) episode
        self._global_tick: int = -1  # increases by 1 per append
        self.device = device

    @torch.no_grad()
    # ---- public API ----
    def append_update(self, world_update: WorldUpdate, camera_position: np.ndarray, is_first: bool, done: bool) -> None:
        self._global_tick += 1

        # Start new episode?
        if is_first or self._current is None:
            new_trajectory = WorldTrajectory(device=self.device)
            record = _TrajectoryRecord(start_tick=self._global_tick, latest_tick=self._global_tick, trajectory=new_trajectory)
            self._records.append(record)
            self._current = record

        # Append updates to the current episode
        self._current.trajectory.add_update(world_update, camera_position)
        self._current.latest_tick = self._global_tick  # keep end updated even while open

        # If episode ended, we simply leave the record as finished; the next reset will open a new one.
        self._current.done = done

        # Prune at most one oldest finished trajectory if it fell outside the window.
        self._prune_oldest_if_needed()

    @torch.no_grad()
    def get_trajectories(
        self, sample_index: np.ndarray, start_mask: np.ndarray
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Return a lazy view that can resolve (batch, time) -> world state tensors on demand.
        """
        if sample_index.ndim != 2 or start_mask.ndim != 2:
            raise ValueError("sample_index and start_mask must both be 2D (batch_size, batch_length)")
        if sample_index.shape != start_mask.shape:
            raise ValueError("sample_index and start_mask must have the same shape")
        if start_mask.dtype != np.bool_:
            raise ValueError("start_mask must be a boolean array")

        batch_size, batch_length = sample_index.shape
        gens_idx = []
        for i in range(batch_size):
            starts_cols = np.flatnonzero(start_mask[i])
            if starts_cols.size == 0 or starts_cols[0] != 0:
                starts_cols = np.r_[0, starts_cols]
            lengths = np.diff(np.r_[starts_cols, batch_length])
            row_pairs = [(int(sample_index[i, c]), int(l)) for c, l in zip(starts_cols, lengths)]
            gens_idx.append(row_pairs)

        gens: List[List[Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = []
        for i in range(len(gens_idx)):
            gens.append([])
            for global_start, length in gens_idx[i]:
                record, local_start = self._locate_record_and_local_tick(global_start)
                local_end = local_start + length - 1
                gens[i].append(record.trajectory.get_world_trajectory(self.scene_size, local_start, local_end))

        gens = [chain.from_iterable(gen) for gen in gens]
        for _ in range(batch_length):
            a_step = [next(g) for g in gens]  # per batch row: (cam, origin, vol)
            cams, origins, vols = zip(*a_step)
            yield (torch.stack(cams, dim=0), torch.stack(origins, dim=0), torch.stack(vols, dim=0))

    def _prune_oldest_if_needed(self) -> None:
        if not self._records:
            return
        cutoff = self._global_tick - self.capacity
        oldest = self._records[0]
        # prune only finished episodes
        if oldest.done and oldest.latest_tick < cutoff:
            self._records.popleft()

    def _locate_record_and_local_tick(self, global_tick: int):
        for record in self._records:
            if record.start_tick <= global_tick <= record.latest_tick:
                return record, global_tick - record.start_tick
        raise KeyError("tick pruned or out of range")


def compute_vox_origin_batch_np(
    player_xyz: np.ndarray,  # shape (B, 3), float or int
    scene_size: Tuple[float, float, float],
    chunk_size: int = 16,
) -> np.ndarray:
    size_x, size_y, size_z = scene_size
    min_chunk_x = (np.floor_divide(player_xyz[:, 0], chunk_size) * chunk_size) - (size_x // 2)
    min_chunk_z = (np.floor_divide(player_xyz[:, 2], chunk_size) * chunk_size) - (size_z // 2)

    half_y = size_y // 2
    min_chunk_y = np.clip(np.floor(player_xyz[:, 1] - half_y), 0, None)

    mins = np.stack([min_chunk_x, min_chunk_y, min_chunk_z], axis=-1)
    return mins.astype(np.int32, copy=False)
