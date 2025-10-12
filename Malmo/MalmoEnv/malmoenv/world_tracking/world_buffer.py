from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from malmoenv.world_tracking.utils import SectionBlob, WorldUpdate


@dataclass
class ActiveChunk:
    cx: int
    cz: int
    loaded_on_tick: int
    sections: List[SectionBlob]
    unloaded_on_tick: Optional[int] = None
    # store single-block updates as (tick, x, y, z, state)
    updates: List[Tuple[int, int, int, int, int]] = field(default_factory=list)

    def append_update(self, tick: int, x: int, y: int, z: int, state: int) -> None:
        self.updates.append((tick, x, y, z, state))



# ---- World buffer ------------------------------------------------------------
class WorldTrajectory:
    def __init__(self) -> None:
        self.chunks: List[ActiveChunk] = []
        self.tick_views: List[Dict[Tuple[int, int], int]] = []
        self.current_view: Dict[Tuple[int, int], int] = {}
        self.age = -1


    def add_update(self, world_update: WorldUpdate) -> None:
        """Append one tick of updates (loads/unloads/single-block edits)."""
        self.age += 1
        tick = self.age

        # 1) Handle unloads first: mark and remove from current view.
        for event in world_update.chunk_unloads:
            key = (event.cx, event.cz)
            idx = self.current_view.pop(key, None)
            if idx is not None:
                self.chunks[idx].unloaded_on_tick = tick

        # 2) Handle loads: create new ActiveChunk incarnations and add to current view.
        for event in world_update.chunk_loads:
            chunk = ActiveChunk(cx=event.cx, cz=event.cz, loaded_on_tick=tick, sections=list(event.sections))
            self.chunks.append(chunk)
            self.current_view[(event.cx, event.cz)] = len(self.chunks) - 1

        # 3) Handle single-block updates: route to the currently active chunk for that (cx,cz).
        if world_update.block_updates is not None and world_update.block_updates.size > 0:
            block_updates = world_update.block_updates  # structured array with fields x,y,z,state
            # Iterate rows (usually small); we keep it simple.
            for x, y, z, state in zip(block_updates["x"], block_updates["y"], block_updates["z"], block_updates["state"]):
                cx = int(x) >> 4
                cz = int(z) >> 4
                idx = self.current_view.get((cx, cz))
                if idx is not None:
                    self.chunks[idx].append_update(tick, int(x), int(y), int(z), int(state))

        # 4) Snapshot the current view for this tick.
        #    Store a shallow copy so later mutations don't affect past ticks.
        self.tick_views.append(dict(self.current_view))


    def get_world_state(self, tick: int) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Return a dictionary mapping (cx,cz) to a dense (256,16,16) uint16 array
        representing the chunk content at the specified tick.
        """
        if tick < 0 or tick > self.age:
            raise IndexError(f"tick {tick} out of range [0, {self.age}]")

        view = self.tick_views[tick]
        result: Dict[Tuple[int, int], np.ndarray] = {}

        for key, chunk_idx in view.items():
            active_chunk = self.chunks[chunk_idx]

            # Start with empty air chunk [y=0..255, z=0..15, x=0..15]
            chunk_arr = np.zeros((256, 16, 16), dtype=np.uint16)

            # Decode each present section and place it at the right y-range.
            for section in active_chunk.sections:
                sy = int(section.sy) & 0x0F  # 0..15
                sec_arr = section.as_numpy_array()
                y0 = sy * 16
                chunk_arr[y0 : y0 + 16, :, :] = sec_arr

            # Apply single-block updates up to the requested tick.
            for utick, x, y, z, state in active_chunk.updates:
                if utick <= tick: #TODO can be slightly optimised by vectorising updates
                    ly = int(y)
                    lx = int(x) & 15
                    lz = int(z) & 15
                    if 0 <= ly < 256:
                        chunk_arr[ly, lz, lx] = np.uint16(state & 0xFFFF)

            result[key] = chunk_arr

        return result


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

    class TrajectoryView:
        def __init__(self, world_buffer: "WorldBuffer", sample_index: np.ndarray) -> None:
            self._world_buffer = world_buffer
            self.sample_index = np.asarray(sample_index, dtype=np.int64)
            if self.sample_index.ndim != 2:
                raise ValueError("sample_index must be 2D (batchSize, sequenceSize)")

        @property
        def shape(self) -> Tuple[int, int]:
            return tuple(self.sample_index.shape)  # (batch, time)

        def get(self, batch: int, time: int):
            """
            Resolve the (batch, time) entry to a world state dict for that global tick.
            Returns: Dict[(cx,cz) -> uint16[256,16,16]]
            """
            global_tick = int(self.sample_index[batch, time])
            return self._world_buffer.get_world_state(global_tick)

        def __getitem__(self, idx):
            """Sugar: view[batch, time]."""
            batch, time = idx
            return self.get(batch, time)

    # ---- WorldBuffer proper ----
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._records: deque[_TrajectoryRecord] = deque()
        self._current: Optional[_TrajectoryRecord] = None  # current (open or just-closed) episode
        self._global_tick: int = -1                       # increases by 1 per append

    # ---- public API ----
    def append_update(self, world_update: WorldUpdate, is_first: bool, done: bool) -> None:
        self._global_tick += 1

        # Start new episode?
        if is_first or self._current is None:
            new_trajectory = WorldTrajectory()
            record = _TrajectoryRecord(start_tick=self._global_tick, latest_tick=self._global_tick, trajectory=new_trajectory)
            self._records.append(record)
            self._current = record

        # Append updates to the current episode
        self._current.trajectory.add_update(world_update)
        self._current.latest_tick = self._global_tick  # keep end updated even while open TODO

        # If episode ended, we simply leave the record as finished; the next reset will open a new one.
        self._current.done = done

        # Prune at most one oldest finished trajectory if it fell outside the window.
        self._prune_oldest_if_needed()

    def get_world_state(self, global_tick: int) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Return {(cx,cz): uint16[256,16,16]} for a given GLOBAL tick.
        """
        record, local_tick = self._locate_record_and_local_tick(global_tick)
        return record.trajectory.get_world_state(local_tick)

    def get_trajectories(self, sample_index: np.ndarray) -> "WorldBuffer.TrajectoryView":
        """
        Return a lazy view that can resolve (batch, time) -> world state dicts on demand.
        """
        return WorldBuffer.TrajectoryView(self, sample_index)

    # ---- internals ----
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