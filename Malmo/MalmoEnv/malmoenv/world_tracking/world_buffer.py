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

class WorldBuffer:

    class TrajectoryView:
        # Wrapper around WorldTrajectory's to represent batch of size: batchSize, sequenceSize
        # Stores batchSize pointers to WorldTrajectory, for each maps world index to index from 0 to sequenceSize
        pass

        def get(self, index):
            pass

    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.worlds: List[WorldTrajectory] = []

