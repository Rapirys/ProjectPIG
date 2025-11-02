import struct
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

MAGIC = b"BLSK"

@dataclass(frozen=True)
class SectionBlob:
    """One 16×16×16 section payload as written by ExtendedBlockStorage.getData().write(PacketBuffer)."""
    sy: int
    data: memoryview

    def as_numpy_array(self) -> np.ndarray:
        return _decode_section(self.data)


@dataclass(frozen=True)
class ChunkLoad:
    cx: int
    cz: int
    sections: List[SectionBlob]  # zero-copy section payloads


@dataclass(frozen=True)
class ChunkUnload:
    cx: int
    cz: int

@dataclass(frozen=True)
class WorldUpdate:
    """Container for one BLSK blob, zero-copy views into the original bytes."""
    chunk_loads: List[ChunkLoad]
    chunk_unloads: List[ChunkUnload]
    block_updates: Optional[np.ndarray]

    def summary(self) -> dict:
        n_load_secs = sum(len(c.sections) for c in self.chunk_loads)
        return {
            "loads": n_load_secs,
            "unloads": len(self.chunk_unloads),
            "blocks": 0 if self.block_updates is None else int(self.block_updates.shape[0]),
        }


def _read_varint(buf: memoryview, offset: int) -> tuple[int, int]:
    """Minecraft VarInt (base-128, 7-bit groups). Returns (value, new_offset)."""
    num = 0
    shift = 0
    while True:
        if offset >= len(buf):
            raise ValueError("truncated VarInt")
        b = buf[offset]
        offset += 1
        num |= (b & 0b01111111) << shift
        if (b & 0b10000000) == 0:
            break
        shift += 7
        if shift > 35:
            raise ValueError("VarInt too long")
    return num, offset


def _unpack_data_array(packed_words: np.ndarray, bits_per_entry: int) -> np.ndarray:
    if packed_words.dtype != np.uint64:
        raise ValueError("packed_words must be uint64")

    padded = np.concatenate([packed_words, np.zeros(1, dtype=np.uint64)])

    voxel_index = np.arange(4096, dtype=np.uint32)
    bpe32 = np.uint32(bits_per_entry)

    bit_index = voxel_index * bpe32
    word_index = (bit_index >> 6).astype(np.int64)   # // 64
    intra = (bit_index & 63).astype(np.uint64)       # % 64

    base = padded[word_index] >> intra
    # If intra == 0, there is no spill; else pull high bits from next word
    add = np.where(intra == 0, np.uint64(0), padded[word_index + 1] << (np.uint64(64) - intra))

    mask = (np.uint64(1) << np.uint64(bits_per_entry)) - np.uint64(1)
    return (base | add) & mask


def _decode_section(section_bytes: memoryview) -> np.ndarray:
    """
    Decode bytes produced by ExtendedBlockStorage.getData().write(PacketBuffer)
    (Minecraft 1.12 DataPaletteBlock) into a (16, 16, 16) uint16 array.

    Wire layout (1.12 palettized format):
      u8  bits_per_entry
      if bits_per_entry <= 8:
          VarInt palette_size
          palette_size * VarInt  # each is a GLOBAL block-state ID (not legacy)
      VarInt data_array_length   # number of 64-bit words that follow
      data_array_length * u64    # packed indices, u64 written BIG-endian
      VarInt non_air_count       # trailing count; not needed for decoding
    """
    buf = section_bytes
    offset = 0
    if len(buf) < 1:
        raise ValueError("empty section payload")

    bits_per_entry = buf[offset]
    offset += 1
    if bits_per_entry == 0:
        # TODO verify if correct. 1.12 never writes bpe==0. Assume that Entire section is air.
        return np.zeros((16, 16, 16), dtype=np.uint16)

    # ---- Optional local palette (present only when bits_per_entry <= 8) ----
    palette = None
    if bits_per_entry <= 8:
        palette_size, offset = _read_varint(buf, offset)
        palette_values = np.empty(palette_size, dtype=np.uint16)
        for i in range(palette_size):
            state_id, offset = _read_varint(buf, offset)
            palette_values[i] = state_id & 0xFFFF
        palette = palette_values

    # ---- Packed index array (u64 words written BIG-endian on the wire) ----
    data_array_length, offset = _read_varint(buf, offset)
    bytes_in_words = data_array_length * 8
    if offset + bytes_in_words > len(buf):
        raise ValueError("truncated packed-index array")

    # Read as big-endian 64-bit unsigned, then convert to native endianness
    # so bit shifts operate as expected on the host.
    packed_words = (
        np.frombuffer(buf[offset : offset + bytes_in_words], dtype=">u8")
        .astype(np.uint64, copy=False)
    )
    offset += bytes_in_words

    # ---- Consume (and ignore) optional trailing non-air count ----
    if offset < len(buf):  # TODO likely not needed
        try:
            _, offset = _read_varint(buf, offset)
        except ValueError:
            # Some writers may omit it; safe to ignore.
            pass

    # ---- Vectorized unpack of 4096 indices (no Python loop over voxels) ----
    # Index i encodes (y, z, x) with i = (y*16 + z)*16 + x (y fastest),
    # which matches reshape((16, 16, 16)) at the end.
    values = _unpack_data_array(packed_words, bits_per_entry)

    # ---- Map indices to global state IDs ----
    if palette is not None:
        # Safely map only indices that fall inside the palette.
        result = np.zeros_like(values, dtype=np.uint16)
        valid = values < palette.size
        result[valid] = palette[values[valid]]
    else:
        # Direct/global palette: indices are already global state IDs.
        result = values.astype(np.uint16, copy=False)

    # Reshape to [y, z, x]
    return result.reshape(16, 16, 16).transpose((2, 0, 1))



def decode_world_update(blob: bytes) -> WorldUpdate:
    """Apply one world-state update blob. Returns a small summary."""
    if not blob:
        return WorldUpdate([], [], None)

    buf = memoryview(blob)
    magic, n_loads, n_unloads, n_blocks = _read_header(buf)

    offset = 16
    chunk_loads: List[ChunkLoad] = []
    chunk_unloads: List[ChunkUnload] = []
    block_updates: Optional[np.ndarray] = None

    # ---- chunk loads ----
    for _ in range(n_loads):
        cx, cz = struct.unpack_from("<ii", buf, offset)
        offset += 8
        (section_mask,) = struct.unpack_from("<H", buf, offset)
        offset += 2

        sections: List[SectionBlob] = []
        # For each set bit in mask, a section follows: [u8 sy][u32 len][bytes...]
        for sy in range(16):
            if (section_mask >> sy) & 1:
                (sy_wire,) = struct.unpack_from("<B", buf, offset)
                offset += 1
                (sec_len,) = struct.unpack_from("<I", buf, offset)
                offset += 4
                sec_bytes = buf[offset : offset + sec_len]
                offset += sec_len
                sections.append(SectionBlob(sy=sy_wire, data=sec_bytes))
        chunk_loads.append(ChunkLoad(cx=cx, cz=cz, sections=sections))

    # ---- chunk unloads ----
    for _ in range(n_unloads):
        cx, cz = struct.unpack_from("<ii", buf, offset)
        offset += 8
        chunk_unloads.append(ChunkUnload(cx, cz))

    # ---- single block updates ----
    if n_blocks:
        nbytes = n_blocks * 14
        block_slice = buf[offset : offset + nbytes]
        offset += nbytes

        # Structured, little-endian, no alignment (sum itemsize=14): zero-copy view.
        block_dtype = np.dtype(
            [("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("state", "<u2")], align=False
        )
        block_updates = np.frombuffer(block_slice, dtype=block_dtype, count=n_blocks)

    return WorldUpdate(
        chunk_loads=chunk_loads,
        chunk_unloads=chunk_unloads,
        block_updates=block_updates,
    )

def _read_header(buf):
    if len(buf) < 16:
        raise ValueError("world-state blob too short")

    magic, n_loads, n_unloads, n_blocks = struct.unpack_from("<4sIII", buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    return magic, n_loads, n_unloads, n_blocks
