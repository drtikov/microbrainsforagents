"""Pure-stdlib POSIX/Windows shared-memory 1 kHz sensorimotor bus with Seqlock framing.

Provides ``SharedMemorySensorimotorBus`` backed by ``multiprocessing.shared_memory.SharedMemory``
and fixed-size binary ``struct`` slots protected by even/odd Seqlock sequence counters:

- Writer increments slot ``seq`` from even to odd before updating ``timestamp_ns``,
  ``state``, ``u_nominal``, ``u_filtered``, ``min_barrier_slack``, ``tank_energy``,
  and ``watchdog_scale``, then increments ``seq`` to the next even number.
- Reader loops until ``seq1 == seq2`` and ``(seq1 & 1) == 0`` (and the header
  ``head_index`` remained unchanged across the read), ensuring lock-free torn-read
  immunity without kernel mutexes in the 1 kHz control path.
"""

from __future__ import annotations

import math
import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional, Sequence, Tuple

_MAGIC = b"ASMB"
_FORMAT_VERSION = 1

# Header: magic(4s), version(I), state_dim(I), control_dim(I), capacity(I), reserved(I), head_index(Q)
_HEADER_STRUCT = struct.Struct("<4sIIIIIQ")
_HEADER_SIZE = _HEADER_STRUCT.size  # 32 bytes

# Slot prefix: seq(Q), frame_index(Q), timestamp_ns(Q), min_barrier_slack(d), tank_energy(d), watchdog_scale(d)
_SLOT_SEQ_STRUCT = struct.Struct("<Q")
_SLOT_META_STRUCT = struct.Struct("<QQddd")


def _validate_finite_vector(vec: Sequence[float], expected_dim: int, label: str) -> Tuple[float, ...]:
    if isinstance(vec, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of numbers, not {type(vec).__name__}")
    out = []
    for item in vec:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} elements must be finite numbers, got {type(item).__name__}")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(f"{label} contains non-finite value")
        out.append(val)
    if len(out) != expected_dim:
        raise ValueError(f"{label} length {len(out)} != expected {expected_dim}")
    return tuple(out)


def _validate_finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{label} must be finite")
    return val


class SharedMemorySensorimotorBus:
    """Lock-free Seqlock shared-memory ring bus for 1 kHz sensorimotor telemetry."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        state_dim: Optional[int] = None,
        control_dim: Optional[int] = None,
        capacity: int = 16,
        create: bool = True,
        unlink_on_exit: bool = False,
    ) -> None:
        self._closed = False
        self._unlinked = False
        self._unlink_on_exit = bool(unlink_on_exit)
        self._created = bool(create)

        if create:
            eff_state_dim = 6 if state_dim is None else int(state_dim)
            eff_control_dim = 3 if control_dim is None else int(control_dim)
            eff_capacity = int(capacity)
            if eff_state_dim < 1 or eff_control_dim < 1 or eff_capacity < 1:
                raise ValueError("state_dim, control_dim, and capacity must be >= 1")
            self.state_dim = eff_state_dim
            self.control_dim = eff_control_dim
            self.capacity = eff_capacity
            self._payload_struct = struct.Struct(
                f"<QQddd{self.state_dim}d{self.control_dim}d{self.control_dim}d"
            )
            self.slot_size = _SLOT_SEQ_STRUCT.size + self._payload_struct.size
            self.total_bytes = _HEADER_SIZE + self.capacity * self.slot_size

            self._shm = SharedMemory(name=name, create=True, size=self.total_bytes)
            buf = self._shm.buf
            buf[: self.total_bytes] = b"\x00" * self.total_bytes
            _HEADER_STRUCT.pack_into(
                buf,
                0,
                _MAGIC,
                _FORMAT_VERSION,
                self.state_dim,
                self.control_dim,
                self.capacity,
                0,
                0,
            )
        else:
            if not name:
                raise ValueError("name must be provided when create=False")
            self._shm = SharedMemory(name=name, create=False)
            buf = self._shm.buf
            if len(buf) < _HEADER_SIZE:
                self._shm.close()
                raise ValueError("Shared memory segment smaller than bus header")
            magic, ver, hdr_sdim, hdr_cdim, hdr_cap, _reserved, _head = _HEADER_STRUCT.unpack_from(buf, 0)
            if magic != _MAGIC:
                self._shm.close()
                raise ValueError(f"Invalid sensorimotor bus magic: {magic!r}")
            if ver != _FORMAT_VERSION:
                self._shm.close()
                raise ValueError(f"Unsupported sensorimotor bus version: {ver}")
            if state_dim is not None and int(state_dim) != hdr_sdim:
                self._shm.close()
                raise ValueError(f"state_dim mismatch: requested {state_dim}, header has {hdr_sdim}")
            if control_dim is not None and int(control_dim) != hdr_cdim:
                self._shm.close()
                raise ValueError(f"control_dim mismatch: requested {control_dim}, header has {hdr_cdim}")
            self.state_dim = int(hdr_sdim)
            self.control_dim = int(hdr_cdim)
            self.capacity = int(hdr_cap)
            self._payload_struct = struct.Struct(
                f"<QQddd{self.state_dim}d{self.control_dim}d{self.control_dim}d"
            )
            self.slot_size = _SLOT_SEQ_STRUCT.size + self._payload_struct.size
            self.total_bytes = _HEADER_SIZE + self.capacity * self.slot_size
            if len(buf) < self.total_bytes:
                self._shm.close()
                raise ValueError("Shared memory segment smaller than expected slot layout")

        self.name: str = self._shm.name

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SharedMemorySensorimotorBus is closed")

    def _slot_offset(self, slot_idx: int) -> int:
        return _HEADER_SIZE + slot_idx * self.slot_size

    def _read_head_index(self) -> int:
        return int(_HEADER_STRUCT.unpack_from(self._shm.buf, 0)[6])

    def publish_frame(
        self,
        *,
        state: Sequence[float],
        u_nominal: Sequence[float],
        u_filtered: Sequence[float],
        min_barrier_slack: float = 0.0,
        tank_energy: float = 1.0,
        watchdog_scale: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Write one sensorimotor frame into the next slot using an even/odd Seqlock."""
        self._ensure_open()

        state_tup = _validate_finite_vector(state, self.state_dim, "state")
        u_nom_tup = _validate_finite_vector(u_nominal, self.control_dim, "u_nominal")
        u_filt_tup = _validate_finite_vector(u_filtered, self.control_dim, "u_filtered")
        slack_f = _validate_finite_scalar(min_barrier_slack, "min_barrier_slack")
        tank_f = _validate_finite_scalar(tank_energy, "tank_energy")
        wd_f = _validate_finite_scalar(watchdog_scale, "watchdog_scale")
        ts_ns = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
        if ts_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

        buf = self._shm.buf
        prev_head = self._read_head_index()
        next_head = prev_head + 1
        slot_idx = (next_head - 1) % self.capacity
        slot_off = self._slot_offset(slot_idx)

        cur_seq = int(_SLOT_SEQ_STRUCT.unpack_from(buf, slot_off)[0])
        odd_seq = cur_seq + 1 if (cur_seq & 1) == 0 else cur_seq + 2
        even_seq = odd_seq + 1

        # 1. Mark slot write-in-progress (odd sequence number)
        _SLOT_SEQ_STRUCT.pack_into(buf, slot_off, odd_seq)

        # 2. Write fixed-width binary payload
        self._payload_struct.pack_into(
            buf,
            slot_off + _SLOT_SEQ_STRUCT.size,
            next_head,
            ts_ns,
            slack_f,
            tank_f,
            wd_f,
            *state_tup,
            *u_nom_tup,
            *u_filt_tup,
        )

        # 3. Publish completed frame (even sequence number), then advance header index
        _SLOT_SEQ_STRUCT.pack_into(buf, slot_off, even_seq)
        struct.pack_into("<Q", buf, _HEADER_SIZE - 8, next_head)

        return {
            "seq": even_seq,
            "frame_index": next_head,
            "slot_index": slot_idx,
            "timestamp_ns": ts_ns,
            "state": list(state_tup),
            "u_nominal": list(u_nom_tup),
            "u_filtered": list(u_filt_tup),
            "min_barrier_slack": slack_f,
            "tank_energy": tank_f,
            "watchdog_scale": wd_f,
        }

    def read_latest(self, max_retries: int = 1024) -> Optional[Dict[str, Any]]:
        """Read the latest consistent frame via Seqlock retry (`seq1 == seq2` and even)."""
        self._ensure_open()
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        buf = self._shm.buf
        sdim = self.state_dim
        cdim = self.control_dim

        for _ in range(max_retries):
            head_before = self._read_head_index()
            if head_before == 0:
                return None

            slot_idx = (head_before - 1) % self.capacity
            slot_off = self._slot_offset(slot_idx)

            seq1 = int(_SLOT_SEQ_STRUCT.unpack_from(buf, slot_off)[0])
            if seq1 == 0 or (seq1 & 1) != 0:
                continue

            raw_tuple = self._payload_struct.unpack_from(buf, slot_off + _SLOT_SEQ_STRUCT.size)
            seq2 = int(_SLOT_SEQ_STRUCT.unpack_from(buf, slot_off)[0])
            head_after = self._read_head_index()

            if seq1 != seq2 or (seq1 & 1) != 0 or head_after != head_before:
                continue

            frame_index = int(raw_tuple[0])
            timestamp_ns = int(raw_tuple[1])
            min_barrier_slack = float(raw_tuple[2])
            tank_energy = float(raw_tuple[3])
            watchdog_scale = float(raw_tuple[4])

            pos_end = 5 + sdim
            unom_end = pos_end + cdim
            ufilt_end = unom_end + cdim

            return {
                "seq": seq1,
                "frame_index": frame_index,
                "slot_index": slot_idx,
                "timestamp_ns": timestamp_ns,
                "state": [float(v) for v in raw_tuple[5:pos_end]],
                "u_nominal": [float(v) for v in raw_tuple[pos_end:unom_end]],
                "u_filtered": [float(v) for v in raw_tuple[unom_end:ufilt_end]],
                "min_barrier_slack": min_barrier_slack,
                "tank_energy": tank_energy,
                "watchdog_scale": watchdog_scale,
            }

        raise RuntimeError(f"Seqlock read failed to observe consistent frame after {max_retries} retries")

    def close(self) -> None:
        """Close this handle to the shared-memory segment."""
        if not self._closed:
            self._shm.close()
            self._closed = True

    def unlink(self) -> None:
        """Unlink (destroy) the underlying shared-memory segment."""
        if not self._unlinked:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._unlinked = True

    def __enter__(self) -> "SharedMemorySensorimotorBus":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
        if self._unlink_on_exit:
            self.unlink()


__all__ = ["SharedMemorySensorimotorBus"]
