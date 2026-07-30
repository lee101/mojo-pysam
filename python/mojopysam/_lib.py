"""ctypes bridge to the Mojo parsing kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJOPYSAM_LIB", os.path.join(ROOT, "dist", "libmojo-pysam.so")
)

I = ctypes.c_int64

_SIGNATURES = {
    "mp_scan_sam": ([I, I, I, I], I),
    "mp_scan_bam": ([I, I, I, I], I),
    "mp_reg2bin": ([I, I], I),
    "mp_count_coverage": ([I] * 15, I),
    "mp_count_coverage_bam": ([I] * 12, I),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if os.path.exists(LIB_PATH):
        return LIB_PATH
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB_PATH):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous NumPy arrays")
    if array.size and not array.flags.aligned:
        raise ValueError("native buffers must be aligned")
    return int(array.ctypes.data)


def byte_array(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


def scan_sam(data: bytes) -> np.ndarray:
    source = byte_array(data)
    capacity = data.count(b"\n") + (not data.endswith(b"\n"))
    rows = np.empty((max(capacity, 1), 24), dtype=np.int64)
    if not capacity:
        return rows[:0]
    count = lib().mp_scan_sam(addr(source), source.size, addr(rows), capacity)
    if count < 0:
        messages = {
            -1: "record output capacity exceeded",
            -2: "SAM record has fewer than 11 fields",
            -3: "invalid SAM buffer dimensions",
            -4: "null SAM buffer address",
        }
        raise ValueError(messages.get(count, f"SAM scan failed ({count})"))
    return rows[:count]


def scan_bam(data: bytes) -> np.ndarray:
    source = byte_array(data)
    capacity = max(1, len(data) // 36 + 1)
    rows = np.empty((capacity, 22), dtype=np.int64)
    count = lib().mp_scan_bam(addr(source), source.size, addr(rows), capacity)
    if count < 0:
        messages = {
            -1: "record output capacity exceeded",
            -2: "truncated BAM record core",
            -3: "invalid BAM block size",
            -4: "invalid BAM variable-length fields",
            -5: "invalid BAM buffer dimensions",
            -6: "null BAM buffer address",
        }
        raise ValueError(messages.get(count, f"BAM scan failed ({count})"))
    return rows[:count]
