"""A Mojo-accelerated, read-focused subset of pysam."""

from __future__ import annotations

import array

from ._lib import lib
from .alignmentfile import AlignmentFile, IndexStats, Samfile
from .header import AlignmentHeader
from .segment import AlignedSegment

__version__ = "0.1.0"


def qualitystring_to_array(qualities: str):
    return array.array("B", (ord(char) - 33 for char in qualities))


def array_to_qualitystring(qualities, offset=33):
    return "".join(chr(int(value) + offset) for value in qualities)


def get_include():
    return ()


__all__ = [
    "AlignedSegment",
    "AlignmentFile",
    "AlignmentHeader",
    "IndexStats",
    "Samfile",
    "array_to_qualitystring",
    "qualitystring_to_array",
]
