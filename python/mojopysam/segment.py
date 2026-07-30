"""AlignedSegment and SAM/BAM field decoding."""

from __future__ import annotations

import array
import re
import struct
from typing import Any

import numpy as np

from ._lib import lib, scan_sam
from .header import AlignmentHeader

CIGAR_OPS = "MIDNSHP=XB"
CIGAR_CODES = {op: index for index, op in enumerate(CIGAR_OPS)}
SEQ_CODES = "=ACMGRSVTWYHKDBN"
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=XB])")
_BAM_SCALARS = {
    "c": ("<b", 1),
    "C": ("<B", 1),
    "s": ("<h", 2),
    "S": ("<H", 2),
    "i": ("<i", 4),
    "I": ("<I", 4),
    "f": ("<f", 4),
    "d": ("<d", 8),
}
_BAM_ARRAY_TYPES = {
    "c": "b",
    "C": "B",
    "s": "h",
    "S": "H",
    "i": "i",
    "I": "I",
    "f": "f",
}


def _parse_cigar(text: str | None) -> list[tuple[int, int]] | None:
    if not text or text == "*":
        return None
    fields = _CIGAR_RE.findall(text)
    result = [(CIGAR_CODES[op], int(length)) for length, op in fields]
    if sum(len(length) + 1 for length, _ in fields) != len(text):
        raise ValueError(f"invalid CIGAR string: {text}")
    return result


def _format_cigar(cigartuples) -> str | None:
    if not cigartuples:
        return None
    return "".join(f"{length}{CIGAR_OPS[op]}" for op, length in cigartuples)


def _decode_sam_tag(field: str):
    tag, value_type, value = field.split(":", 2)
    if value_type in "cCsSiI":
        decoded: Any = int(value)
        if decoded >= 0:
            value_type = "C" if decoded <= 255 else "S" if decoded <= 65535 else "I"
        else:
            value_type = "c" if decoded >= -128 else "s" if decoded >= -32768 else "i"
    elif value_type in "fd":
        decoded = float(value)
        value_type = "f"
    elif value_type == "B":
        subtype, _, body = value.partition(",")
        typecode = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "f": "f"}[subtype]
        convert = float if subtype == "f" else int
        decoded = array.array(
            typecode, (() if not body else (convert(item) for item in body.split(",")))
        )
    else:
        decoded = value
    return tag, decoded, value_type


def _decode_bam_tag(data: bytes, pos: int, end: int):
    if pos + 3 > end:
        raise ValueError("truncated BAM auxiliary field")
    tag = data[pos : pos + 2].decode("ascii")
    kind = chr(data[pos + 2])
    pos += 3
    if kind == "A":
        value = chr(data[pos])
        pos += 1
    elif kind in _BAM_SCALARS:
        fmt, width = _BAM_SCALARS[kind]
        value = struct.unpack_from(fmt, data, pos)[0]
        pos += width
    elif kind in {"Z", "H"}:
        stop = data.find(b"\0", pos, end)
        if stop < 0:
            raise ValueError("unterminated BAM string tag")
        value = data[pos:stop].decode("ascii")
        pos = stop + 1
    elif kind == "B":
        subtype = chr(data[pos])
        count = struct.unpack_from("<I", data, pos + 1)[0]
        pos += 5
        fmt, width = _BAM_SCALARS[subtype]
        values = struct.unpack_from("<" + fmt[-1] * count, data, pos)
        value = array.array(_BAM_ARRAY_TYPES[subtype], values)
        pos += width * count
    else:
        raise ValueError(f"unsupported BAM auxiliary type {kind!r}")
    return (tag, value, kind), pos


def _decode_bam_tags(data: bytes, begin: int, end: int):
    result = []
    pos = begin
    while pos < end:
        item, pos = _decode_bam_tag(data, pos, end)
        result.append(item)
    return result


class AlignedSegment:
    __slots__ = ("header", "_source", "_row", "_format", "_values")

    def __init__(self, header: AlignmentHeader | None = None):
        self.header = header or AlignmentHeader()
        self._source: bytes | None = None
        self._row: np.ndarray | None = None
        self._format: str | None = None
        self._values: dict[str, Any] = {
            "query_name": None,
            "flag": 0,
            "reference_id": -1,
            "reference_start": -1,
            "mapping_quality": 0,
            "cigartuples": None,
            "next_reference_id": -1,
            "next_reference_start": -1,
            "template_length": 0,
            "query_sequence": None,
            "query_qualities": None,
            "tags": [],
        }

    @classmethod
    def _from_row(cls, source: bytes, row: np.ndarray, fmt: str, header: AlignmentHeader):
        segment = cls.__new__(cls)
        segment.header = header
        segment._source = source
        segment._row = row
        segment._format = fmt
        segment._values = {}
        return segment

    @classmethod
    def fromstring(cls, sam: str, header: AlignmentHeader):
        data = sam.encode()
        rows = scan_sam(data)
        if len(rows) != 1:
            raise ValueError("expected exactly one SAM record")
        return cls._from_row(data, rows[0], "sam", header)

    @classmethod
    def from_dict(cls, sam_dict: dict, header: AlignmentHeader):
        record = cls(header)
        record.query_name = sam_dict.get("name")
        record.flag = int(sam_dict.get("flag", 0))
        record.reference_name = sam_dict.get("ref_name")
        record.reference_start = int(sam_dict.get("ref_pos", -1))
        record.mapping_quality = int(sam_dict.get("map_quality", 0))
        record.cigarstring = sam_dict.get("cigar")
        record.next_reference_name = sam_dict.get("next_ref_name")
        record.next_reference_start = int(sam_dict.get("next_ref_pos", -1))
        record.template_length = int(sam_dict.get("length", 0))
        record.query_sequence = sam_dict.get("seq")
        record.query_qualities = sam_dict.get("qual")
        record.tags = sam_dict.get("tags", [])
        return record

    def _sam_text(self, begin: int, end: int) -> str:
        return self._source[begin:end].decode("ascii")

    @property
    def query_name(self):
        if self._format == "sam":
            return self._sam_text(int(self._row[0]), int(self._row[2]))
        if self._format == "bam":
            begin, length = int(self._row[13]), int(self._row[4])
            return self._source[begin : begin + length - 1].decode("ascii")
        return self._values["query_name"]

    @query_name.setter
    def query_name(self, value):
        self._values["query_name"] = value

    qname = query_name

    @property
    def flag(self):
        if self._format == "sam":
            return int(self._row[3])
        if self._format == "bam":
            return int(self._row[8])
        return self._values["flag"]

    @flag.setter
    def flag(self, value):
        self._values["flag"] = int(value)

    @property
    def reference_id(self):
        if self._format == "bam":
            return int(self._row[2])
        if self._format == "sam":
            return self.header.get_tid(self.reference_name)
        return self._values["reference_id"]

    @reference_id.setter
    def reference_id(self, value):
        self._values["reference_id"] = int(value)

    tid = reference_id

    @property
    def reference_name(self):
        if self._format == "sam":
            value = self._sam_text(int(self._row[4]), int(self._row[5]))
            return None if value == "*" else value
        return self.header.get_reference_name(self.reference_id)

    @reference_name.setter
    def reference_name(self, value):
        self.reference_id = self.header.get_tid(value) if value not in {None, "*"} else -1

    @property
    def reference_start(self):
        if self._format == "sam":
            return int(self._row[6])
        if self._format == "bam":
            return int(self._row[3])
        return self._values["reference_start"]

    @reference_start.setter
    def reference_start(self, value):
        self._values["reference_start"] = int(value)

    pos = reference_start

    @property
    def mapping_quality(self):
        if self._format == "sam":
            return int(self._row[7])
        if self._format == "bam":
            return int(self._row[5])
        return self._values["mapping_quality"]

    @mapping_quality.setter
    def mapping_quality(self, value):
        self._values["mapping_quality"] = int(value)

    mapq = mapping_quality

    @property
    def cigarstring(self):
        if self._format == "sam":
            value = self._sam_text(int(self._row[8]), int(self._row[9]))
            return None if value == "*" else value
        return _format_cigar(self.cigartuples)

    @cigarstring.setter
    def cigarstring(self, value):
        self._values["cigartuples"] = _parse_cigar(value)

    @property
    def cigartuples(self):
        if self._format == "sam":
            return _parse_cigar(self.cigarstring)
        if self._format == "bam":
            begin, count = int(self._row[14]), int(self._row[7])
            return [
                (packed & 15, packed >> 4)
                for (packed,) in struct.iter_unpack(
                    "<I", self._source[begin : begin + count * 4]
                )
            ] or None
        return self._values["cigartuples"]

    @cigartuples.setter
    def cigartuples(self, value):
        self._values["cigartuples"] = None if value is None else list(value)

    cigar = cigartuples

    @property
    def next_reference_id(self):
        if self._format == "bam":
            return int(self._row[10])
        if self._format == "sam":
            name = self.next_reference_name
            return self.reference_id if name == self.reference_name and name is not None else self.header.get_tid(name)
        return self._values["next_reference_id"]

    @next_reference_id.setter
    def next_reference_id(self, value):
        self._values["next_reference_id"] = int(value)

    mrnm = next_reference_id

    @property
    def next_reference_name(self):
        if self._format == "sam":
            value = self._sam_text(int(self._row[10]), int(self._row[11]))
            if value == "*":
                return None
            return self.reference_name if value == "=" else value
        return self.header.get_reference_name(self.next_reference_id)

    @next_reference_name.setter
    def next_reference_name(self, value):
        self.next_reference_id = self.header.get_tid(value) if value not in {None, "*"} else -1

    @property
    def next_reference_start(self):
        if self._format == "sam":
            return int(self._row[12])
        if self._format == "bam":
            return int(self._row[11])
        return self._values["next_reference_start"]

    @next_reference_start.setter
    def next_reference_start(self, value):
        self._values["next_reference_start"] = int(value)

    pnext = next_reference_start

    @property
    def template_length(self):
        if self._format == "sam":
            return int(self._row[13])
        if self._format == "bam":
            return int(self._row[12])
        return self._values["template_length"]

    @template_length.setter
    def template_length(self, value):
        self._values["template_length"] = int(value)

    tlen = template_length

    @property
    def query_sequence(self):
        if self._format == "sam":
            value = self._sam_text(int(self._row[14]), int(self._row[15]))
            return None if value == "*" else value
        if self._format == "bam":
            begin, length = int(self._row[15]), int(self._row[9])
            chars = []
            for index in range(length):
                packed = self._source[begin + index // 2]
                code = packed >> 4 if index % 2 == 0 else packed & 15
                chars.append(SEQ_CODES[code])
            return "".join(chars)
        return self._values["query_sequence"]

    @query_sequence.setter
    def query_sequence(self, value):
        self._values["query_sequence"] = value

    seq = query_sequence

    @property
    def query_qualities(self):
        if self._format == "sam":
            text = self._sam_text(int(self._row[16]), int(self._row[17]))
            return None if text == "*" else array.array("B", (ord(char) - 33 for char in text))
        if self._format == "bam":
            begin, length = int(self._row[16]), int(self._row[9])
            values = self._source[begin : begin + length]
            return None if values and values[0] == 255 else array.array("B", values)
        value = self._values["query_qualities"]
        return None if value is None else array.array("B", value)

    @query_qualities.setter
    def query_qualities(self, value):
        self._values["query_qualities"] = None if value is None else array.array("B", value)

    qual = query_qualities

    def _tag_tuples(self):
        if self._format == "sam":
            begin, end = int(self._row[18]), int(self._row[1])
            if begin >= end:
                return []
            return [_decode_sam_tag(field) for field in self._sam_text(begin, end).split("\t")]
        if self._format == "bam":
            return _decode_bam_tags(self._source, int(self._row[17]), int(self._row[1]))
        result = []
        for item in self._values["tags"]:
            result.append(tuple(item) if len(item) == 3 else (item[0], item[1], None))
        return result

    @property
    def tags(self):
        return [(tag, value) for tag, value, _ in self._tag_tuples()]

    @tags.setter
    def tags(self, values):
        self._values["tags"] = list(values)

    def get_tags(self, with_value_type=False):
        tags = self._tag_tuples()
        return tags if with_value_type else [(tag, value) for tag, value, _ in tags]

    def get_tag(self, tag, with_value_type=False):
        if self._format == "sam":
            encoded = tag.encode("ascii")
            pos, end = int(self._row[18]), int(self._row[1])
            while pos < end:
                field_end = self._source.find(b"\t", pos, end)
                if field_end < 0:
                    field_end = end
                if (
                    field_end - pos >= 5
                    and self._source[pos : pos + 2] == encoded
                    and self._source[pos + 2] == 58
                ):
                    value_type = chr(self._source[pos + 3])
                    value_begin = pos + 5
                    if value_type in "cCsSiI":
                        value = int(self._source[value_begin:field_end])
                        if value >= 0:
                            value_type = (
                                "C" if value <= 255 else "S" if value <= 65535 else "I"
                            )
                        else:
                            value_type = (
                                "c" if value >= -128 else "s" if value >= -32768 else "i"
                            )
                    elif value_type in "fd":
                        value = float(self._source[value_begin:field_end])
                        value_type = "f"
                    elif value_type == "A":
                        value = chr(self._source[value_begin])
                    elif value_type in {"Z", "H"}:
                        value = self._source[value_begin:field_end].decode("ascii")
                    else:
                        _, value, value_type = _decode_sam_tag(
                            self._source[pos:field_end].decode("ascii")
                        )
                    return (value, value_type) if with_value_type else value
                pos = field_end + 1
            raise KeyError(f"tag {tag!r} not present")
        if self._format == "bam":
            pos, end = int(self._row[17]), int(self._row[1])
            while pos < end:
                item, pos = _decode_bam_tag(self._source, pos, end)
                key, value, value_type = item
                if key == tag:
                    return (value, value_type) if with_value_type else value
            raise KeyError(f"tag {tag!r} not present")
        for key, value, value_type in self._tag_tuples():
            if key == tag:
                return (value, value_type) if with_value_type else value
        raise KeyError(f"tag {tag!r} not present")

    def has_tag(self, tag):
        return any(key == tag for key, _, _ in self._tag_tuples())

    def set_tag(self, tag, value, value_type=None, replace=True):
        tags = self.get_tags(with_value_type=True)
        if replace:
            tags = [item for item in tags if item[0] != tag]
        if value is not None:
            tags.append((tag, value, value_type))
        self._source = self._row = self._format = None
        self._values["tags"] = tags

    @property
    def reference_end(self):
        if self._format == "sam":
            start = int(self._row[6])
            if start < 0:
                return None
            begin, end = int(self._row[8]), int(self._row[9])
            if end - begin == 1 and self._source[begin] == 42:
                return None
            span = int(self._row[19])
        elif self._format == "bam":
            start = int(self._row[3])
            if start < 0 or int(self._row[7]) == 0:
                return None
            span = int(self._row[18])
        else:
            start = self.reference_start
            if start < 0 or not self.cigartuples:
                return None
            span = sum(length for op, length in self.cigartuples if op in {0, 2, 3, 7, 8})
        return start + span

    aend = reference_end

    @property
    def bin(self):
        if self._format == "bam":
            return int(self._row[6])
        end = self.reference_end
        if self.reference_start < 0 or end is None:
            return 4680
        return int(lib().mp_reg2bin(self.reference_start, end))

    @property
    def query_length(self):
        if self._format == "sam":
            begin, end = int(self._row[14]), int(self._row[15])
            return None if end - begin == 1 and self._source[begin] == 42 else end - begin
        if self._format == "bam":
            return int(self._row[9])
        sequence = self.query_sequence
        return len(sequence) if sequence is not None else self.infer_query_length()

    @property
    def query_alignment_start(self):
        if self._format == "sam":
            return int(self._row[21])
        if self._format == "bam":
            return int(self._row[20])
        cigar = self.cigartuples or []
        return cigar[0][1] if cigar and cigar[0][0] == 4 else 0

    @property
    def query_alignment_end(self):
        length = self.query_length
        if length is None:
            return None
        if self._format == "sam":
            trailing = int(self._row[22])
        elif self._format == "bam":
            trailing = int(self._row[21])
        else:
            cigar = self.cigartuples or []
            trailing = cigar[-1][1] if cigar and cigar[-1][0] == 4 else 0
        return length - trailing

    @property
    def query_alignment_length(self):
        if self._format == "sam":
            length = self.query_length
            return None if length is None else length - int(self._row[21]) - int(self._row[22])
        if self._format == "bam":
            return int(self._row[9]) - int(self._row[20]) - int(self._row[21])
        end = self.query_alignment_end
        return None if end is None else end - self.query_alignment_start

    @property
    def query_alignment_sequence(self):
        sequence = self.query_sequence
        return None if sequence is None else sequence[self.query_alignment_start : self.query_alignment_end]

    @property
    def query_alignment_qualities(self):
        qualities = self.query_qualities
        return None if qualities is None else qualities[self.query_alignment_start : self.query_alignment_end]

    def infer_query_length(self, always=False):
        cigar = self.cigartuples
        if not cigar:
            return None
        accepted = {0, 1, 4, 7, 8} | ({5} if always else set())
        return sum(length for op, length in cigar if op in accepted)

    def get_reference_positions(self, full_length=False):
        result = []
        ref_pos = self.reference_start
        for op, length in self.cigartuples or []:
            if op in {0, 7, 8}:
                result.extend(range(ref_pos, ref_pos + length))
                ref_pos += length
            elif op in {2, 3}:
                ref_pos += length
            elif full_length and op in {1, 4}:
                result.extend([None] * length)
        return result

    def get_blocks(self):
        result = []
        ref_pos = self.reference_start
        for op, length in self.cigartuples or []:
            if op in {0, 7, 8}:
                result.append((ref_pos, ref_pos + length))
                ref_pos += length
            elif op in {2, 3}:
                ref_pos += length
        return result

    def get_overlap(self, start, end):
        if not self.cigartuples:
            return None
        overlap = 0
        for block_start, block_end in self.get_blocks():
            overlap += max(0, min(block_end, end) - max(block_start, start))
        return overlap

    def get_forward_sequence(self):
        sequence = self.query_sequence
        if sequence is None or not self.is_reverse:
            return sequence
        complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        return sequence.translate(complement)[::-1]

    def get_forward_qualities(self):
        qualities = self.query_qualities
        if qualities is None or not self.is_reverse:
            return qualities
        return array.array("B", reversed(qualities))

    def get_aligned_pairs(self, matches_only=False, with_seq=False, with_cigar=False):
        if with_seq:
            raise NotImplementedError("with_seq requires MD-tag reconstruction")
        result = []
        query_pos = 0
        ref_pos = self.reference_start
        for op, length in self.cigartuples or []:
            if op in {0, 7, 8}:
                for _ in range(length):
                    item = (query_pos, ref_pos, op) if with_cigar else (query_pos, ref_pos)
                    result.append(item)
                    query_pos += 1
                    ref_pos += 1
            elif op == 1:
                if not matches_only:
                    for _ in range(length):
                        result.append((query_pos, None, op) if with_cigar else (query_pos, None))
                        query_pos += 1
                else:
                    query_pos += length
            elif op in {2, 3}:
                if not matches_only:
                    for _ in range(length):
                        result.append((None, ref_pos, op) if with_cigar else (None, ref_pos))
                        ref_pos += 1
                else:
                    ref_pos += length
            elif op == 4:
                if not matches_only:
                    for _ in range(length):
                        result.append((query_pos, None, op) if with_cigar else (query_pos, None))
                        query_pos += 1
                else:
                    query_pos += length
        return result

    def get_cigar_stats(self):
        lengths = array.array("I", [0] * 11)
        blocks = array.array("I", [0] * 11)
        for op, length in self.cigartuples or []:
            if op < 10:
                lengths[op] += length
                blocks[op] += 1
        if self.has_tag("NM"):
            lengths[10] = int(self.get_tag("NM"))
        return lengths, blocks

    def to_string(self):
        mate = "*"
        if self.next_reference_name is not None:
            mate = "=" if self.next_reference_name == self.reference_name else self.next_reference_name
        qualities = self.query_qualities
        qual = "*" if qualities is None else "".join(chr(value + 33) for value in qualities)
        fields = [
            self.query_name or "*",
            str(self.flag),
            self.reference_name or "*",
            str(self.reference_start + 1 if self.reference_start >= 0 else 0),
            str(self.mapping_quality),
            self.cigarstring or "*",
            mate,
            str(self.next_reference_start + 1 if self.next_reference_start >= 0 else 0),
            str(self.template_length),
            self.query_sequence or "*",
            qual,
        ]
        for tag, value, kind in self._tag_tuples():
            if kind is None:
                if isinstance(value, int):
                    kind = "i"
                elif isinstance(value, float):
                    kind = "f"
                else:
                    kind = "Z"
            if kind == "B":
                subtype = getattr(value, "typecode", "i")
                fields.append(f"{tag}:B:{subtype}," + ",".join(map(str, value)))
            else:
                if kind in "cCsSI":
                    kind = "i"
                fields.append(f"{tag}:{kind}:{value}")
        return "\t".join(fields)

    tostring = to_string

    def to_dict(self):
        return {
            "name": self.query_name,
            "flag": str(self.flag),
            "ref_name": self.reference_name,
            "ref_pos": str(self.reference_start),
            "map_quality": str(self.mapping_quality),
            "cigar": self.cigarstring,
            "next_ref_name": self.next_reference_name,
            "next_ref_pos": str(self.next_reference_start),
            "length": str(self.template_length),
            "seq": self.query_sequence,
            "qual": self.query_qualities,
            "tags": self.tags,
        }

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"<mojopysam.AlignedSegment({self.query_name!r}, flags={self.flag})>"


for _name, _mask in {
    "is_paired": 0x1,
    "is_proper_pair": 0x2,
    "is_unmapped": 0x4,
    "mate_is_unmapped": 0x8,
    "is_reverse": 0x10,
    "mate_is_reverse": 0x20,
    "is_read1": 0x40,
    "is_read2": 0x80,
    "is_secondary": 0x100,
    "is_qcfail": 0x200,
    "is_duplicate": 0x400,
    "is_supplementary": 0x800,
}.items():
    setattr(
        AlignedSegment,
        _name,
        property(
            lambda self, mask=_mask: bool(self.flag & mask),
            lambda self, value, mask=_mask: setattr(
                self, "flag", self.flag | mask if value else self.flag & ~mask
            ),
        ),
    )

AlignedSegment.is_forward = property(lambda self: not self.is_reverse)
AlignedSegment.mate_is_forward = property(lambda self: not self.mate_is_reverse)
AlignedSegment.is_mapped = property(lambda self: not self.is_unmapped)
