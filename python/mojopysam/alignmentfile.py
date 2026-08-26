"""Read-only AlignmentFile for SAM, BAM, and CRAM."""

from __future__ import annotations

import gzip
import os
import struct
from collections import namedtuple
from collections.abc import Iterator

import numpy as np

from ._lib import addr, lib, scan_bam, scan_sam
from .header import AlignmentHeader
from .segment import AlignedSegment

IndexStats = namedtuple("IndexStats", "contig mapped unmapped total")


def _sam_header(data: bytes) -> AlignmentHeader:
    lines = []
    for line in data.splitlines():
        if not line.startswith(b"@"):
            break
        lines.append(line.decode("ascii"))
    return AlignmentHeader.from_text("\n".join(lines))


def _bam_payload(data: bytes):
    if data[:4] != b"BAM\1":
        raise ValueError("invalid BAM magic")
    if len(data) < 12:
        raise ValueError("truncated BAM header")
    text_length = struct.unpack_from("<i", data, 4)[0]
    if text_length < 0 or text_length > len(data) - 12:
        raise ValueError("invalid BAM header text length")
    pos = 8
    text = data[pos : pos + text_length].rstrip(b"\0").decode("ascii")
    pos += text_length
    if pos + 4 > len(data):
        raise ValueError("truncated BAM reference count")
    reference_count = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    if reference_count < 0:
        raise ValueError("invalid BAM reference count")
    references = []
    lengths = []
    for _ in range(reference_count):
        if pos + 4 > len(data):
            raise ValueError("truncated BAM reference name length")
        name_length = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if name_length < 1 or name_length > len(data) - pos - 4:
            raise ValueError("invalid BAM reference name length")
        if data[pos + name_length - 1] != 0:
            raise ValueError("BAM reference name is not NUL-terminated")
        references.append(data[pos : pos + name_length - 1].decode("ascii"))
        pos += name_length
        reference_length = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        if reference_length < 0:
            raise ValueError("invalid BAM reference length")
        lengths.append(reference_length)
    header = AlignmentHeader.from_text(text)
    if header.references != tuple(references) or header.lengths != tuple(lengths):
        header = AlignmentHeader(text if text.endswith("\n") else text + "\n", references, lengths)
    return header, data[pos:]


def _parse_region(region: str):
    if ":" not in region:
        return region, None, None
    reference, coordinates = region.rsplit(":", 1)
    if "-" in coordinates:
        begin, end = coordinates.replace(",", "").split("-", 1)
        return reference, int(begin) - 1, int(end)
    return reference, int(coordinates.replace(",", "")) - 1, None


class AlignmentFile:
    def __init__(
        self,
        filename,
        mode=None,
        template=None,
        reference_names=None,
        reference_lengths=None,
        text=None,
        header=None,
        add_sq_text=False,
        check_header=True,
        check_sq=True,
        reference_filename=None,
        filepath_index=None,
        require_index=False,
        duplicate_filehandle=True,
        ignore_truncation=False,
        threads=1,
        format_options=None,
        **kwargs,
    ):
        del add_sq_text, check_header, check_sq, filepath_index, duplicate_filehandle
        del ignore_truncation, threads, format_options, kwargs
        self.filename = os.fspath(filename) if not hasattr(filename, "read") else getattr(filename, "name", "-")
        self.mode = mode or "r"
        if any(char in self.mode for char in "wax"):
            raise NotImplementedError("mojopysam AlignmentFile currently supports reading only")
        if require_index:
            raise ValueError("mojopysam scans in memory and does not consume BAM/CRAM indexes")

        self.closed = False
        self._cursor = 0
        self._native = None
        self._source = b""
        self._rows = np.empty((0, 0), dtype=np.int64)
        self._row_buffer = None
        self._format = ""

        if hasattr(filename, "read"):
            raw = filename.read()
            if isinstance(raw, str):
                raw = raw.encode()
        else:
            with open(filename, "rb") as handle:
                raw = handle.read()

        requested = self.mode.replace("b", "").replace("c", "")
        if raw.startswith(b"CRAM") or "c" in self.mode or str(self.filename).lower().endswith(".cram"):
            if hasattr(filename, "read"):
                raise NotImplementedError("CRAM file objects are not supported by the codec backend")
            try:
                import pysam as upstream
            except ImportError as error:
                raise ImportError("CRAM decoding requires the optional pysam codec backend") from error
            self._native = upstream.AlignmentFile(
                filename, mode or "rc", reference_filename=reference_filename
            )
            self.header = AlignmentHeader.from_text(str(self._native.header))
            self._format = "cram"
            self._native_records = list(self._native.fetch(until_eof=True))
        elif raw.startswith(b"\x1f\x8b") or "b" in self.mode or str(self.filename).lower().endswith(".bam"):
            decompressed = gzip.decompress(raw)
            self.header, self._source = _bam_payload(decompressed)
            self._rows = scan_bam(self._source)
            self._format = "bam"
        else:
            self._source = raw
            self.header = (
                AlignmentHeader.from_dict(header)
                if isinstance(header, dict)
                else header
                or (template.header.copy() if template is not None else None)
                or (AlignmentHeader.from_text(text) if text is not None else None)
                or (
                    AlignmentHeader.from_references(reference_names, reference_lengths)
                    if reference_names is not None
                    else _sam_header(raw)
                )
            )
            self._rows = scan_sam(raw)
            self._format = "sam"
        if self._rows.size:
            self._row_buffer = memoryview(self._rows).cast("B").cast("q")
        self._count = (
            len(self._native_records) if self._format == "cram" else len(self._rows)
        )
        _ = requested

    @property
    def is_sam(self):
        return self._format == "sam"

    @property
    def is_bam(self):
        return self._format == "bam"

    @property
    def is_cram(self):
        return self._format == "cram"

    @property
    def references(self):
        return self.header.references

    @property
    def lengths(self):
        return self.header.lengths

    @property
    def nreferences(self):
        return self.header.nreferences

    @property
    def text(self):
        return str(self.header)

    @property
    def mapped(self):
        return sum(not record.is_unmapped for record in self._records())

    @property
    def unmapped(self):
        return sum(record.is_unmapped for record in self._records())

    @property
    def nocoordinate(self):
        return self.unmapped

    def get_reference_name(self, tid):
        return self.header.get_reference_name(tid)

    def get_tid(self, reference):
        return self.header.get_tid(reference)

    def has_index(self):
        return False

    def check_index(self):
        raise ValueError("mojopysam performs in-memory scans and does not load index files")

    def get_index_statistics(self):
        stats = []
        for reference in self.references:
            records = [record for record in self._records() if record.reference_name == reference]
            mapped = sum(not record.is_unmapped for record in records)
            unmapped = sum(record.is_unmapped for record in records)
            stats.append(IndexStats(reference, mapped, unmapped, mapped + unmapped))
        return stats

    def _records(self):
        if self.closed:
            raise ValueError("I/O operation on closed AlignmentFile")
        if self._format == "cram":
            for native in self._native_records:
                yield AlignedSegment.fromstring(native.to_string(), self.header)
        else:
            row_width = self._rows.shape[1]
            for index in range(self._count):
                begin = index * row_width
                yield AlignedSegment._from_row(
                    self._source,
                    self._row_buffer[begin : begin + row_width],
                    self._format,
                    self.header,
                )

    def __iter__(self) -> Iterator[AlignedSegment]:
        return self

    def __next__(self):
        if self._cursor >= self._count:
            raise StopIteration
        if self._format == "cram":
            native = self._native_records[self._cursor]
            record = AlignedSegment.fromstring(native.to_string(), self.header)
        else:
            row_width = self._rows.shape[1]
            begin = self._cursor * row_width
            record = AlignedSegment._from_row(
                self._source,
                self._row_buffer[begin : begin + row_width],
                self._format,
                self.header,
            )
        self._cursor += 1
        return record

    def _record_count(self):
        return self._count

    def fetch(
        self,
        contig=None,
        start=None,
        stop=None,
        region=None,
        tid=None,
        until_eof=False,
        multiple_iterators=False,
        reference=None,
        end=None,
    ):
        del until_eof, multiple_iterators
        if reference is not None:
            contig = reference
        if end is not None:
            stop = end
        if region is not None:
            contig, region_start, region_stop = _parse_region(region)
            start = region_start if start is None else start
            stop = region_stop if stop is None else stop
        if tid is not None:
            contig = self.get_reference_name(tid)
        if isinstance(contig, int):
            contig = self.get_reference_name(contig)
        if contig is not None and self.get_tid(contig) < 0:
            raise ValueError(f"invalid contig {contig!r}")
        begin = 0 if start is None else int(start)
        finish = None if stop is None else int(stop)
        for record in self._records():
            if contig is not None and record.reference_name != contig:
                continue
            record_end = record.reference_end
            if record_end is None:
                if contig is None:
                    yield record
                continue
            if record_end <= begin:
                continue
            if finish is not None and record.reference_start >= finish:
                continue
            yield record

    def count(self, contig=None, start=None, stop=None, region=None, until_eof=False, read_callback="nofilter", reference=None, end=None):
        records = self.fetch(
            contig=contig,
            start=start,
            stop=stop,
            region=region,
            until_eof=until_eof,
            reference=reference,
            end=end,
        )
        if read_callback == "all":
            records = (
                record
                for record in records
                if not (record.flag & (0x4 | 0x100 | 0x200 | 0x400))
            )
        elif callable(read_callback):
            records = filter(read_callback, records)
        return sum(1 for _ in records)

    def count_coverage(
        self,
        contig,
        start=None,
        stop=None,
        region=None,
        quality_threshold=15,
        read_callback="all",
        reference=None,
        end=None,
    ):
        if reference is not None:
            contig = reference
        if end is not None:
            stop = end
        if region is not None:
            contig, start, stop = _parse_region(region)
        if start is None:
            start = 0
        tid = self.get_tid(contig)
        if tid < 0:
            raise ValueError(f"invalid contig {contig!r}")
        if stop is None:
            stop = self.lengths[tid]
        start, stop = int(start), int(stop)
        if stop < start:
            raise ValueError("stop must not be smaller than start")
        if (
            self._format == "bam"
            and read_callback in {"all", "nofilter"}
            and stop > start
        ):
            counts = np.zeros((4, stop - start), dtype=np.int64)
            filter_flags = 0x4 | 0x100 | 0x200 | 0x400 if read_callback == "all" else 0
            source = np.frombuffer(self._source, dtype=np.uint8)
            status = lib().mp_count_coverage_bam(
                addr(source),
                source.size,
                addr(self._rows),
                self._rows.size,
                len(self._rows),
                tid,
                start,
                stop,
                int(quality_threshold),
                filter_flags,
                addr(counts),
                counts.size,
            )
            if status:
                raise RuntimeError(f"native BAM coverage failed ({status})")
            return tuple(counts[index].astype(np.uint64, copy=False) for index in range(4))
        selected = list(self.fetch(contig, start, stop))
        if read_callback == "all":
            selected = [
                record
                for record in selected
                if not (record.flag & (0x4 | 0x100 | 0x200 | 0x400))
            ]
        elif callable(read_callback):
            selected = [record for record in selected if read_callback(record)]
        elif read_callback != "nofilter":
            raise ValueError("read_callback must be 'all', 'nofilter', or callable")

        starts = np.asarray([record.reference_start for record in selected], dtype=np.int64)
        cigar_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
        seq_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
        packed_cigars = []
        packed_sequences = []
        base_codes = {"A": 0, "C": 1, "G": 2, "T": 3}
        for index, record in enumerate(selected):
            cigar = record.cigartuples or []
            packed_cigars.extend((length << 4) | op for op, length in cigar)
            cigar_offsets[index + 1] = len(packed_cigars)
            sequence = record.query_sequence or ""
            qualities = record.query_qualities
            for query_index, base in enumerate(sequence):
                passes = qualities is None or qualities[query_index] >= quality_threshold
                packed_sequences.append(base_codes.get(base.upper(), 4) if passes else 4)
            seq_offsets[index + 1] = len(packed_sequences)
        cigars = np.asarray(packed_cigars, dtype=np.uint32)
        sequences = np.asarray(packed_sequences, dtype=np.uint8)
        counts = np.zeros((4, stop - start), dtype=np.int64)
        if selected and stop > start:
            status = lib().mp_count_coverage(
                addr(starts),
                starts.size,
                addr(cigar_offsets),
                cigar_offsets.size,
                addr(cigars),
                cigars.size,
                addr(seq_offsets),
                seq_offsets.size,
                addr(sequences),
                sequences.size,
                len(selected),
                start,
                stop,
                addr(counts),
                counts.size,
            )
            if status:
                raise RuntimeError(f"native coverage failed ({status})")
        return tuple(counts[index].astype(np.uint64, copy=False) for index in range(4))

    def head(self, n, multiple_iterators=True):
        del multiple_iterators
        iterator = self._records()
        for _, record in zip(range(n), iterator):
            yield record

    def reset(self):
        self._cursor = 0
        if self._native is not None:
            self._native.reset()

    def tell(self):
        return self._cursor

    def seek(self, offset):
        offset = int(offset)
        if not 0 <= offset <= self._record_count():
            raise ValueError("record offset out of range")
        self._cursor = offset
        return offset

    def close(self):
        if self._native is not None:
            self._native.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __repr__(self):
        return f"<mojopysam.AlignmentFile {self.filename!r} mode={self.mode!r}>"


Samfile = AlignmentFile
