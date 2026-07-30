"""End-to-end parser benchmarks against pysam on identical files."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojopysam as mp  # noqa: E402
import pysam  # noqa: E402


RECORDS = 100_000
SEQUENCE = "ACGT" * 25
QUALITY = "I" * 100
CIGAR = "5S40M2I30M3D23M"


def make_inputs(root: Path):
    sam = root / "bench.sam"
    bam = root / "bench.bam"
    reference_length = RECORDS * 2 + 200
    with sam.open("w") as handle:
        handle.write(f"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:{reference_length}\n")
        for index in range(RECORDS):
            handle.write(
                f"read{index}\t0\tchr1\t{index * 2 + 1}\t60\t{CIGAR}\t*\t0\t0\t"
                f"{SEQUENCE}\t{QUALITY}\tNM:i:3\n"
            )
    with pysam.AlignmentFile(sam, "r") as source:
        with pysam.AlignmentFile(bam, "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.index(str(bam))
    return sam, bam


def timeit(function, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def mojo_checksum(path):
    with mp.AlignmentFile(path) as alignment:
        return sum(
            record.reference_end + record.mapping_quality + record.get_tag("NM")
            for record in alignment
        )


def pysam_checksum(path):
    with pysam.AlignmentFile(path) as alignment:
        return sum(
            record.reference_end + record.mapping_quality + record.get_tag("NM")
            for record in alignment.fetch(until_eof=True)
        )


def mojo_cigar(path):
    with mp.AlignmentFile(path) as alignment:
        records = list(alignment)
        return sum(
            record.reference_end
            + record.query_alignment_length
            + len(record.get_reference_positions())
            for record in records
        )


def pysam_cigar(path):
    with pysam.AlignmentFile(path) as alignment:
        records = list(alignment.fetch(until_eof=True))
        return sum(
            record.reference_end
            + record.query_alignment_length
            + len(record.get_reference_positions())
            for record in records
        )


def mojo_coverage(path):
    with mp.AlignmentFile(path) as alignment:
        return sum(
            int(values.sum())
            for values in alignment.count_coverage(
                "chr1", 0, RECORDS * 2 + 100, quality_threshold=0
            )
        )


def pysam_coverage(path):
    with pysam.AlignmentFile(path) as alignment:
        return sum(
            int(sum(values))
            for values in alignment.count_coverage(
                "chr1", 0, RECORDS * 2 + 100, quality_threshold=0
            )
        )


def machine():
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    with tempfile.TemporaryDirectory(prefix="mojo-pysam-bench-") as directory:
        sam, bam = make_inputs(Path(directory))
        cases = [
            ("SAM parse + fields, 100k", lambda: mojo_checksum(sam), lambda: pysam_checksum(sam)),
            ("BAM parse + fields, 100k", lambda: mojo_checksum(bam), lambda: pysam_checksum(bam)),
            ("CIGAR coordinate methods, 100k", lambda: mojo_cigar(bam), lambda: pysam_cigar(bam)),
            ("count_coverage, 100k x 100 bp", lambda: mojo_coverage(bam), lambda: pysam_coverage(bam)),
        ]
        rows = []
        for name, ours, theirs in cases:
            ours()  # load the shared library outside the measurement
            mojo_time, mojo_result = timeit(ours)
            pysam_time, pysam_result = timeit(theirs)
            if mojo_result != pysam_result:
                raise AssertionError(f"benchmark parity failed for {name}")
            rows.append((name, mojo_time, pysam_time, pysam_time / mojo_time))

        print(f"Machine: {machine()}")
        print()
        print("| Case | mojo-pysam | pysam | Ratio | Result |")
        print("|---|---:|---:|---:|---|")
        for name, mojo_time, pysam_time, ratio in rows:
            result = "faster" if ratio > 1 else "slower"
            print(
                f"| {name} | {mojo_time * 1e3:.2f} ms | {pysam_time * 1e3:.2f} ms "
                f"| {ratio:.2f}x | {result} |"
            )


if __name__ == "__main__":
    main()

