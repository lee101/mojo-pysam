# mojo-pysam

`mojo-pysam` is a read-focused port of useful `pysam` alignment functionality
to Mojo. It provides a Python API shaped like pysam's `AlignmentFile`,
`AlignmentHeader`, and `AlignedSegment`, while using compiled Mojo kernels for
record scanning, BAM core decoding, CIGAR coordinate summaries, binning, and
base coverage.

The project is useful without pysam for SAM and BAM input. CRAM uses pysam as
an optional codec backend because a correct CRAM implementation requires the
full reference, compression, and codec stack; pretending that a small parser
covered that format would be misleading. The Pixi development environment
installs pysam both for CRAM and for parity tests.

## Coverage

Implemented:

- read-only SAM and BAM loading, including concatenated BGZF decompression;
- CRAM loading through the pysam codec backend;
- `AlignmentFile` iteration, `fetch`, `count`, `count_coverage`, `head`,
  `reset`, reference lookup, context management, and format/header properties;
- `AlignmentHeader` construction from text, dictionaries, or reference lists;
- common `AlignedSegment` fields, SAM/BAM auxiliary tags, flag properties,
  CIGAR tuples and strings, reference/query coordinates, blocks, overlap,
  aligned pairs, CIGAR statistics, forward sequence/qualities, SAM conversion,
  and quality string helpers;
- in-memory coordinate filtering when no BAM index is consumed.

Not implemented:

- SAM/BAM/CRAM writing;
- BAI/CSI/CRAI indexed random I/O (region operations scan the loaded records);
- pileup columns, variant APIs, tabix, VCF/BCF, FASTA/FASTQ, or pysam's command
  wrappers;
- `get_aligned_pairs(with_seq=True)`, which requires MD/reference
  reconstruction;
- a standalone CRAM codec.

This is therefore a drop-in replacement for the covered read APIs, not for all
of pysam.

## Install

Install the pinned Mojo nightly and Python dependencies:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-pysam.so`. Tests and benchmarks run through
Pixi:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import io

import mojopysam as pysam

sam = b"""@SQ\tSN:chr1\tLN:100
read1\t0\tchr1\t11\t60\t5M\t*\t0\t0\tACGTA\tIIIII
"""

with pysam.AlignmentFile(io.BytesIO(sam)) as alignments:
    print(alignments.references)
    for read in alignments.fetch("chr1", 10, 20):
        print(read.query_name, read.reference_start, read.cigarstring)

    a, c, g, t = alignments.count_coverage(
        "chr1", 10, 15, quality_threshold=20
    )
    print(int(a.sum() + c.sum() + g.sum() + t.sum()))
```

This prints `('chr1',)`, the record summary, and a total coverage of `5`. BAM
uses the same interface. For CRAM, pass `reference_filename=` when the file
does not embed the required reference.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14. Times are the best of three runs on
100,000 coordinate-sorted 100-base records. Ratio is pysam time divided by
mojo-pysam time, so values above 1 are faster.

| Case | mojo-pysam | pysam | Ratio | Result |
|---|---:|---:|---:|---|
| SAM parse + fields, 100k | 328.25 ms | 119.65 ms | 0.36x | slower |
| BAM parse + fields, 100k | 383.10 ms | 106.13 ms | 0.28x | slower |
| CIGAR coordinate methods, 100k | 833.83 ms | 379.04 ms | 0.45x | slower |
| count_coverage, 100k x 100 bp | 180.59 ms | 1035.49 ms | 5.73x | faster |

The base-coverage kernel wins because it traverses packed BAM sequence and
CIGAR data in one Mojo call. Object-oriented field and CIGAR access is slower
than pysam: pysam's mature Cython/htslib objects avoid the Python property and
`ctypes` costs this compatibility layer pays.

No GPU path is provided. Parsing, packed-CIGAR traversal, and coverage are
memory-bound integer kernels with well under two arithmetic operations per byte
moved. Host/device transfers would dominate, so the implementation remains
CPU-only. Parallel BAM field decoding was also benchmarked with a 4,096-record
threshold, but its extra boundary pass and synchronization made the end-to-end
case slower; the serial scanner is retained.

## How it works

The Python layer reads SAM bytes directly or decompresses BGZF/BAM with the
standard library. One Mojo scan uses SIMD delimiter search with a scalar tail
and fills a contiguous `int64` record table with field offsets, fixed BAM core
values, and CIGAR summaries. `AlignedSegment` objects retain the immutable
source bytes and lightweight memoryview slices of the shared native row table,
avoiding a NumPy row-view allocation for every record traversal. Variable-length
names, sequences, qualities, CIGARs, and tags are decoded lazily. BAM CIGAR words
are unpacked directly from the source buffer without an intermediate byte copy.
Coordinate and query-length properties consume the native summaries directly
without reconstructing CIGAR or sequence objects.

Buffers cross the C ABI as integer addresses. Exported Mojo functions rebuild
typed `UnsafePointer` values using `AnyOrigin[mut=True]`; Python owns every
input and output allocation. BAM coverage stays in the native packed layout:
the kernel walks 32-bit CIGAR words and 4-bit sequence codes directly into
four contiguous `int64` count rows, which Python exposes as `uint64` NumPy
views.

## License

MIT
