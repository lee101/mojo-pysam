from __future__ import annotations

import array
import gzip
import io
import struct

import numpy as np
import pysam
import pytest

import mojopysam as mp


SAM_TEXT = """@HD\tVN:1.6\tSO:coordinate
@SQ\tSN:chr1\tLN:200
@SQ\tSN:chr2\tLN:150
@RG\tID:rg1\tSM:sample
r1\t99\tchr1\t11\t60\t3S5M1I4M2D3M1S\t=\t41\t47\tTTTACGTAGCCCATTAA\tIIIIIIIIIIIIIIIII\tNM:i:3\tRG:Z:rg1\tAS:i:12
dup\t1024\tchr1\t12\t10\t5M\t*\t0\t0\tAAAAA\tIIIII\tNM:i:0
sec\t256\tchr1\t13\t10\t5M\t*\t0\t0\tCCCCC\tIIIII\tNM:i:0
r2\t147\tchr1\t16\t37\t2M3N5M\t=\t11\t-47\tCGTTGCA\tABCDEFG\tNM:i:1\tXA:A:Q\tXF:f:1.5
r3\t0\tchr2\t5\t20\t4M1D4M\t*\t0\t0\tAACCGGTT\tHHHHHHHH\tNM:i:1\tZZ:Z:hello\tBI:B:i,1,-2\tBF:B:f,1.5,2.5
unmapped\t4\t*\t0\t0\t*\t*\t0\t0\tNNNN\t*\tXX:i:7
"""


@pytest.fixture(scope="module")
def alignment_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("alignments")
    sam = root / "sample.sam"
    bam = root / "sample.bam"
    cram = root / "sample.cram"
    fasta = root / "reference.fa"
    sam.write_text(SAM_TEXT)
    fasta.write_text(">chr1\n" + "ACGT" * 50 + "\n>chr2\n" + "TGCA" * 37 + "TG\n")
    pysam.faidx(str(fasta))
    with pysam.AlignmentFile(sam, "r") as source:
        with pysam.AlignmentFile(bam, "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.index(str(bam))
    with pysam.AlignmentFile(sam, "r") as source:
        with pysam.AlignmentFile(
            cram, "wc", header=source.header, reference_filename=str(fasta)
        ) as target:
            for record in source:
                target.write(record)
    return sam, bam, cram, fasta


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_header_parity(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(path) as theirs:
        assert ours.references == theirs.references
        assert ours.lengths == theirs.lengths
        assert ours.nreferences == theirs.nreferences
        assert ours.header.to_dict() == theirs.header.to_dict()
        assert str(ours.header) == str(theirs.header)
        assert ours.is_sam is (kind == "sam")
        assert ours.is_bam is (kind == "bam")
        assert not ours.is_cram
        assert ours.text == str(ours.header)


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_all_core_record_fields_match(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(path) as theirs:
        left, right = list(ours), list(theirs)
    assert len(left) == len(right) == 6
    attributes = [
        "query_name",
        "flag",
        "reference_id",
        "reference_name",
        "reference_start",
        "reference_end",
        "mapping_quality",
        "cigarstring",
        "cigartuples",
        "next_reference_id",
        "next_reference_name",
        "next_reference_start",
        "template_length",
        "query_sequence",
        "query_length",
        "query_alignment_start",
        "query_alignment_end",
        "query_alignment_length",
        "query_alignment_sequence",
    ]
    for one, two in zip(left, right):
        for attribute in attributes:
            assert getattr(one, attribute) == getattr(two, attribute), (
                one.query_name,
                attribute,
            )
        assert one.query_qualities == two.query_qualities
        assert one.query_alignment_qualities == two.query_alignment_qualities
        assert one.get_tags(with_value_type=True) == two.get_tags(with_value_type=True)
        assert one.to_string() == two.to_string()


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_cigar_methods_match(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(path) as theirs:
        for one, two in zip(ours, theirs):
            assert one.get_reference_positions() == two.get_reference_positions()
            assert one.get_reference_positions(full_length=True) == two.get_reference_positions(
                full_length=True
            )
            assert one.get_aligned_pairs() == two.get_aligned_pairs()
            assert one.get_aligned_pairs(matches_only=True) == two.get_aligned_pairs(
                matches_only=True
            )
            assert one.get_aligned_pairs(with_cigar=True) == two.get_aligned_pairs(
                with_cigar=True
            )
            assert one.get_blocks() == two.get_blocks()
            assert one.get_overlap(12, 22) == two.get_overlap(12, 22)
            assert one.get_forward_sequence() == two.get_forward_sequence()
            assert one.get_forward_qualities() == two.get_forward_qualities()
            assert one.bin == two.bin
            got_lengths, got_blocks = one.get_cigar_stats()
            ref_lengths, ref_blocks = two.get_cigar_stats()
            assert got_lengths == ref_lengths
            assert got_blocks == ref_blocks


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_flags_and_tags_match(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    flag_names = [
        "is_paired",
        "is_proper_pair",
        "is_unmapped",
        "mate_is_unmapped",
        "is_reverse",
        "mate_is_reverse",
        "is_read1",
        "is_read2",
        "is_secondary",
        "is_qcfail",
        "is_duplicate",
        "is_supplementary",
    ]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(path) as theirs:
        for one, two in zip(ours, theirs):
            for name in flag_names:
                assert getattr(one, name) == getattr(two, name)
            for tag in [key for key, _ in two.tags]:
                assert one.has_tag(tag)
                assert one.get_tag(tag) == pytest.approx(two.get_tag(tag))
            with pytest.raises(KeyError):
                one.get_tag("NO")


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_fetch_count_and_iteration(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(path) as theirs:
        got = [record.query_name for record in ours.fetch("chr1", 14, 19)]
        expected = [record.query_name for record in theirs.fetch(until_eof=True) if record.reference_name == "chr1" and record.reference_end > 14 and record.reference_start < 19]
        assert got == expected
        assert [r.query_name for r in ours.fetch(region="chr2:1-20")] == ["r3"]
        assert ours.count("chr1", 0, 100, read_callback="nofilter") == 4
        assert ours.count("chr1", 0, 100, read_callback="all") == 2
        assert [r.query_name for r in ours.head(2)] == ["r1", "dup"]
        first = next(ours)
        assert first.query_name == "r1"
        assert ours.tell() == 1
        ours.reset()
        assert next(ours).query_name == "r1"


@pytest.mark.parametrize("kind", ["sam", "bam"])
def test_count_coverage_matches_pysam(alignment_paths, kind):
    path = alignment_paths[0 if kind == "sam" else 1]
    with mp.AlignmentFile(path) as ours, pysam.AlignmentFile(alignment_paths[1]) as theirs:
        for quality in [0, 25, 40]:
            got = ours.count_coverage("chr1", 7, 35, quality_threshold=quality)
            expected = theirs.count_coverage(
                "chr1", 7, 35, quality_threshold=quality, read_callback="all"
            )
            for one, two in zip(got, expected):
                assert one.dtype == np.uint64
                assert np.array_equal(one, two)
        with pytest.raises(ValueError, match="invalid contig"):
            ours.count_coverage("missing", 0, 10)


def test_cram_codec_backend_matches_upstream(alignment_paths):
    cram, fasta = alignment_paths[2], alignment_paths[3]
    with mp.AlignmentFile(cram, reference_filename=str(fasta)) as ours:
        with pysam.AlignmentFile(cram, reference_filename=str(fasta)) as theirs:
            left, right = list(ours), list(theirs)
    assert [record.to_string() for record in left] == [record.to_string() for record in right]
    assert all(type(record).__module__.startswith("mojopysam") for record in left)


def test_readme_usage_example():
    sam = b"""@SQ\tSN:chr1\tLN:100
read1\t0\tchr1\t11\t60\t5M\t*\t0\t0\tACGTA\tIIIII
"""
    with mp.AlignmentFile(io.BytesIO(sam)) as alignments:
        assert alignments.references == ("chr1",)
        records = list(alignments.fetch("chr1", 10, 20))
        assert [(r.query_name, r.reference_start, r.cigarstring) for r in records] == [
            ("read1", 10, "5M")
        ]
        coverage = alignments.count_coverage(
            "chr1", 10, 15, quality_threshold=20
        )
        assert sum(int(values.sum()) for values in coverage) == 5


def test_quality_helpers_match_pysam():
    text = "!+5?IS]"
    assert mp.qualitystring_to_array(text) == pysam.qualitystring_to_array(text)
    values = array.array("B", [0, 10, 20, 30, 40])
    assert mp.array_to_qualitystring(values) == pysam.array_to_qualitystring(values)


def test_aligned_segment_fromstring_matches_pysam():
    header = mp.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100}]}
    )
    upstream_header = pysam.AlignmentHeader.from_dict(header.to_dict())
    line = "q\t16\tchr1\t3\t44\t2S4M1D2M\t*\t0\t0\tTTACGTGG\tHHHHHHHH\tNM:i:1"
    ours = mp.AlignedSegment.fromstring(line, header)
    theirs = pysam.AlignedSegment.fromstring(line, upstream_header)
    assert ours.to_string() == theirs.to_string()
    assert ours.get_aligned_pairs(with_cigar=True) == theirs.get_aligned_pairs(with_cigar=True)


def test_hard_clips_do_not_hide_terminal_soft_clips():
    header = mp.AlignmentHeader.from_references(["chr1"], [100])
    upstream_header = pysam.AlignmentHeader.from_references(["chr1"], [100])
    line = "q\t0\tchr1\t3\t44\t2H3S4M2S1H\t*\t0\t0\tTTTACGTGG\tHHHHHHHHH"
    ours = mp.AlignedSegment.fromstring(line, header)
    theirs = pysam.AlignedSegment.fromstring(line, upstream_header)
    assert ours.query_alignment_start == theirs.query_alignment_start == 3
    assert ours.query_alignment_end == theirs.query_alignment_end == 7


def test_user_constructed_segment_and_flag_setter():
    header = mp.AlignmentHeader.from_references(["chr1"], [100])
    record = mp.AlignedSegment(header)
    record.query_name = "new"
    record.reference_id = 0
    record.reference_start = 4
    record.cigarstring = "5M"
    record.query_sequence = "AACGT"
    record.query_qualities = [30] * 5
    record.is_reverse = True
    record.set_tag("NM", 1)
    assert record.flag == 16
    assert record.reference_end == 9
    assert record.get_tag("NM") == 1
    assert record.to_string().startswith("new\t16\tchr1\t5\t0\t5M")


def test_malformed_sam_is_rejected(tmp_path):
    path = tmp_path / "bad.sam"
    path.write_text("@SQ\tSN:chr1\tLN:10\nonly\tthree\tfields\n")
    with pytest.raises(ValueError, match="fewer than 11"):
        mp.AlignmentFile(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"BAM\1" + struct.pack("<i", -1) + b"\0\0\0\0", "text length"),
        (b"BAM\1" + struct.pack("<i", 0) + struct.pack("<i", -1), "reference count"),
    ],
)
def test_malformed_bam_headers_are_rejected(tmp_path, payload, message):
    path = tmp_path / "bad.bam"
    path.write_bytes(gzip.compress(payload))
    with pytest.raises(ValueError, match=message):
        mp.AlignmentFile(path)


def test_malformed_bam_record_lengths_are_rejected(tmp_path):
    header = b"BAM\1" + struct.pack("<i", 0) + struct.pack("<i", 0)
    core = bytearray(32)
    core[8] = 0  # l_read_name must include at least the terminating NUL
    record = struct.pack("<i", 32) + core
    path = tmp_path / "bad-record.bam"
    path.write_bytes(gzip.compress(header + record))
    with pytest.raises(ValueError, match="variable-length"):
        mp.AlignmentFile(path)


def test_native_bridge_rejects_noncontiguous_arrays():
    from mojopysam._lib import addr

    with pytest.raises(TypeError, match="C-contiguous"):
        addr(np.zeros((2, 2), dtype=np.int64)[:, 0])


@pytest.mark.parametrize("length", [1, 3, 4, 5, 9])
def test_sam_simd_scanner_handles_scalar_tails(length):
    sequence = "A" * length
    record = (
        f"tail{length}\t0\tchr1\t2\t60\t{length}M\t*\t0\t0\t"
        f"{sequence}\t{'I' * length}\tNM:i:0"
    )
    with mp.AlignmentFile(
        io.BytesIO(record.encode()),
        header={"SQ": [{"SN": "chr1", "LN": 100}]},
    ) as alignment:
        segment = next(alignment)
    assert segment.query_name == f"tail{length}"
    assert segment.reference_end == length + 1
    assert segment.query_alignment_length == length
