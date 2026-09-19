"""End-to-end CLI tests: filtering, windows, orientation, dosage, and errors.

The tiny fixture has known ALT-dosage vectors, so the expected correlations are
computed by hand:

* target ``1:100:A:G``        ALT=G copies ``[0,0,1,1,2,2]``
* ``1:100:G:A`` (swapped)     ALT=A copies ``[0,0,1,1,2,2]``  -> r_alt = +1
* ``1:150:A:G``               ALT=G copies ``[0,0,1,1,2,2]``  -> r_alt = +1
* ``1:250:T:C``               ALT=C copies ``[0,1,0,2,0,2]``  -> r^2 = 0.0517241
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from crossld.crossld import main as cli_main
from tests.conftest import make_bfile, make_pfile, write_bed_intervals, write_extract

IDS = [f"S{i}" for i in range(6)]
VARIANTS_A = [
    ("1", "1:100:A:G", 100, "A", "G"),
    ("1", "1:200:C:T", 200, "C", "T"),
]
VARIANTS_B = [
    ("1", "1:100:G:A", 100, "G", "A"),
    ("1", "1:150:A:G", 150, "A", "G"),
    ("1", "1:250:T:C", 250, "T", "C"),
]
GENOTYPES_A = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=np.int8)
GENOTYPES_B = np.array(
    [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 0, 2]], dtype=np.int8
)

R2_BY_ID = {"1:100:G:A": 1.0, "1:150:A:G": 1.0, "1:250:T:C": 0.0517241}
R_MAJOR_BY_ID = {"1:100:G:A": 1.0, "1:150:A:G": 1.0, "1:250:T:C": 0.227429}


@pytest.fixture()
def tiny_pair(tmp_path):
    prefix_a = make_bfile(tmp_path, "a", GENOTYPES_A, IDS, VARIANTS_A)
    prefix_b = make_bfile(tmp_path, "b", GENOTYPES_B, IDS, VARIANTS_B)
    return prefix_a, prefix_b


@pytest.fixture()
def tiny_pair_pfile(tmp_path):
    prefix_a = make_pfile(tmp_path, "a", GENOTYPES_A, IDS, VARIANTS_A)
    prefix_b = make_pfile(tmp_path, "b", GENOTYPES_B, IDS, VARIANTS_B)
    return prefix_a, prefix_b


def _read_rows(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return lines[0], [line.split("\t") for line in lines[1:] if line]


def _base_args(prefix_a, prefix_b, out_path, target="1:100:A:G", fmt="bfile"):
    flag = "--bfile1" if fmt == "bfile" else "--pfile1"
    flag2 = "--bfile2" if fmt == "bfile" else "--pfile2"
    return [
        flag, prefix_a,
        flag2, prefix_b,
        "--ld-snp", target,
        "--ld-window", "0",
        "--ld-window-kb", "10",
        "--ld-window-r2", "0",
        "--out", str(out_path),
    ]


def _set(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


# --------------------------------------------------------------------------- #
# output shapes
# --------------------------------------------------------------------------- #

def test_end_to_end_reports_expected_r2(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out_path)) == 0
    header, rows = _read_rows(out_path)
    assert header.endswith("UNPHASED_R2")
    assert {row[5]: float(row[6]) for row in rows} == pytest.approx(R2_BY_ID, abs=1e-6)
    assert {row[2] for row in rows} == {"1:100:A:G"}


def test_r_flag_reports_signed_r_and_alleles(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out_path) + ["--r"]) == 0
    header, rows = _read_rows(out_path)
    assert header.split("\t") == [
        "#CHROM_A", "POS_A", "ID_A", "MAJ_A",
        "CHROM_B", "POS_B", "ID_B", "MAJ_B", "UNPHASED_R",
    ]
    assert {row[6]: float(row[8]) for row in rows} == pytest.approx(
        R_MAJOR_BY_ID, abs=1e-5
    )
    # A tied allele frequency (0.5) reports REF as the coding allele.
    assert {row[3] for row in rows} == {"A"}
    assert {row[7] for row in rows} == {"G", "A", "T"}


def test_pfile_input_matches_bfile(tiny_pair, tiny_pair_pfile, tmp_path):
    pa, pb = tiny_pair
    bfile_out = tmp_path / "bfile.vcor"
    assert cli_main(_base_args(pa, pb, bfile_out)) == 0
    qa, qb = tiny_pair_pfile
    pfile_out = tmp_path / "pfile.vcor"
    assert cli_main(_base_args(qa, qb, pfile_out, fmt="pfile")) == 0
    assert bfile_out.read_text() == pfile_out.read_text()


def test_output_defaults_to_stdout(tiny_pair, capsys):
    prefix_a, prefix_b = tiny_pair
    args = _base_args(prefix_a, prefix_b, Path("/dev/null"))
    idx = args.index("--out")
    del args[idx : idx + 2]
    assert cli_main(args) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("#CHROM_A\tPOS_A\tID_A")
    assert len(captured.out.splitlines()) == 4  # header + 3 pairs


def test_ref_based_orientation_changes_sign(tmp_path):
    """--ref-based orients to REF; --r exposes the resulting sign.

    Dataset 1 codes ALT=G at 11/12 copies (ALT major), dataset 2 records the
    same physical variant with swapped REF/ALT at 2/12 ALT copies (REF major),
    so switching orientation mirrors only the dataset-1 side and flips the sign.
    """

    variants_a = [("1", "1:100:A:G", 100, "A", "G")]
    variants_b = [("1", "1:100:G:A", 100, "G", "A")]
    prefix_a = make_bfile(
        tmp_path, "ref_a", np.array([[2, 2, 2, 2, 2, 1]], dtype=np.int8), IDS, variants_a
    )
    prefix_b = make_bfile(
        tmp_path, "ref_b", np.array([[0, 0, 1, 1, 0, 0]], dtype=np.int8), IDS, variants_b
    )
    common = [
        "--bfile1", prefix_a, "--bfile2", prefix_b,
        "--ld-snp", "1:100:A:G",
        "--ld-window-kb", "10",
        "--ld-window-r2", "0",
        "--r",
    ]

    major_path = tmp_path / "major.vcor"
    assert cli_main(common + ["--out", str(major_path)]) == 0
    major_header, major = _read_rows(major_path)
    assert major_header.split("\t")[3] == "MAJ_A"
    assert major_header.split("\t")[7] == "MAJ_B"
    assert float(major[0][8]) == pytest.approx(-0.316228, abs=1e-5)

    ref_path = tmp_path / "ref.vcor"
    assert cli_main(common + ["--ref-based", "--out", str(ref_path)]) == 0
    ref_header, ref = _read_rows(ref_path)
    assert ref_header.split("\t")[3] == "REF_A"
    assert ref_header.split("\t")[7] == "REF_B"
    assert float(ref[0][8]) == pytest.approx(0.316228, abs=1e-5)
    assert float(ref[0][8]) == pytest.approx(-float(major[0][8]))
    assert major[0][3] != ref[0][3]  # different coding allele on the A side


# --------------------------------------------------------------------------- #
# thresholds, windows, filtering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "threshold, expected",
    [
        ("1.0", {"1:100:G:A", "1:150:A:G"}),
        ("0.5", {"1:100:G:A", "1:150:A:G"}),
        ("0.0517242", {"1:100:G:A", "1:150:A:G"}),
        ("0.0517241", {"1:100:G:A", "1:150:A:G", "1:250:T:C"}),
        ("0", {"1:100:G:A", "1:150:A:G", "1:250:T:C"}),
    ],
)
def test_r2_threshold_is_inclusive(tiny_pair, tmp_path, threshold, expected):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _set(_base_args(prefix_a, prefix_b, out_path), "--ld-window-r2", threshold)
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == expected


@pytest.mark.parametrize(
    "window, expected",
    [
        ("0", {"1:100:G:A", "1:150:A:G", "1:250:T:C"}),
        ("1", {"1:100:G:A"}),
        ("2", {"1:100:G:A", "1:150:A:G"}),
    ],
)
def test_ld_window_keeps_the_nearest_variants(tiny_pair, tmp_path, window, expected):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _set(_base_args(prefix_a, prefix_b, out_path), "--ld-window", window)
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == expected


@pytest.mark.parametrize(
    "kb, expected",
    [
        ("10", {"1:100:G:A", "1:150:A:G", "1:250:T:C"}),
        ("0.06", {"1:100:G:A", "1:150:A:G"}),
        ("0.04", {"1:100:G:A"}),
        ("0", {"1:100:G:A"}),
    ],
)
def test_ld_window_kb_bounds_distance(tiny_pair, tmp_path, kb, expected):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _set(_base_args(prefix_a, prefix_b, out_path), "--ld-window-kb", kb)
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == expected


# --------------------------------------------------------------------------- #
# multi-chromosome windows: sorted contiguous vs interleaved unsorted
# --------------------------------------------------------------------------- #

MULTI_IDS = [f"S{i}" for i in range(6)]
MULTI_TARGETS = "1:1000:A:G,2:1000:C:T"
MULTI_VARIANTS_A = [
    ("1", "1:1000:A:G", 1000, "A", "G"),
    ("2", "2:1000:C:T", 1000, "C", "T"),
]
MULTI_GENOTYPES_A = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 0, 2]], dtype=np.int8)

# Per chromosome: proxy at the target position, near 100 bp away, distant 4 kb away.
MULTI_VARIANTS_SORTED = [
    ("1", "1:1000:A:G", 1000),
    ("1", "1:1100:A:G", 1100),
    ("1", "1:5000:A:G", 5000),
    ("2", "2:1000:C:T", 1000),
    ("2", "2:1100:C:T", 1100),
    ("2", "2:5000:C:T", 5000),
]
# The same six variants with interleaved chromosomes and descending positions.
MULTI_VARIANTS_INTERLEAVED = [
    ("2", "2:5000:C:T", 5000),
    ("1", "1:1100:A:G", 1100),
    ("2", "2:1000:C:T", 1000),
    ("1", "1:5000:A:G", 5000),
    ("2", "2:1100:C:T", 1100),
    ("1", "1:1000:A:G", 1000),
]
MULTI_LAYOUTS = {
    "sorted": MULTI_VARIANTS_SORTED,
    "interleaved": MULTI_VARIANTS_INTERLEAVED,
}


def _multi_build(tmp_path, name, layout, fmt):
    """Build the two-target dataset 1 and one dataset-2 layout (bfile or pfile)."""

    variants_b = [(chrom, vid, pos, "A", "G") for chrom, vid, pos in MULTI_LAYOUTS[layout]]
    genotypes_b = np.tile(np.array([0, 0, 1, 1, 2, 2], dtype=np.int8), (len(variants_b), 1))
    make = make_bfile if fmt == "bfile" else make_pfile
    prefix_a = make(tmp_path, f"a_{name}", MULTI_GENOTYPES_A, MULTI_IDS, MULTI_VARIANTS_A)
    prefix_b = make(tmp_path, f"b_{name}", genotypes_b, MULTI_IDS, variants_b)
    return prefix_a, prefix_b


@pytest.fixture(params=sorted(MULTI_LAYOUTS))
def multi_pair(request, tmp_path):
    """The multi-chromosome pair in one of the two dataset-2 layouts (bfile)."""

    return _multi_build(tmp_path, request.param, request.param, "bfile")


def _multi_args(prefix_a, prefix_b, out_path, extra, target=MULTI_TARGETS, fmt="bfile"):
    flag = "--bfile1" if fmt == "bfile" else "--pfile1"
    flag2 = "--bfile2" if fmt == "bfile" else "--pfile2"
    return [
        flag, prefix_a,
        flag2, prefix_b,
        "--ld-snps", target,
        "--ld-window-r2", "0",
        *extra,
        "--out", str(out_path),
    ]


def _b_ids(path) -> set[str]:
    return {row[5] for row in _read_rows(path)[1]}


@pytest.mark.parametrize(
    "kb, expected",
    [
        ("0", {"1:1000:A:G"}),
        ("0.099", {"1:1000:A:G"}),
        ("0.1", {"1:1000:A:G", "1:1100:A:G"}),  # boundary is inclusive
        ("0.2", {"1:1000:A:G", "1:1100:A:G"}),
        ("10", {"1:1000:A:G", "1:1100:A:G", "1:5000:A:G"}),
    ],
)
def test_multichrom_window_kb_excludes_distant(multi_pair, tmp_path, kb, expected):
    prefix_a, prefix_b = multi_pair
    out_path = tmp_path / "out.vcor"
    args = _multi_args(
        prefix_a, prefix_b, out_path, ["--ld-window-kb", kb], target="1:1000:A:G"
    )
    assert cli_main(args) == 0
    assert _b_ids(out_path) == expected


@pytest.mark.parametrize(
    "count, expected",
    [
        ("1", {"1:1000:A:G"}),
        ("2", {"1:1000:A:G", "1:1100:A:G"}),
        ("3", {"1:1000:A:G", "1:1100:A:G", "1:5000:A:G"}),
    ],
)
def test_multichrom_window_count_keeps_nearest(multi_pair, tmp_path, count, expected):
    """The nearest variants win by position, not by file order (interleaved)."""

    prefix_a, prefix_b = multi_pair
    out_path = tmp_path / "out.vcor"
    args = _multi_args(
        prefix_a,
        prefix_b,
        out_path,
        ["--ld-window-kb", "10", "--ld-window", count],
        target="1:1000:A:G",
    )
    assert cli_main(args) == 0
    assert _b_ids(out_path) == expected


def test_multichrom_window_never_leaks_across_chromosomes(multi_pair, tmp_path):
    prefix_a, prefix_b = multi_pair
    out_path = tmp_path / "out.vcor"
    args = _multi_args(prefix_a, prefix_b, out_path, ["--ld-window-kb", "10"])
    assert cli_main(args) == 0
    _, rows = _read_rows(out_path)
    assert rows  # both targets produced rows
    assert all(row[0] == row[3] for row in rows)
    # The distant same-chromosome partners are present, so the run is unbounded
    # and a cross-chromosome leak would have been visible.
    assert {"1:5000:A:G", "2:5000:C:T"} <= {row[5] for row in rows}


@pytest.mark.parametrize("fmt", ["bfile", "pfile"])
@pytest.mark.parametrize(
    "extra",
    [
        ["--ld-window-kb", "10", "--ld-window", "0"],
        ["--ld-window-kb", "10", "--ld-window", "1"],
        ["--ld-window-kb", "0.2", "--ld-window", "0"],
    ],
    ids=["kb10", "nearest1", "kb0.2"],
)
def test_sorted_and_interleaved_layouts_agree(tmp_path, fmt, extra):
    """File order must not change a window result, for bfile and pfile alike."""

    sorted_a, sorted_b = _multi_build(tmp_path, f"sorted_{fmt}", "sorted", fmt)
    inter_a, inter_b = _multi_build(tmp_path, f"inter_{fmt}", "interleaved", fmt)
    sorted_out = tmp_path / "sorted.vcor"
    inter_out = tmp_path / "interleaved.vcor"
    assert cli_main(_multi_args(sorted_a, sorted_b, sorted_out, extra, fmt=fmt)) == 0
    assert cli_main(_multi_args(inter_a, inter_b, inter_out, extra, fmt=fmt)) == 0
    assert sorted_out.read_text(encoding="utf-8") == inter_out.read_text(encoding="utf-8")


def test_window_applies_after_extract(tmp_path):
    """Windows see only the variants surviving --extract, interleaved included."""

    prefix_a, prefix_b = _multi_build(tmp_path, "extract", "interleaved", "pfile")
    extract_path = tmp_path / "extract.txt"
    out_path = tmp_path / "out.vcor"
    window = ["--ld-window-kb", "10", "--ld-window", "1"]

    write_extract(extract_path, ["1:1000:A:G", "1:5000:A:G"])
    args = _multi_args(
        prefix_a,
        prefix_b,
        out_path,
        [*window, "--extract", str(extract_path)],
        target="1:1000:A:G",
        fmt="pfile",
    )
    assert cli_main(args) == 0
    assert _b_ids(out_path) == {"1:1000:A:G"}

    write_extract(extract_path, ["1:5000:A:G"])
    args = _multi_args(
        prefix_a,
        prefix_b,
        out_path,
        [*window, "--extract", str(extract_path)],
        target="1:1000:A:G",
        fmt="pfile",
    )
    assert cli_main(args) == 0
    assert _b_ids(out_path) == {"1:5000:A:G"}


def test_extract_and_exclude_shrink_results(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    extract_path = tmp_path / "extract.txt"
    write_extract(extract_path, ["1:150:A:G"])
    out_extract = tmp_path / "extract.vcor"
    args = _base_args(prefix_a, prefix_b, out_extract) + ["--extract", str(extract_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_extract)[1]} == {"1:150:A:G"}

    exclude_path = tmp_path / "exclude.txt"
    write_extract(exclude_path, ["1:150:A:G"])
    out_exclude = tmp_path / "exclude.vcor"
    args = _base_args(prefix_a, prefix_b, out_exclude) + ["--exclude", str(exclude_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_exclude)[1]} == {"1:100:G:A", "1:250:T:C"}


def test_extract_bed_and_exclude_bed(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    # Variants at 1-based 100 and 150 span [99,100) and [149,150).
    bed_path = tmp_path / "region.bed"
    write_bed_intervals(bed_path, [("1", 99, 100), ("1", 149, 150)])

    out_extract = tmp_path / "extract.vcor"
    args = _base_args(prefix_a, prefix_b, out_extract) + ["--extract-bed", str(bed_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_extract)[1]} == {"1:100:G:A", "1:150:A:G"}

    out_exclude = tmp_path / "exclude.vcor"
    args = _base_args(prefix_a, prefix_b, out_exclude) + ["--exclude-bed", str(bed_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_exclude)[1]} == {"1:250:T:C"}


def test_keep_narrows_the_shared_samples(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("#IID\nS0\nS1\nS2\nS3\n", encoding="utf-8")
    full_path = tmp_path / "full.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, full_path)) == 0
    kept_path = tmp_path / "kept.vcor"
    args = _base_args(prefix_a, prefix_b, kept_path) + ["--keep", str(keep_path)]
    assert cli_main(args) == 0
    assert len(_read_rows(kept_path)[1]) == 3
    assert kept_path.read_text() != full_path.read_text()


# --------------------------------------------------------------------------- #
# dosage + blocking
# --------------------------------------------------------------------------- #

def test_dosage_reads_match_hardcalls(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    hard = tmp_path / "hard.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, hard)) == 0
    for flag in ("--dosage", "--dosage1", "--dosage2"):
        out = tmp_path / f"dos_{flag.strip('-')}.vcor"
        assert cli_main(_base_args(prefix_a, prefix_b, out) + [flag]) == 0
        assert out.read_text() == hard.read_text()


def test_block_size_does_not_change_results(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    default = tmp_path / "default.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, default)) == 0
    grouped = tmp_path / "grouped.vcor"
    args = _base_args(prefix_a, prefix_b, grouped) + ["--block-size", "1"]
    assert cli_main(args) == 0
    assert grouped.read_text() == default.read_text()


# --------------------------------------------------------------------------- #
# dataset-2 read cache
# --------------------------------------------------------------------------- #

CACHE_LIMITS = ("0", "1", "2", "100000")

# Two dataset-1 targets 100 bp apart on chromosome 1, and six dataset-2 variants
# around them, so both windows fully overlap (with --ld-window-kb 1).
OVERLAP_IDS = [f"S{i}" for i in range(6)]
OVERLAP_VARIANTS_A = [
    ("1", "1:1000:A:G", 1000, "A", "G"),
    ("1", "1:1100:C:T", 1100, "C", "T"),
]
OVERLAP_VARIANTS_B = [
    ("1", f"1:{pos}:A:G", pos, "A", "G")
    for pos in (500, 1000, 1050, 1100, 1200, 1600)
]
OVERLAP_GENOTYPES_A = np.array(
    [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 0, 2]], dtype=np.int8
)
OVERLAP_GENOTYPES_B = np.array(
    [
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 0, 2, 2],
        [0, 1, 0, 2, 0, 2],
        [1, 1, 0, 0, 2, 2],
        [0, 2, 1, 1, 0, 2],
    ],
    dtype=np.int8,
)


@pytest.fixture()
def overlap_pair(tmp_path):
    prefix_a = make_bfile(
        tmp_path, "cache_a", OVERLAP_GENOTYPES_A, OVERLAP_IDS, OVERLAP_VARIANTS_A
    )
    prefix_b = make_bfile(
        tmp_path, "cache_b", OVERLAP_GENOTYPES_B, OVERLAP_IDS, OVERLAP_VARIANTS_B
    )
    return prefix_a, prefix_b


def _overlap_args(prefix_a, prefix_b, out_path, extra=()):
    return [
        "--bfile1", prefix_a,
        "--bfile2", prefix_b,
        "--ld-snps", "1:1000:A:G,1:1100:C:T",
        "--ld-window-kb", "1",
        "--ld-window-r2", "0",
        *extra,
        "--out", str(out_path),
    ]


@pytest.mark.parametrize("cache_limit", CACHE_LIMITS)
@pytest.mark.parametrize(
    "extra",
    [(), ("--block-size", "1"), ("--ld-window", "2"), ("--r",)],
    ids=["plain", "block1", "window2", "r"],
)
def test_cache_limit_does_not_change_results(tiny_pair, tmp_path, cache_limit, extra):
    prefix_a, prefix_b = tiny_pair
    out0 = tmp_path / "cache0.vcor"
    args0 = _base_args(prefix_a, prefix_b, out0) + ["--cache-limit", "0", *extra]
    assert cli_main(args0) == 0
    out = tmp_path / f"cache{cache_limit}.vcor"
    args = _base_args(prefix_a, prefix_b, out) + ["--cache-limit", cache_limit, *extra]
    assert cli_main(args) == 0
    assert out.read_text(encoding="utf-8") == out0.read_text(encoding="utf-8")


@pytest.mark.parametrize("cache_limit", CACHE_LIMITS)
@pytest.mark.parametrize("extra", [
    ["--ld-window-kb", "10", "--ld-window", "0"],
    ["--ld-window-kb", "10", "--ld-window", "2"],
], ids=["kb10", "nearest2"])
def test_cache_limit_does_not_change_results_multichrom(
    multi_pair, tmp_path, cache_limit, extra
):
    prefix_a, prefix_b = multi_pair
    out0 = tmp_path / "multi_cache0.vcor"
    assert cli_main(
        _multi_args(prefix_a, prefix_b, out0, extra) + ["--cache-limit", "0"]
    ) == 0
    out = tmp_path / f"multi_cache{cache_limit}.vcor"
    assert cli_main(
        _multi_args(prefix_a, prefix_b, out, extra) + ["--cache-limit", cache_limit]
    ) == 0
    assert out.read_text(encoding="utf-8") == out0.read_text(encoding="utf-8")


@pytest.mark.parametrize("cache_limit", CACHE_LIMITS)
def test_cache_limit_does_not_change_results_for_overlapping_targets(
    overlap_pair, tmp_path, cache_limit
):
    prefix_a, prefix_b = overlap_pair
    out0 = tmp_path / "overlap_cache0.vcor"
    assert cli_main(
        _overlap_args(prefix_a, prefix_b, out0, ["--cache-limit", "0"])
    ) == 0
    out = tmp_path / f"overlap_cache{cache_limit}.vcor"
    assert cli_main(
        _overlap_args(prefix_a, prefix_b, out, ["--cache-limit", cache_limit])
    ) == 0
    assert out.read_text(encoding="utf-8") == out0.read_text(encoding="utf-8")


class _MessageCapture(logging.Handler):
    """Collects rendered log messages for assertions."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_cache_reuses_blocks_for_overlapping_targets(overlap_pair, tmp_path, monkeypatch):
    import crossld.crossld as crossld_module

    prefix_a, prefix_b = overlap_pair
    instances = []

    class _RecordingCache(crossld_module.BlockCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(crossld_module, "BlockCache", _RecordingCache)
    captured = _MessageCapture()
    logger = logging.getLogger("crossld.crossld")
    logger.addHandler(captured)
    try:
        assert cli_main(
            _overlap_args(
                prefix_a, prefix_b, tmp_path / "hits.vcor",
                ["--cache-limit", "3", "--debug"],
            )
        ) == 0
    finally:
        logger.removeHandler(captured)

    cache = instances[-1]
    assert cache.limit == 3
    assert cache.reused_variants > 0
    assert any("reused_variants=" in message for message in captured.messages)


@pytest.mark.parametrize(
    "min_overlap",
    ["0", "0.25", "0.5", "0.75", "1"],
)
def test_cache_min_overlap_does_not_change_results(overlap_pair, tmp_path, min_overlap):
    prefix_a, prefix_b = overlap_pair
    out0 = tmp_path / "minoverlap0.vcor"
    assert cli_main(
        _overlap_args(prefix_a, prefix_b, out0, ["--cache-limit", "0"])
    ) == 0
    out = tmp_path / f"minoverlap{min_overlap}.vcor"
    assert cli_main(
        _overlap_args(
            prefix_a, prefix_b, out,
            ["--cache-limit", "3", "--cache-min-overlap", min_overlap],
        )
    ) == 0
    assert out.read_text(encoding="utf-8") == out0.read_text(encoding="utf-8")

# --------------------------------------------------------------------------- #
# error paths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda a: [x for x in a if x not in ("--ld-snp", "1:100:A:G")],
            id="no-target",
        ),
        pytest.param(lambda a: _set(a, "--ld-window", "-1"), id="negative-window"),
        pytest.param(lambda a: _set(a, "--ld-window-r2", "1.5"), id="r2-high"),
        pytest.param(lambda a: _set(a, "--ld-window-r2", "-1"), id="r2-negative"),
        pytest.param(lambda a: _set(a, "--ld-window-kb", "-1"), id="negative-kb"),
        pytest.param(lambda a: a + ["--block-size", "0"], id="zero-block-size"),
        pytest.param(lambda a: a + ["--cache-limit", "-1"], id="negative-cache-limit"),
        pytest.param(lambda a: a + ["--cache-min-overlap", "1.5"], id="overlap-high"),
        pytest.param(lambda a: a + ["--cache-min-overlap", "-0.1"], id="overlap-negative"),
    ],
)
def test_validation_errors_exit_2(tiny_pair, tmp_path, mutate):
    prefix_a, prefix_b = tiny_pair
    args = _base_args(prefix_a, prefix_b, tmp_path / "out.vcor")
    assert cli_main(mutate(args)) == 2


def test_missing_dataset_file_exits_2(tmp_path):
    assert (
        cli_main(
            ["--bfile1", str(tmp_path / "nope"), "--bfile2", str(tmp_path / "nope")]
        )
        == 2
    )


def test_no_shared_samples_exits_2(tmp_path):
    variants = [("1", "1:100:A:G", 100, "A", "G")]
    genotypes = np.array([[0, 1, 2]], dtype=np.int8)
    prefix_a = make_bfile(tmp_path, "a", genotypes, ["A1", "A2", "A3"], variants)
    prefix_b = make_bfile(tmp_path, "b", genotypes, ["B1", "B2", "B3"], variants)
    assert (
        cli_main(
            [
                "--bfile1", prefix_a,
                "--bfile2", prefix_b,
                "--ld-snp", "1:100:A:G",
                "--out", str(tmp_path / "out.vcor"),
            ]
        )
        == 2
    )


def test_unmatched_target_exits_0(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out_path, target="not-a-variant")) == 0
    assert len(_read_rows(out_path)[1]) == 0  # header + no rows


def test_degenerate_target_writes_no_rows(tmp_path):
    """A target with fewer than three valid samples skips every pair at once."""

    variants = [("1", "1:100:A:G", 100, "A", "G"), ("1", "1:150:A:G", 150, "A", "G")]
    prefix_a = make_bfile(
        tmp_path,
        "a",
        np.array([[-9, -9, -9, -9, 1, 2]], dtype=np.int8),
        IDS,
        variants[:1],
    )
    prefix_b = make_bfile(
        tmp_path,
        "b",
        np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 0, 2]], dtype=np.int8),
        IDS,
        variants,
    )
    out_path = tmp_path / "out.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out_path)) == 0
    header, rows = _read_rows(out_path)
    assert rows == []
    assert header.endswith("UNPHASED_R2")
