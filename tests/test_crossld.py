"""End-to-end CLI tests: filtering, windows, dosage, output, and error paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crossld.crossld import build_parser, main as cli_main
from tests.conftest import make_bfile, write_bed_intervals, write_extract


def _read_rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0], [line.split("\t") for line in lines[1:] if line]


@pytest.fixture()
def tiny_pair(tmp_path):
    """Synthetic pair of datasets with known allele representations.

    ``1:100:A:G``      ALT=G, G copies [0,0,1,1,2,2] -> major is REF (A)
    ``1:100:G:A``      the same variant written with REF/ALT swapped
    ``1:150:G:A``      REF=A is the major allele, G copies [2,2,1,1,0,0]
    ``1:250:T:C``      ALT=C copies [0,1,0,2,0,2], uncorrelated
    """

    ids = [f"S{i}" for i in range(6)]
    variants_a = [
        ("1", "1:100:A:G", 100, "A", "G"),
        ("1", "1:200:C:T", 200, "C", "T"),
    ]
    variants_b = [
        ("1", "1:100:G:A", 100, "G", "A"),
        ("1", "1:150:A:G", 150, "A", "G"),
        ("1", "1:250:T:C", 250, "T", "C"),
    ]
    genotypes_a = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
        ],
        dtype=np.int8,
    )
    genotypes_b = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 2, 0, 2],
        ],
        dtype=np.int8,
    )
    prefix_a = make_bfile(tmp_path, "a", genotypes_a, ids, variants_a)
    prefix_b = make_bfile(tmp_path, "b", genotypes_b, ids, variants_b)
    return prefix_a, prefix_b


def _base_args(prefix_a, prefix_b, out_path, target="1:100:A:G"):
    return [
        "--bfile1",
        prefix_a,
        "--bfile2",
        prefix_b,
        "--ld-snp",
        target,
        "--ld-window",
        "2",
        "--ld-window-kb",
        "10",
        "--ld-window-r2",
        "0",
        "--out",
        str(out_path),
    ]


def _set_arg(args: list[str], flag: str, value: str) -> list[str]:
    """Return a copy of ``args`` with ``flag``'s value replaced by ``value``."""

    out = list(args)
    index = out.index(flag)
    out[index + 1] = value
    return out


def test_end_to_end_reports_expected_pairs(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out_path)) == 0
    header, rows = _read_rows(out_path)
    assert header.endswith("UNPHASED_R2")
    by_id = {row[5]: float(row[6]) for row in rows}
    # --ld-window 2 allows up to 2 dataset-2 variants on each side, and the
    # 10 kb window is wide enough for all three.
    assert by_id == pytest.approx(
        {"1:100:G:A": 1.0, "1:150:A:G": 1.0, "1:250:T:C": 0.0517241}, abs=1e-6
    )
    assert {row[2] for row in rows} == {"1:100:A:G"}


def test_r_flag_adds_signed_column(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _base_args(prefix_a, prefix_b, out_path)
    args.append("--r")
    assert cli_main(args) == 0
    header, rows = _read_rows(out_path)
    # --r reports the coding allele per side instead of UNPHASED_R2.
    assert header.split("\t") == [
        "#CHROM_A", "POS_A", "ID_A", "MAJ_A",
        "CHROM_B", "POS_B", "ID_B", "MAJ_B", "UNPHASED_R",
    ]
    by_id = {row[6]: float(row[8]) for row in rows}
    # 1:100:G:A is the same physical variant as the target with REF/ALT swapped;
    # both high-LD pairs are positively oriented to their major alleles.
    assert by_id["1:100:G:A"] == pytest.approx(1.0, abs=1e-9)
    assert by_id["1:150:A:G"] == pytest.approx(1.0, abs=1e-9)
    assert by_id["1:250:T:C"] == pytest.approx(0.227429, abs=1e-5)


def test_ref_based_orientation_changes_sign(tmp_path):
    """--ref-based orients to REF; --r exposes the resulting sign.

    The two datasets record the same physical variant with swapped REF/ALT.
    Dataset 1 has ALT=G at 11/12 copies so its major allele is ALT (G), while
    dataset 2 has ALT=A at 2/12 copies so its major allele is REF (G). Switching
    to reference orientation therefore mirrors only dataset 1, which flips the
    sign of r while leaving r^2 untouched.
    """

    ids = [f"S{i}" for i in range(6)]
    variants_a = [("1", "1:100:A:G", 100, "A", "G")]
    variants_b = [("1", "1:100:G:A", 100, "G", "A")]
    g_target = [2, 2, 2, 2, 2, 1]
    g_other = [0, 0, 1, 1, 0, 0]
    prefix_a = make_bfile(
        tmp_path, "ref_a", np.array([g_target], dtype=np.int8), ids, variants_a
    )
    prefix_b = make_bfile(
        tmp_path, "ref_b", np.array([g_other], dtype=np.int8), ids, variants_b
    )

    common = [
        "--bfile1",
        prefix_a,
        "--bfile2",
        prefix_b,
        "--ld-snp",
        "1:100:A:G",
        "--ld-window",
        "1",
        "--ld-window-kb",
        "10",
        "--ld-window-r2",
        "0",
        "--r",
        "--out",
    ]

    major_path = tmp_path / "major.vcor"
    assert cli_main(common + [str(major_path)]) == 0
    major_header, major = _read_rows(major_path)
    assert major_header.split("\t")[3] == "MAJ_A"
    assert major_header.split("\t")[7] == "MAJ_B"
    assert len(major) == 1
    assert major[0][6] == "1:100:G:A"
    assert float(major[0][8]) == pytest.approx(-0.316228, abs=1e-5)

    ref_path = tmp_path / "ref.vcor"
    assert cli_main(common + [str(ref_path), "--ref-based"]) == 0
    ref_header, ref = _read_rows(ref_path)
    assert ref_header.split("\t")[3] == "REF_A"
    assert ref_header.split("\t")[7] == "REF_B"
    assert float(ref[0][8]) == pytest.approx(0.316228, abs=1e-5)
    assert float(ref[0][8]) == pytest.approx(-float(major[0][8]))
    # The coding alleles differ between the two orientations on the A side.
    assert major[0][3] != ref[0][3]


def test_r2_threshold_is_inclusive(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, out_path, target="1:100:A:G"),
        "--ld-window-r2",
        "1.0",
    )
    assert cli_main(args) == 0
    # The comparison is inclusive: r^2 == 1 passes a threshold of 1.
    assert {row[5] for row in _read_rows(out_path)[1]} == {"1:100:G:A", "1:150:A:G"}
    assert all(float(row[6]) == 1.0 for row in _read_rows(out_path)[1])

    # 0 disables the r^2 filter entirely (it does not mean "r^2 >= 0").
    strict = tmp_path / "strict.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, strict, target="1:100:A:G"),
        "--ld-window-r2",
        "0",
    )
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(strict)[1]} == {
        "1:100:G:A",
        "1:150:A:G",
        "1:250:T:C",
    }

    # An intermediate threshold keeps only the two high-LD pairs.
    mid = tmp_path / "mid.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, mid, target="1:100:A:G"),
        "--ld-window-r2",
        "0.5",
    )
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(mid)[1]} == {"1:100:G:A", "1:150:A:G"}

    # A threshold just above the weakest pair's r^2 (0.0517241) drops it.
    just_above = tmp_path / "just_above.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, just_above, target="1:100:A:G"),
        "--ld-window-r2",
        "0.0517242",
    )
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(just_above)[1]} == {"1:100:G:A", "1:150:A:G"}

    # At exactly the weakest pair's r^2 the inclusive comparison keeps it.
    at_boundary = tmp_path / "at_boundary.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, at_boundary, target="1:100:A:G"),
        "--ld-window-r2",
        "0.0517241",
    )
    assert cli_main(args) == 0
    assert len(_read_rows(at_boundary)[1]) == 3


def test_window_count_is_max_variants_per_side(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair

    zero = tmp_path / "zero.vcor"
    args = _set_arg(_base_args(prefix_a, prefix_b, zero), "--ld-window", "0")
    assert cli_main(args) == 0
    assert len(_read_rows(zero)[1]) == 3  # 0 means no count limit

    one = tmp_path / "one.vcor"
    args = _set_arg(_base_args(prefix_a, prefix_b, one), "--ld-window", "1")
    assert cli_main(args) == 0
    # The target is dataset-2's first variant, so only one variant lies to its
    # right within a 1-variant window.
    assert {row[5] for row in _read_rows(one)[1]} == {"1:100:G:A", "1:150:A:G"}

    two = tmp_path / "two.vcor"
    args = _set_arg(_base_args(prefix_a, prefix_b, two), "--ld-window", "2")
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(two)[1]} == {
        "1:100:G:A",
        "1:150:A:G",
        "1:250:T:C",
    }


def test_ld_window_zero_means_no_count_limit(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    args = _set_arg(
        _base_args(prefix_a, prefix_b, out_path), "--ld-window", "0"
    )
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == {
        "1:100:G:A",
        "1:150:A:G",
        "1:250:T:C",
    }


def test_ld_window_takes_precedence_over_kb_window(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    # Include every variant by kb, then restrict by count.
    args = _set_arg(
        _set_arg(_base_args(prefix_a, prefix_b, out_path), "--ld-window", "2"),
        "--ld-window-kb",
        "1000",
    )
    assert cli_main(args) == 0
    assert len(_read_rows(out_path)[1]) == 3


def test_extract_and_exclude_shrink_results(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    out_path = tmp_path / "out.vcor"
    extract_path = tmp_path / "extract.txt"
    write_extract(extract_path, ["1:150:A:G"])
    args = _base_args(prefix_a, prefix_b, out_path) + ["--extract", str(extract_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == {"1:150:A:G"}

    out_exclude = tmp_path / "exclude.vcor"
    exclude_path = tmp_path / "exclude.txt"
    write_extract(exclude_path, ["1:150:A:G"])
    args = _base_args(prefix_a, prefix_b, out_exclude) + ["--exclude", str(exclude_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_exclude)[1]} == {"1:100:G:A", "1:250:T:C"}


def test_extract_bed_and_exclude_bed(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    bed_path = tmp_path / "region.bed"
    write_bed_intervals(bed_path, [("1", 100, 151)])

    out_extract = tmp_path / "extract.vcor"
    args = _base_args(prefix_a, prefix_b, out_extract) + ["--extract-bed", str(bed_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_extract)[1]} == {
        "1:100:G:A",
        "1:150:A:G",
    }

    out_exclude = tmp_path / "exclude.vcor"
    args = _base_args(prefix_a, prefix_b, out_exclude) + ["--exclude-bed", str(bed_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_exclude)[1]} == {"1:250:T:C"}


def test_dosage_mode_matches_hardcalls_on_hardcall_data(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    hard = tmp_path / "hard.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, hard)) == 0
    dos = tmp_path / "dos.vcor"
    args = _base_args(prefix_a, prefix_b, dos) + ["--dosage"]
    assert cli_main(args) == 0
    assert hard.read_text() == dos.read_text()


def test_dosage1_only_flips_orientation(tiny_pair, tmp_path):
    """--dosage1/--dosage2 must not change results on hardcall-only data."""

    prefix_a, prefix_b = tiny_pair
    out1 = tmp_path / "d1.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out1) + ["--dosage1"]) == 0
    out2 = tmp_path / "d2.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, out2) + ["--dosage2"]) == 0
    assert _read_rows(out1)[1] == _read_rows(out2)[1]
    assert len(_read_rows(out1)[1]) == 3


def test_keep_subset_changes_shared_samples(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("#IID\nS0\nS1\nS2\nS3\n", encoding="utf-8")
    out_path = tmp_path / "keep.vcor"
    args = _base_args(prefix_a, prefix_b, out_path) + ["--keep", str(keep_path)]
    assert cli_main(args) == 0
    assert {row[5] for row in _read_rows(out_path)[1]} == {
        "1:100:G:A",
        "1:150:A:G",
        "1:250:T:C",
    }


def test_sparse_never_matches_sparse_auto(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    auto = tmp_path / "auto.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, auto)) == 0
    never = tmp_path / "never.vcor"
    assert cli_main(_base_args(prefix_a, prefix_b, never) + ["--sparse", "never"]) == 0
    assert auto.read_text() == never.read_text()


def test_missing_dataset_file_exits_2(tmp_path, capsys):
    assert cli_main(["--bfile1", str(tmp_path / "nope"), "--bfile2", str(tmp_path / "nope")]) == 2


def test_no_shared_samples_exits_2(tmp_path):
    variants = [("1", "1:100:A:G", 100, "A", "G")]
    genotypes = np.array([[0, 1, 2]], dtype=np.int8)
    prefix_a = make_bfile(tmp_path, "a", genotypes, ["A1", "A2", "A3"], variants)
    prefix_b = make_bfile(tmp_path, "b", genotypes, ["B1", "B2", "B3"], variants)
    assert (
        cli_main(
            [
                "--bfile1",
                prefix_a,
                "--bfile2",
                prefix_b,
                "--ld-snp",
                "1:100:A:G",
                "--out",
                str(tmp_path / "out.vcor"),
            ]
        )
        == 2
    )


def test_negative_ld_window_is_rejected(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    args = _set_arg(_base_args(prefix_a, prefix_b, tmp_path / "out.vcor"), "--ld-window", "-1")
    assert cli_main(args) == 2


def test_ld_window_r2_out_of_range_is_rejected(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    args = _base_args(prefix_a, prefix_b, tmp_path / "out.vcor")
    idx = args.index("--ld-window-r2")
    args[idx + 1] = "1.5"
    assert cli_main(args) == 2
    args = _set_arg(
        _base_args(prefix_a, prefix_b, tmp_path / "out.vcor"), "--ld-window-r2", "-1"
    )
    assert cli_main(args) == 2


def test_negative_ld_window_kb_is_rejected(tiny_pair, tmp_path):
    prefix_a, prefix_b = tiny_pair
    args = _base_args(prefix_a, prefix_b, tmp_path / "out.vcor")
    idx = args.index("--ld-window-kb")
    args[idx + 1] = "-1"
    assert cli_main(args) == 2


def test_verbose_and_quiet_conflict(tmp_path, capsys):
    from crossld.plinkio import CrossLDError

    with pytest.raises((SystemExit, CrossLDError)):
        build_parser().parse_args(["-v", "-q"])


def test_output_defaults_to_stdout(tiny_pair, capsys):
    prefix_a, prefix_b = tiny_pair
    args = _base_args(prefix_a, prefix_b, Path("/dev/null"))
    idx = args.index("--out")
    del args[idx : idx + 2]
    assert cli_main(args) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("#CHROM_A\tPOS_A\tID_A")
    assert len(captured.out.splitlines()) == 4
