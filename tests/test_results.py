"""Tests for the output formatter and the module entry point."""

from __future__ import annotations

import runpy

import pytest

from crossld.results import (
    HEADER_R2,
    HEADER_R_MAJOR,
    HEADER_R_REF,
    VALUE_FORMAT,
    format_row,
    format_value,
    header_for,
    write_header,
    write_row,
)

BASE = {
    "chrom_a": "20",
    "pos_a": 114311,
    "id_a": "20:114311:G:T",
    "chrom_b": "20",
    "pos_b": 101498,
    "id_b": "20:101498:A:G",
    "r": -0.3016754,
    "r2": 0.09100806,
    "allele_a": "G",
    "allele_b": "A",
}


def test_value_format_matches_reference_precision():
    assert VALUE_FORMAT == "%.6g"
    assert format_value(1.0) == "1"
    assert format_value(0.35567) == "0.35567"
    assert format_value(0.09100806) == "0.0910081"
    assert format_value(0.0505139) == "0.0505139"


def test_r2_row_has_no_allele_columns():
    assert format_row(**BASE, with_r=False) == (
        "20\t114311\t20:114311:G:T\t20\t101498\t20:101498:A:G\t0.0910081"
    )


def test_r_row_reports_alleles_without_r2():
    expected = (
        "20\t114311\t20:114311:G:T\tG\t20\t101498\t20:101498:A:G\tA\t-0.301675"
    )
    assert format_row(**BASE, with_r=True, allele_label="MAJ") == expected
    assert format_row(**BASE, with_r=True, allele_label="REF") == expected


def test_header_for_selects_the_column_set():
    assert header_for(with_r=False) == HEADER_R2
    assert header_for(with_r=True, allele_label="MAJ") == HEADER_R_MAJOR
    assert header_for(with_r=True, allele_label="REF") == HEADER_R_REF
    with pytest.raises(ValueError):
        header_for(with_r=True, allele_label="NOPE")


def test_headers_have_the_expected_column_counts():
    assert HEADER_R2.split("\t") == [
        "#CHROM_A", "POS_A", "ID_A", "CHROM_B", "POS_B", "ID_B", "UNPHASED_R2",
    ]
    for header, label in ((HEADER_R_MAJOR, "MAJ"), (HEADER_R_REF, "REF")):
        assert header.split("\t") == [
            "#CHROM_A", "POS_A", "ID_A", f"{label}_A",
            "CHROM_B", "POS_B", "ID_B", f"{label}_B", "UNPHASED_R",
        ]


def test_write_header_selects_column_set(tmp_path):
    path = tmp_path / "out.vcor"
    with open(path, "w", encoding="utf-8") as handle:
        write_header(handle, with_r=False)
        write_header(handle, with_r=True, allele_label="MAJ")
        write_header(handle, with_r=True, allele_label="REF")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [HEADER_R2, HEADER_R_MAJOR, HEADER_R_REF]


def test_write_row_emits_a_trailing_newline(tmp_path):
    path = tmp_path / "out.vcor"
    with open(path, "w", encoding="utf-8") as handle:
        write_row(handle, **BASE, with_r=True, allele_label="MAJ")
    assert path.read_text(encoding="utf-8") == (
        "20\t114311\t20:114311:G:T\tG\t20\t101498\t20:101498:A:G\tA\t-0.301675\n"
    )


def test_python_m_entry_point_runs():
    """``python -m crossld`` must delegate to :func:`crossld.crossld.main`."""

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("crossld", run_name="__main__", alter_sys=True)
    # ``crossld --version``-free invocation exits through argparse's error path.
    assert excinfo.value.code != 0
