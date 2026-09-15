"""Tests for the output formatter in :mod:`crossld.results`."""

from __future__ import annotations

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

R2_ROW = "20\t114311\t20:114311:G:T\t20\t101498\t20:101498:A:G\t0.0910081"
R_ROW = "20\t114311\t20:114311:G:T\tG\t20\t101498\t20:101498:A:G\tA\t-0.301675"


def test_value_format_matches_reference_precision():
    assert VALUE_FORMAT == "%.6g"
    assert format_value(1.0) == "1"
    assert format_value(0.35567) == "0.35567"
    assert format_value(0.09100806) == "0.0910081"
    assert format_value(0.0505139) == "0.0505139"


@pytest.mark.parametrize(
    "with_r, expected",
    [(False, R2_ROW), (True, R_ROW)],
    ids=["r2", "r"],
)
def test_format_row_shapes(with_r, expected):
    assert format_row(**BASE, with_r=with_r) == expected


def test_r2_row_has_no_allele_columns():
    assert BASE["allele_a"] not in format_row(**BASE, with_r=False).split("\t")


@pytest.mark.parametrize(
    "with_r, label, header",
    [
        (False, "MAJ", HEADER_R2),
        (False, "REF", HEADER_R2),
        (True, "MAJ", HEADER_R_MAJOR),
        (True, "REF", HEADER_R_REF),
    ],
)
def test_header_for_selects_the_column_set(with_r, label, header):
    assert header_for(with_r=with_r, allele_label=label) == header


def test_header_for_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown allele label"):
        header_for(with_r=True, allele_label="NOPE")


def test_headers_have_the_expected_columns():
    assert HEADER_R2.split("\t") == [
        "#CHROM_A", "POS_A", "ID_A", "CHROM_B", "POS_B", "ID_B", "UNPHASED_R2",
    ]
    for header, label in ((HEADER_R_MAJOR, "MAJ"), (HEADER_R_REF, "REF")):
        assert header.split("\t") == [
            "#CHROM_A", "POS_A", "ID_A", f"{label}_A",
            "CHROM_B", "POS_B", "ID_B", f"{label}_B", "UNPHASED_R",
        ]


def test_write_header_selects_the_column_set(tmp_path):
    path = tmp_path / "out.vcor"
    with open(path, "w", encoding="utf-8") as handle:
        write_header(handle, with_r=False)
        write_header(handle, with_r=True, allele_label="MAJ")
        write_header(handle, with_r=True, allele_label="REF")
    assert path.read_text(encoding="utf-8").splitlines() == [
        HEADER_R2,
        HEADER_R_MAJOR,
        HEADER_R_REF,
    ]


def test_write_row_emits_a_trailing_newline(tmp_path):
    path = tmp_path / "out.vcor"
    with open(path, "w", encoding="utf-8") as handle:
        write_row(handle, **BASE, with_r=True)
    assert path.read_text(encoding="utf-8") == R_ROW + "\n"
