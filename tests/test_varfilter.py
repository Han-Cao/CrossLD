"""Tests for dataset-2 variant filtering (``--extract``/``--exclude`` and BED).

BED regions are 0-based half-open over the variant's 0-based coordinate, so a
variant at 1-based position ``p`` spans ``[p - 1, p)`` and is kept by an
interval only when ``START <= p - 1`` and ``p <= END``.
"""

from __future__ import annotations

import numpy as np
import pytest

from crossld.plinkio import PlinkIOError, load_dataset
from crossld.varfilter import (
    VarFilterError,
    VariantFilter,
    read_bed_file,
    read_vid_file,
)
from tests.conftest import make_pfile, write_bed_intervals, write_extract

SAMPLES = [f"S{i}" for i in range(6)]
VARIANTS = [
    ("1", "1:100:A:G", 100, "A", "G"),
    ("1", "1:200:A:G", 200, "A", "G"),
    ("1", "1:300:A:G", 300, "A", "G"),
    ("1", "1:400:A:G", 400, "A", "G"),
    ("2", "2:100:C:T", 100, "C", "T"),
    ("2", "2:200:C:T", 200, "C", "T"),
]


@pytest.fixture()
def ds2(tmp_path):
    rng = np.random.default_rng(0)
    genotypes = rng.integers(0, 3, size=(len(VARIANTS), len(SAMPLES))).astype(np.int8)
    prefix = make_pfile(tmp_path, "ds2", genotypes, SAMPLES, VARIANTS)
    dataset = load_dataset(prefix, "pfile", np.array(SAMPLES))
    yield dataset
    dataset.close()


def _kept_positions(ds, mask):
    return [int(ds.pos[i]) for i in np.flatnonzero(mask)]


# --------------------------------------------------------------------------- #
# option bookkeeping
# --------------------------------------------------------------------------- #

def test_no_filter_is_all_active(ds2):
    filt = VariantFilter()
    assert filt.active is False
    assert np.all(filt.build_mask(ds2))


def test_describe_lists_active_options():
    assert VariantFilter(extract="ids.txt").describe() == "--extract"
    assert (
        VariantFilter(extract="ids.txt", extract_bed="region.bed").describe()
        == "--extract and --extract-bed"
    )
    assert VariantFilter(exclude="x").describe() == "--exclude"
    assert VariantFilter().describe() == ""


def test_variant_filter_error_is_a_plink_io_error():
    assert issubclass(VarFilterError, PlinkIOError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extract": "a", "exclude": "b"},
        {"extract_bed": "a", "exclude_bed": "b"},
        {"extract": "a", "exclude_bed": "b"},
        {"exclude": "a", "extract_bed": "b"},
    ],
)
def test_mixed_polarities_are_rejected(kwargs):
    with pytest.raises(VarFilterError, match="cannot be combined"):
        VariantFilter(**kwargs).validate()


def test_same_polarity_union_is_allowed():
    VariantFilter(extract="a", extract_bed="b").validate()
    VariantFilter(exclude="a", exclude_bed="b").validate()


# --------------------------------------------------------------------------- #
# extract / exclude by ID
# --------------------------------------------------------------------------- #

def test_extract_keeps_only_listed_ids(ds2, tmp_path):
    path = tmp_path / "extract.txt"
    write_extract(path, ["1:200:A:G", "1:400:A:G"])
    mask = VariantFilter(extract=str(path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [200, 400]


def test_extract_reports_unmatched_ids(ds2, tmp_path, caplog):
    path = tmp_path / "extract.txt"
    write_extract(path, [ds2.ids[0], "definitely-not-a-variant"])
    with caplog.at_level("WARNING", logger="crossld.varfilter"):
        VariantFilter(extract=str(path)).build_mask(ds2)
    assert any("matched no dataset-2 variant" in r.getMessage() for r in caplog.records)
    assert "1 of 2" in caplog.text


def test_exclude_drops_listed_ids(ds2, tmp_path):
    path = tmp_path / "exclude.txt"
    write_extract(path, ["1:200:A:G", "2:200:C:T"])
    mask = VariantFilter(exclude=str(path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [100, 300, 400, 100]


def test_extract_with_no_matching_ids_errors(ds2, tmp_path):
    path = tmp_path / "none.txt"
    write_extract(path, ["definitely-not-here"])
    with pytest.raises(VarFilterError, match="left no variants"):
        VariantFilter(extract=str(path)).build_mask(ds2)


# --------------------------------------------------------------------------- #
# BED regions
# --------------------------------------------------------------------------- #

def test_extract_bed_is_zero_based_half_open(ds2, tmp_path):
    path = tmp_path / "region.bed"
    # Variant at 1-based 200 spans [199, 200); at 1-based 300 spans [299, 300).
    write_bed_intervals(path, [("1", 199, 300)])
    mask = VariantFilter(extract_bed=str(path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [200, 300]


def test_extract_bed_boundaries_are_half_open(ds2, tmp_path):
    path = tmp_path / "region.bed"
    # Variant at 1-based 200 spans [199, 200), so (199, 200) keeps it...
    write_bed_intervals(path, [("1", 199, 200)])
    assert _kept_positions(ds2, VariantFilter(extract_bed=str(path)).build_mask(ds2)) == [200]
    # ...while (200, 201) covers only 1-based 201 (absent) and not 200.
    write_bed_intervals(path, [("1", 200, 201), ("1", 399, 400)])
    assert _kept_positions(ds2, VariantFilter(extract_bed=str(path)).build_mask(ds2)) == [400]


def test_bed_accepts_chr_prefix(ds2, tmp_path):
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("chr1", 199, 200)])
    mask = VariantFilter(extract_bed=str(path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [200]


def test_exclude_bed_removes_region(ds2, tmp_path):
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("1", 199, 300)])  # removes 1-based 200 and 300
    mask = VariantFilter(exclude_bed=str(path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [100, 400, 100, 200]


def test_extract_union_with_bed(ds2, tmp_path):
    id_path = tmp_path / "ids.txt"
    write_extract(id_path, ["1:100:A:G"])
    bed_path = tmp_path / "region.bed"
    write_bed_intervals(bed_path, [("1", 299, 300)])  # adds 1-based 300
    mask = VariantFilter(extract=str(id_path), extract_bed=str(bed_path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [100, 300]


def test_exclude_and_exclude_bed_both_apply(ds2, tmp_path):
    """Regression: BED exclusion must not be skipped when --exclude is set."""

    id_path = tmp_path / "ids.txt"
    write_extract(id_path, ["1:200:A:G"])
    bed_path = tmp_path / "region.bed"
    write_bed_intervals(bed_path, [("1", 299, 300)])  # removes 1-based 300
    mask = VariantFilter(exclude=str(id_path), exclude_bed=str(bed_path)).build_mask(ds2)
    assert _kept_positions(ds2, mask) == [100, 400, 100, 200]


def test_exclude_everything_errors(ds2, tmp_path):
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("1", 0, 1000), ("2", 0, 1000)])
    with pytest.raises(VarFilterError, match="left no variants"):
        VariantFilter(exclude_bed=str(path)).build_mask(ds2)


# --------------------------------------------------------------------------- #
# file parsers
# --------------------------------------------------------------------------- #

def test_read_vid_file_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# comment\n\nrs1\nrs2\textra\n", encoding="utf-8")
    assert read_vid_file(str(path)) == ["rs1", "rs2"]


@pytest.mark.parametrize(
    "content, message",
    [
        ("20\t100\n", r"2 field"),
        ("20\tabc\t200\n", "non-integer"),
        ("20\t200\t100\n", "invalid 0-based"),
        ("20\t-1\t100\n", "invalid 0-based"),
    ],
)
def test_bed_parse_errors(tmp_path, content, message):
    path = tmp_path / "bad.bed"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(VarFilterError, match=message):
        read_bed_file(str(path))


def test_bed_blank_and_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "region.bed"
    path.write_text("# header\n\nchr20\t10\t20\n", encoding="utf-8")
    assert read_bed_file(str(path)) == [("20", 10, 20)]
