"""Tests for dataset-2 variant filtering (``--extract``/``--exclude`` and BED)."""

from __future__ import annotations

import numpy as np
import pytest

from crossld.plinkio import CrossLDError, load_dataset
from crossld.varfilter import VariantFilter, read_bed_file, read_id_file
from tests.conftest import write_bed_intervals, write_extract


@pytest.fixture(scope="module")
def ds2(sample2_prefix):
    dataset = load_dataset(sample2_prefix, "dataset2", "pfile")
    yield dataset
    dataset.close()


def test_no_filter_is_all_active():
    filt = VariantFilter()
    assert filt.active is False
    assert filt.describe() == "none"


def test_extract_keeps_only_listed_ids(ds2, tmp_path, extract_list):
    mask = VariantFilter(extract=extract_list).build_mask(ds2)
    kept = set(np.flatnonzero(mask).tolist())
    listed = {
        line.strip()
        for line in open(extract_list, encoding="utf-8")
        if line.strip()
    }
    expected = {i for i, vid in enumerate(ds2.ids) if vid in listed}
    assert kept == expected
    assert len(kept) == 537


def test_extract_reports_unmatched_ids(ds2, tmp_path, caplog):
    path = tmp_path / "extract.txt"
    write_extract(path, [ds2.ids[0], "definitely-not-a-variant"])
    with caplog.at_level("WARNING", logger="crossld.varfilter"):
        VariantFilter(extract=str(path)).build_mask(ds2)
    assert any("matched no variant" in record.message for record in caplog.records)
    assert "1 of 2" in caplog.text


def test_clean_extract_list_warns_about_nothing(ds2, extract_list, caplog):
    """The curated dataset-2 extract list matches every listed ID."""

    with caplog.at_level("WARNING", logger="crossld.varfilter"):
        mask = VariantFilter(extract=extract_list).build_mask(ds2)
    assert not [r for r in caplog.records if "matched no variant" in r.message]
    assert int(mask.sum()) == 537


def test_exclude_drops_listed_ids(ds2, tmp_path):
    ids = ds2.ids[:10]
    path = tmp_path / "exclude.txt"
    write_extract(path, ids)
    mask = VariantFilter(exclude=str(path)).build_mask(ds2)
    assert int(mask.sum()) == ds2.variant_ct - 10
    for i, vid in enumerate(ds2.ids[:10]):
        assert mask[i] == False  # noqa: E712 - explicit numpy bool check


def test_extract_bed_is_zero_based_half_open(ds2, tmp_path):
    target = ds2.ids.index("20:114311:G:T")
    pos = int(ds2.pos[target])
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("20", pos, pos + 1)])
    mask = VariantFilter(extract_bed=str(path)).build_mask(ds2)
    assert mask[target] == True  # noqa: E712
    assert int(mask.sum()) == 1

    # START == pos is kept; END == pos is dropped.
    write_bed_intervals(path, [("20", pos, pos)])
    with pytest.raises(CrossLDError, match="invalid 0-based half-open interval"):
        read_bed_file(str(path))


def test_bed_accepts_chr_prefix(ds2, tmp_path):
    target = ds2.ids.index("20:114311:G:T")
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("chr20", int(ds2.pos[target]), int(ds2.pos[target]) + 1)])
    mask = VariantFilter(extract_bed=str(path)).build_mask(ds2)
    assert mask[target] == True  # noqa: E712


def test_exclude_bed_removes_region(ds2, tmp_path):
    path = tmp_path / "region.bed"
    write_bed_intervals(path, [("20", 100000, 110000)])
    mask = VariantFilter(exclude_bed=str(path)).build_mask(ds2)
    removed = [i for i, pos in enumerate(ds2.pos) if 100000 <= int(pos) < 110000]
    assert removed
    for i in removed:
        assert mask[i] == False  # noqa: E712
    assert int(mask.sum()) == ds2.variant_ct - len(removed)


def test_bed_requires_three_fields(tmp_path):
    path = tmp_path / "bad.bed"
    path.write_text("20\t100\n", encoding="utf-8")
    with pytest.raises(CrossLDError, match=r"2 field"):
        read_bed_file(str(path))


def test_bed_rejects_non_integer_bounds(tmp_path):
    path = tmp_path / "bad.bed"
    path.write_text("20\tabc\t200\n", encoding="utf-8")
    with pytest.raises(CrossLDError, match="non-integer"):
        read_bed_file(str(path))


def test_id_file_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# comment\n\nrs1\nrs2 extra\n", encoding="utf-8")
    ids, unmatched = read_id_file(str(path))
    assert ids == ["rs1", "rs2"]
    assert unmatched == []


def test_mutual_exclusion_rules():
    with pytest.raises(CrossLDError, match="mutually exclusive"):
        VariantFilter(extract="a", exclude="b").validate()
    with pytest.raises(CrossLDError, match="mutually exclusive"):
        VariantFilter(extract_bed="a", exclude_bed="b").validate()
    with pytest.raises(CrossLDError, match="cannot be combined"):
        VariantFilter(extract="a", exclude_bed="b").validate()
    VariantFilter(extract="a", extract_bed="b").validate()  # union is allowed


def test_extract_with_no_matching_ids_errors(ds2, tmp_path):
    path = tmp_path / "none.txt"
    write_extract(path, ["definitely-not-here"])
    with pytest.raises(CrossLDError, match="left no variants"):
        VariantFilter(extract=str(path)).build_mask(ds2)


def test_extract_union_with_bed(ds2, tmp_path):
    id_path = tmp_path / "ids.txt"
    write_extract(id_path, [ds2.ids[0]])
    target = 5
    pos = int(ds2.pos[target])
    bed_path = tmp_path / "region.bed"
    write_bed_intervals(bed_path, [("20", pos, pos + 1)])
    mask = VariantFilter(extract=str(id_path), extract_bed=str(bed_path)).build_mask(ds2)
    assert int(mask.sum()) == 2
    assert mask[0] and mask[target]


def test_describe_lists_active_options():
    filt = VariantFilter(extract="ids.txt", extract_bed="region.bed")
    assert "--extract=ids.txt" in filt.describe()
    assert "--extract-bed=region.bed" in filt.describe()


def test_bed_blank_and_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "region.bed"
    path.write_text("# header\n\n20\t10\t20\n", encoding="utf-8")
    assert read_bed_file(str(path)) == [("20", 10, 20)]
