"""Tests for dataset resolution, sample selection, and target resolution."""

from __future__ import annotations

import numpy as np
import pytest

from crossld.plinkio import (
    CrossLDError,
    candidate_indices,
    expand_ld_snps,
    load_dataset,
    read_keep_file,
    resolve_targets,
    select_samples,
)
from tests.conftest import make_bfile


@pytest.fixture(scope="module")
def dataset_pair(sample1_prefix, sample2_prefix):
    ds1 = load_dataset(sample1_prefix, "dataset1", "pfile")
    ds2 = load_dataset(sample2_prefix, "dataset2", "pfile")
    yield ds1, ds2
    ds1.close()
    ds2.close()


def test_load_dataset_reads_metadata(dataset_pair):
    ds1, ds2 = dataset_pair
    assert ds1.variant_ct == 549
    assert ds2.variant_ct == 569
    assert ds1.sample_ct == ds2.sample_ct == 200
    assert ds1.chrom[0] == "20"
    assert ds1.pos[0] == 100014
    assert ds1.ids[0] == "20:100014:A:G"
    assert ds1.ref[0] == "A" and ds1.alt[0] == "G"


def test_default_selection_is_iid_intersection_in_dataset1_order(dataset_pair, fixture_dir):
    ds1, ds2 = dataset_pair
    subset1, subset2, summary = select_samples(ds1, ds2)
    overlap = [
        line.split()[0]
        for line in (fixture_dir / "1kg.sample_overlap.psam").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(subset1) == len(subset2) == len(overlap) == 100
    assert list(subset1) == sorted(subset1) and list(subset2) == sorted(subset2)
    selected = [ds1.sample_ids[i] for i in subset1]
    assert selected == [sid for sid in ds1.sample_ids if sid in set(overlap)]
    assert set(ds2.sample_ids[i] for i in subset2) == set(overlap)
    assert summary.shared == 100


def test_keep_with_header_and_missing_samples(dataset_pair, tmp_path):
    ds1, ds2 = dataset_pair
    overlap = [
        sid for sid in ds1.sample_ids if sid in set(ds2.sample_ids)
    ][:5]
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text(
        "#IID\tSEX\n" + "".join(f"{sid}\n" for sid in overlap) + "NOT_A_SAMPLE\n",
        encoding="utf-8",
    )
    subset1, subset2, summary = select_samples(ds1, ds2, keep=str(keep_path))
    assert [ds1.sample_ids[i] for i in subset1] == overlap
    assert summary.requested == 6
    assert summary.missing == ["NOT_A_SAMPLE"]


def test_keep_single_column_without_header(dataset_pair, tmp_path):
    ds1, ds2 = dataset_pair
    overlap = [sid for sid in ds1.sample_ids if sid in set(ds2.sample_ids)][:4]
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("\n".join(overlap) + "\n", encoding="utf-8")
    subset1, _, _ = select_samples(ds1, ds2, keep=str(keep_path))
    assert [ds1.sample_ids[i] for i in subset1] == overlap


def test_keep_multicolumn_without_header_is_an_error(tmp_path):
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("HG00096\tNA\n", encoding="utf-8")
    with pytest.raises(CrossLDError, match="single-column"):
        read_keep_file(str(keep_path))


def test_keep_with_fewer_than_three_shared_samples_is_an_error(dataset_pair, tmp_path):
    ds1, ds2 = dataset_pair
    keep_path = tmp_path / "keep.txt"
    keep_path.write_text("HG00096\nHG00097\n", encoding="utf-8")
    with pytest.raises(CrossLDError, match="at least 3"):
        select_samples(ds1, ds2, keep=str(keep_path))


def test_pfile_prefix_falls_back_to_bed(tmp_path):
    ids = [f"S{i}" for i in range(6)]
    variants = [("1", "1:100:A:G", 100, "A", "G")]
    genotypes = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.int8)
    prefix = make_bfile(tmp_path, "tiny", genotypes, ids, variants)
    dataset = load_dataset(prefix, "tiny", "pfile")
    try:
        assert dataset.variant_ct == 1
        assert dataset.ref == ["A"] and dataset.alt == ["G"]
    finally:
        dataset.close()


def test_resolve_targets_reports_unmatched(dataset_pair):
    ds1, _ = dataset_pair
    assert resolve_targets(ds1, single="20:114311:G:T") == [71]
    assert resolve_targets(ds1, single="not-a-variant") == []
    assert resolve_targets(ds1, snps="20:114311:G:T,20:144480:C:T") == [71, 222]
    assert resolve_targets(ds1, snps="20:100000-101000") == [0, 1]
    assert resolve_targets(ds1, snps="20:114311:G:T,20:114311:G:T") == [71]


def test_expand_ld_snps_range_forms(dataset_pair):
    ds1, _ = dataset_pair
    assert expand_ld_snps("20:101338-101500", ds1) == [2, 3, 4, 5, 6, 7]
    assert expand_ld_snps("20:101338-20:101341", ds1) == [2, 3]
    assert expand_ld_snps("bogus", ds1) == []


def test_candidate_window_respects_kb_distance(dataset_pair):
    ds1, ds2 = dataset_pair
    kept = np.flatnonzero(np.ones(ds2.variant_ct, dtype=bool))
    narrow = candidate_indices(ds1, ds2, 71, 1.0, 0, kept_indices=kept)
    wide = candidate_indices(ds1, ds2, 71, 100.0, 0, kept_indices=kept)
    assert len(narrow) < len(wide)
    target_pos = int(ds1.pos[71])
    assert all(abs(int(ds2.pos[i]) - target_pos) <= 1000 for i in narrow)


def test_candidate_window_count_is_per_side(dataset_pair):
    ds1, ds2 = dataset_pair
    kept = np.flatnonzero(np.ones(ds2.variant_ct, dtype=bool))
    unlimited = candidate_indices(ds1, ds2, 71, 1000.0, 0, kept_indices=kept)
    limited = candidate_indices(ds1, ds2, 71, 1000.0, 5, kept_indices=kept)
    assert len(unlimited) > len(limited)
    # At most 5 variants on each side of the target's own column.
    assert len(limited) <= 11

    one = candidate_indices(ds1, ds2, 71, 1000.0, 1, kept_indices=kept)
    assert len(one) <= 3
    rank = int(np.searchsorted(kept, 71))
    assert set(one.tolist()) <= set(kept[max(0, rank - 1) : rank + 2].tolist())


def test_candidate_window_uses_filtered_variants(dataset_pair):
    ds1, ds2 = dataset_pair
    kept = np.array([71, 100, 200, 300], dtype=np.int64)
    candidates = candidate_indices(ds1, ds2, 71, 100000.0, 1, kept_indices=kept)
    assert set(candidates.tolist()) <= set(kept.tolist())
    assert len(candidates) <= 2


def test_candidate_window_when_target_absent_from_dataset2(dataset_pair):
    ds1, ds2 = dataset_pair
    kept = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    # Target index 71 is not in the kept set; the window anchors on its
    # insertion point instead of failing.
    candidates = candidate_indices(ds1, ds2, 71, 1000.0, 1, kept_indices=kept)
    assert set(candidates.tolist()) <= set(kept.tolist())
