"""Tests for dataset resolution, sample selection, and target/window resolution."""

from __future__ import annotations

import numpy as np
import pytest

from crossld.plinkio import (
    PlinkIOError,
    candidate_indices,
    load_dataset,
    load_two_datasets,
    parse_sample,
    read_keep_file,
    read_ld_snp_list,
    read_ld_snps,
    resolve_targets,
    select_samples,
)
from tests.conftest import make_bfile, make_pfile, write_psam, write_pvar

SAMPLES = [f"S{i}" for i in range(6)]
DS1_VARIANTS = [
    ("1", "1:1000:A:G", 1000, "A", "G"),
    ("2", "2:1000:C:T", 1000, "C", "T"),
]
DS2_POS = [500, 900, 1000, 1100, 1500, 2000]
DS2_VARIANTS = [
    ("1", f"1:{p}:A:G", p, "A", "G") for p in DS2_POS
] + [("2", "2:1000:C:T", 1000, "C", "T")]


def _genotypes(n_variants: int, n_samples: int = len(SAMPLES)) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 3, size=(n_variants, n_samples)).astype(np.int8)


def _chrom_variants(chroms: list[str], positions: list[int]) -> list[tuple]:
    """Variant tuples with distinct IDs for an arbitrary chrom/pos layout."""

    return [
        (chrom, f"{chrom}:{pos}:A:G", pos, "A", "G")
        for chrom, pos in zip(chroms, positions)
    ]


@pytest.fixture()
def synthetic_pair(tmp_path):
    """A dataset-1 with two targets and a dataset-2 with known positions."""

    prefix1 = make_pfile(
        tmp_path, "ds1", _genotypes(len(DS1_VARIANTS)), SAMPLES, DS1_VARIANTS
    )
    prefix2 = make_pfile(
        tmp_path, "ds2", _genotypes(len(DS2_VARIANTS)), SAMPLES, DS2_VARIANTS
    )
    ds1 = load_dataset(prefix1, "pfile", np.array(SAMPLES))
    ds2 = load_dataset(prefix2, "pfile", np.array(SAMPLES))
    ds2.sort_pos_by_chrom()
    yield ds1, ds2
    ds1.close()
    ds2.close()


# --------------------------------------------------------------------------- #
# sample-file parsing
# --------------------------------------------------------------------------- #

def test_parse_fam_and_psam(tmp_path):
    fam = tmp_path / "x.fam"
    fam.write_text("F1\tI1\t0\t0\t0\t-9\nF2\tI2\t0\t0\t0\t-9\n", encoding="utf-8")
    assert list(parse_sample(str(fam), "bfile")) == ["I1", "I2"]

    psam = tmp_path / "x.psam"
    psam.write_text("#IID\tSEX\nI1\t1\nI2\t2\n", encoding="utf-8")
    assert list(parse_sample(str(psam), "pfile")) == ["I1", "I2"]


def test_parse_psam_finds_iid_column_by_name(tmp_path):
    psam = tmp_path / "x.psam"
    psam.write_text("#FID\tIID\tSEX\nF1\tI1\t1\nF2\tI2\t2\n", encoding="utf-8")
    assert list(parse_sample(str(psam), "pfile")) == ["I1", "I2"]

    bad = tmp_path / "bad.psam"
    bad.write_text("#FID\tSEX\nF1\t1\n", encoding="utf-8")
    with pytest.raises(PlinkIOError, match="no IID column"):
        parse_sample(str(bad), "pfile")


def test_parse_sample_rejects_empty_duplicate_and_malformed(tmp_path):
    empty = tmp_path / "empty.fam"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(PlinkIOError, match="empty sample file"):
        parse_sample(str(empty), "bfile")

    dup = tmp_path / "dup.fam"
    dup.write_text("F1\tI1\t0\t0\t0\t-9\nF2\tI1\t0\t0\t0\t-9\n", encoding="utf-8")
    with pytest.raises(PlinkIOError, match="duplicate sample"):
        parse_sample(str(dup), "bfile")

    malformed = tmp_path / "bad.fam"
    malformed.write_text("F1\tI1\t0\n", encoding="utf-8")
    with pytest.raises(PlinkIOError, match="malformed sample line"):
        parse_sample(str(malformed), "bfile")

    with pytest.raises(PlinkIOError, match="unknown dataset type"):
        parse_sample(str(dup), "nope")


def test_read_keep_file_ignores_header_blank_and_comments(tmp_path):
    path = tmp_path / "keep.txt"
    path.write_text("#IID\n\nS2\nS1\nS2\n", encoding="utf-8")
    assert list(read_keep_file(str(path))) == ["S1", "S2"]


def test_read_keep_file_requires_one_column(tmp_path):
    path = tmp_path / "keep.txt"
    path.write_text("S1\tSEX\n", encoding="utf-8")
    with pytest.raises(PlinkIOError, match="single-column"):
        read_keep_file(str(path))


# --------------------------------------------------------------------------- #
# sample selection
# --------------------------------------------------------------------------- #

def test_select_samples_is_the_intersection():
    samples1 = np.array(["A", "B", "C", "D"])
    samples2 = np.array(["B", "C", "D", "E"])
    assert list(select_samples(samples1, samples2)) == ["B", "C", "D"]


def test_select_samples_keep_narrows_and_requires_three():
    samples1 = np.array(["A", "B", "C", "D"])
    samples2 = np.array(["B", "C", "D", "E"])
    keep = np.array(["B", "C", "D"])
    assert list(select_samples(samples1, samples2, keep)) == ["B", "C", "D"]
    with pytest.raises(PlinkIOError, match="[Aa]t least 3"):
        select_samples(samples1, samples2, np.array(["B", "C"]))


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #

def test_load_dataset_reads_metadata(synthetic_pair):
    ds1, _ = synthetic_pair
    assert ds1.fmt == "pfile"
    assert ds1.variant_ct == 2 and ds1.sample_ct == 6
    assert ds1.active_sample_ct == 6
    assert ds1.ids == ["1:1000:A:G", "2:1000:C:T"]
    assert ds1.chrom.names == ["1", "2"]
    assert [ds1.chrom[i] for i in range(ds1.variant_ct)] == ["1", "2"]
    assert list(ds1.pos) == [1000, 1000]
    assert ds1.ref_allele(0) == "A" and ds1.alt_allele(0) == "G"
    assert ds1.index_of("2:1000:C:T") == 1
    assert ds1.index_of("nope") is None


def test_load_dataset_rejects_missing_files_and_bad_format(tmp_path):
    with pytest.raises(PlinkIOError, match="missing file"):
        load_dataset(str(tmp_path / "nope"), "bfile", np.array(SAMPLES))
    with pytest.raises(PlinkIOError, match="missing file"):
        load_dataset(str(tmp_path / "nope"), "pfile", np.array(SAMPLES))
    with pytest.raises(PlinkIOError, match="unknown format"):
        load_dataset(str(tmp_path / "nope"), "bogus", np.array(SAMPLES))


def test_load_dataset_rejects_multiallelic(tmp_path):
    prefix = tmp_path / "multi"
    write_psam(tmp_path / "multi.psam", SAMPLES)
    (tmp_path / "multi.pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n1\t100\t1:100:A:G,T\tA\tG,T\n", encoding="utf-8"
    )
    import pgenlib

    writer = pgenlib.PgenWriter(
        f"{prefix}.pgen".encode(), len(SAMPLES), variant_ct=1, nonref_flags=None
    )
    writer.append_biallelic_batch(_genotypes(1))
    writer.close()
    with pytest.raises(PlinkIOError, match="multiallelic"):
        load_dataset(str(prefix), "pfile", np.array(SAMPLES))


def test_load_dataset_sample_count_mismatch(tmp_path):
    prefix = make_pfile(tmp_path, "pf", _genotypes(1), SAMPLES, DS1_VARIANTS[:1])
    with pytest.raises(PlinkIOError, match="describes"):
        load_dataset(prefix, "pfile", np.array(SAMPLES[:5]))


def test_idx_bridge_permutes_dataset1_into_dataset2_order(loaded_pair):
    """The real fixtures list their samples in different orders."""

    ds1, ds2 = loaded_pair
    bridge = ds1.idx_bridge
    assert bridge is not None
    assert ds1.active_sample_ct == ds2.active_sample_ct
    mapped = ds1.sample_ids[ds1.sample_subset][bridge]
    assert list(mapped) == list(ds2.sample_ids[ds2.sample_subset])


def test_load_two_datasets_requires_three_shared_samples(tmp_path):
    variants = [("1", "1:100:A:G", 100, "A", "G")]
    prefix1 = make_bfile(tmp_path, "a", _genotypes(1, 3), ["A1", "A2", "A3"], variants)
    prefix2 = make_bfile(tmp_path, "b", _genotypes(1, 3), ["B1", "B2", "B3"], variants)
    with pytest.raises(PlinkIOError, match="[Aa]t least 3"):
        load_two_datasets(prefix1, "bfile", False, prefix2, "bfile", False)


# --------------------------------------------------------------------------- #
# target resolution
# --------------------------------------------------------------------------- #

def test_read_ld_snps_splits_on_commas():
    assert read_ld_snps(" a , ,b ") == ["a", "b"]
    assert read_ld_snps("") == []


def test_read_ld_snp_list_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# comment\n\n1:1000:A:G\n1:2000:C:T\n", encoding="utf-8")
    assert read_ld_snp_list(str(path)) == ["1:1000:A:G", "1:2000:C:T"]


def test_resolve_targets_deduplicates_and_reports_unmatched(synthetic_pair):
    ds1, _ = synthetic_pair
    # A1 regression: a repeated ID must be resolved once (commit ccb1da7).
    assert resolve_targets(ds1, snps="1:1000:A:G,1:1000:A:G") == [0]
    assert resolve_targets(ds1, snps="1:1000:A:G,2:1000:C:T") == [0, 1]
    assert resolve_targets(ds1, single="2:1000:C:T") == [1]
    assert resolve_targets(ds1, single="not-a-variant") == []


def test_resolve_targets_uses_dataset1_order(synthetic_pair):
    ds1, _ = synthetic_pair
    # Requested 2:1000:C:T first, but output follows dataset-1 variant order.
    assert resolve_targets(ds1, snps="2:1000:C:T,1:1000:A:G") == [0, 1]


def test_resolve_targets_merges_all_sources(synthetic_pair, tmp_path):
    ds1, _ = synthetic_pair
    path = tmp_path / "list.txt"
    path.write_text("2:1000:C:T\n", encoding="utf-8")
    assert resolve_targets(
        ds1, single="1:1000:A:G", snps="1:1000:A:G", snp_list=str(path)
    ) == [0, 1]


# --------------------------------------------------------------------------- #
# variant filter + sorted index
# --------------------------------------------------------------------------- #

def test_apply_variant_filter_restricts_sorted_index(synthetic_pair):
    ds1, ds2 = synthetic_pair
    mask = np.zeros(ds2.variant_ct, dtype=bool)
    mask[[1, 3]] = True  # positions 900 and 1100
    ds2.apply_variant_filter(mask)

    pos, idx = ds2.get_sorted_index("1")
    assert list(idx) == [1, 3]
    assert list(pos) == [900, 1100]

    pos_all, idx_all = ds2.get_sorted_index("1", filtered=False)
    assert len(idx_all) == len(DS2_POS)
    assert list(pos_all) == sorted(DS2_POS)
    assert ds2.get_sorted_index("9") is None


def test_apply_variant_filter_checks_shape(synthetic_pair):
    _, ds2 = synthetic_pair
    with pytest.raises(PlinkIOError, match="variant filter has shape"):
        ds2.apply_variant_filter(np.ones(3, dtype=bool))


def test_sort_pos_by_chrom_is_idempotent(synthetic_pair):
    _, ds2 = synthetic_pair
    before = ds2._group_sort
    ds2.sort_pos_by_chrom()
    assert ds2._group_sort is before


# --------------------------------------------------------------------------- #
# chromosome index
# --------------------------------------------------------------------------- #

def test_chrom_index_contiguous_blocks(tmp_path):
    variants = _chrom_variants(["1", "1", "2", "2", "2"], [20, 10, 5, 15, 1])
    prefix = make_pfile(tmp_path, "contig", _genotypes(5), SAMPLES, variants)
    ds = load_dataset(prefix, "pfile", np.array(SAMPLES))
    try:
        assert ds.chrom.names == ["1", "2"]
        assert ds.chrom.n_chroms == 2
        assert len(ds.chrom) == 5
        assert [ds.chrom[i] for i in range(5)] == ["1", "1", "2", "2", "2"]
        assert ds.chrom.name(4) == "2"
        assert ds.chrom.block("1") == (0, 2)
        assert ds.chrom.block("2") == (2, 5)
        assert ds.chrom.block("9") is None
        assert list(ds.chrom.indices("2")) == [2, 3, 4]
        assert ds.chrom.indices("9") is None
    finally:
        ds.close()


def test_chrom_index_interleaved_uses_codes(tmp_path):
    variants = _chrom_variants(["1", "2", "1", "2"], [1, 2, 3, 4])
    prefix = make_pfile(tmp_path, "inter", _genotypes(4), SAMPLES, variants)
    ds = load_dataset(prefix, "pfile", np.array(SAMPLES))
    try:
        assert ds.chrom.names == ["1", "2"]
        assert ds.chrom._codes is not None  # interleaved -> code-array fallback
        assert ds.chrom.block("1") is None
        assert list(ds.chrom.indices("1")) == [0, 2]
        assert list(ds.chrom.indices("2")) == [1, 3]
        assert [ds.chrom[i] for i in range(4)] == ["1", "2", "1", "2"]
    finally:
        ds.close()


def test_sort_pos_by_chrom_sorts_unsorted_blocks(tmp_path):
    variants = _chrom_variants(["1", "1", "1", "2"], [30, 10, 20, 5])
    prefix = make_pfile(tmp_path, "unsorted", _genotypes(4), SAMPLES, variants)
    ds = load_dataset(prefix, "pfile", np.array(SAMPLES))
    try:
        ds.sort_pos_by_chrom()
        pos, idx = ds.get_sorted_index("1")
        assert list(pos) == [10, 20, 30]
        assert list(idx) == [1, 2, 0]
        assert list(ds.get_sorted_index("2")[0]) == [5]
    finally:
        ds.close()


def test_get_sorted_index_interleaved(tmp_path):
    variants = _chrom_variants(["1", "2", "1", "2"], [30, 2, 10, 1])
    prefix = make_pfile(tmp_path, "ixsort", _genotypes(4), SAMPLES, variants)
    ds = load_dataset(prefix, "pfile", np.array(SAMPLES))
    try:
        ds.sort_pos_by_chrom()
        pos, idx = ds.get_sorted_index("1")
        assert list(pos) == [10, 30]
        assert list(idx) == [2, 0]
    finally:
        ds.close()


def test_candidate_indices_interleaved_are_ascending(tmp_path):
    prefix_a = make_pfile(
        tmp_path, "cia", _genotypes(1), SAMPLES, _chrom_variants(["1"], [20])
    )
    prefix_b = make_pfile(
        tmp_path,
        "cib",
        _genotypes(4),
        SAMPLES,
        _chrom_variants(["1", "2", "1", "2"], [10, 20, 30, 40]),
    )
    ds1 = load_dataset(prefix_a, "pfile", np.array(SAMPLES))
    ds2 = load_dataset(prefix_b, "pfile", np.array(SAMPLES))
    try:
        ds2.sort_pos_by_chrom()
        candidates = candidate_indices(ds1, ds2, 0, 10.0, 0)
        assert list(candidates) == [0, 2]
    finally:
        ds1.close()
        ds2.close()


# --------------------------------------------------------------------------- #
# candidate windows
# --------------------------------------------------------------------------- #

def test_candidate_window_respects_kb_distance(synthetic_pair):
    ds1, ds2 = synthetic_pair
    narrow = candidate_indices(ds1, ds2, 0, 0.2, 0)  # +/- 200 bp around 1000
    assert list(narrow) == [1, 2, 3]  # positions 900, 1000, 1100
    assert [int(ds2.pos[i]) for i in narrow] == [900, 1000, 1100]

    exact = candidate_indices(ds1, ds2, 0, 0.0, 0)
    assert [int(ds2.pos[i]) for i in exact] == [1000]

    wide = candidate_indices(ds1, ds2, 0, 10.0, 0)
    assert [int(ds2.pos[i]) for i in wide] == DS2_POS


def test_candidate_window_count_keeps_nearest(synthetic_pair):
    ds1, ds2 = synthetic_pair
    one = candidate_indices(ds1, ds2, 0, 10.0, 1)
    assert [int(ds2.pos[i]) for i in one] == [1000]

    two = candidate_indices(ds1, ds2, 0, 10.0, 2)
    assert [int(ds2.pos[i]) for i in two] == [900, 1000]


def test_candidate_window_uses_filtered_variants(synthetic_pair):
    ds1, ds2 = synthetic_pair
    mask = np.zeros(ds2.variant_ct, dtype=bool)
    mask[[1, 3]] = True
    ds2.apply_variant_filter(mask)
    candidates = candidate_indices(ds1, ds2, 0, 10.0, 0)
    assert [int(ds2.pos[i]) for i in candidates] == [900, 1100]


def test_candidate_window_for_missing_chrom_is_empty(synthetic_pair):
    ds1, ds2 = synthetic_pair
    mask = np.zeros(ds2.variant_ct, dtype=bool)
    mask[0] = True  # keep only a chrom-1 variant
    ds2.apply_variant_filter(mask)
    assert candidate_indices(ds1, ds2, 1, 10.0, 0).size == 0  # target is on chrom 2
