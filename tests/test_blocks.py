"""Unit tests for the block-wise reader in :mod:`crossld.blocks`.

Synthetic ``.pgen`` data is used so the dense/sparse partition is deterministic:
``pgenlib`` stores a variant in the sparse format when only a couple of samples
carry a non-reference call (verified for ``PgenWriter``), which lets each test
pick exactly which rows land in ``dense`` and which in ``sparse``.
"""

from __future__ import annotations

import numpy as np
import pytest

from crossld.blocks import BlockReader, BlockStats
from crossld.plinkio import load_dataset
from tests.conftest import make_bfile, make_dosage_pfile, make_pfile

SAMPLE_IDS = [f"S{i}" for i in range(20)]

# v0/v2/v4 are sparse-eligible (<=2 non-reference calls out of 20), v1/v3/v5 are
# dense (pgenlib only stores very sparse variants in the sparse format).
GENOTYPES = np.array(
    [
        [0] * 20,  # 0 sparse, constant REF
        [0, 0, 1, 1, 2, 2] + [0] * 14,  # 1 dense
        [2] + [0] * 19,  # 2 sparse, single ALT
        ([0, 1, 0, 2, 0, 1, 0, 2] * 2) + [0] * 4,  # 3 dense
        [2] + [0] * 18 + [-9],  # 4 sparse, ALT + missing
        [1] * 20,  # 5 dense, constant het
    ],
    dtype=np.int8,
)
VARIANTS = [
    ("1", f"1:{100 * (i + 1)}:A:G", 100 * (i + 1), "A", "G")
    for i in range(GENOTYPES.shape[0])
]
SPARSE_VARIANTS = [0, 2, 4]
DENSE_VARIANTS = [1, 3, 5]


def _load(tmp_path, name="blocks", genotypes=GENOTYPES, variants=VARIANTS, **kwargs):
    prefix = make_pfile(tmp_path, name, genotypes, SAMPLE_IDS, variants)
    return load_dataset(prefix, "pfile", np.array(SAMPLE_IDS), **kwargs)


def _reconstruct(block, sample_ct: int) -> np.ndarray:
    """Rebuild the full matrix from the dense/sparse split."""

    out = np.zeros((block.stats.variants, sample_ct), dtype=np.float64)
    if block.dense_rows.size:
        out[block.dense_rows] = block.dense
    for i, position in enumerate(block.sparse_rows.tolist()):
        out[position, block.sparse_sample_idxs[i]] = block.sparse_values[i]
    return out


def test_sparse_partition_and_row_mapping(tmp_path):
    ds = _load(tmp_path)
    try:
        block = BlockReader(ds).read([0, 1, 2, 3, 4, 5])
        assert list(block.dense_rows) == DENSE_VARIANTS
        assert list(block.sparse_rows) == SPARSE_VARIANTS
        assert block.stats.variants == 6
        assert block.stats.dense_variants == 3
        assert block.stats.sparse_variants == 3
        assert block.stats.mode == "hardcall"
        assert block.stats.block_index == 1 and block.stats.block_total == 1
        # Dense rows follow dense_rows order.
        np.testing.assert_array_equal(
            block.dense, GENOTYPES[DENSE_VARIANTS].astype(np.int8)
        )
    finally:
        ds.close()


def test_sparse_and_dense_reads_agree(tmp_path):
    ds = _load(tmp_path)
    try:
        for index in range(ds.variant_ct):
            block = BlockReader(ds).read([index])
            dense = BlockReader(ds, force_dense=True).read([index]).dense
            np.testing.assert_array_equal(
                _reconstruct(block, ds.active_sample_ct), dense
            )
    finally:
        ds.close()


def test_sparse_preserves_missing_and_nonzero_calls(tmp_path):
    ds = _load(tmp_path)
    try:
        block = BlockReader(ds).read([4])
        assert list(block.sparse_rows) == [0]
        values = block.sparse_values[0]
        assert -9 in values  # the listed -9 is not silently turned into 0
        assert 2 in values
    finally:
        ds.close()


def test_force_dense_returns_only_dense_rows(tmp_path):
    ds = _load(tmp_path)
    try:
        block = BlockReader(ds, force_dense=True).read([0, 1, 2, 3, 4, 5])
        assert block.sparse_rows.size == 0
        assert list(block.dense_rows) == list(range(6))
        assert block.stats.sparse_variants == 0
        assert block.stats.dense_variants == 6
        np.testing.assert_array_equal(block.dense, GENOTYPES.astype(np.int8))
    finally:
        ds.close()


def test_bfile_is_always_dense(tmp_path):
    prefix = make_bfile(tmp_path, "bed", GENOTYPES, SAMPLE_IDS, VARIANTS)
    ds = load_dataset(prefix, "bfile", np.array(SAMPLE_IDS))
    try:
        block = BlockReader(ds).read([0, 2, 4])
        assert block.sparse_rows.size == 0
        assert list(block.dense_rows) == [0, 1, 2]
        np.testing.assert_array_equal(block.dense, GENOTYPES[[0, 2, 4]])
    finally:
        ds.close()


def test_empty_index_list_is_allowed(tmp_path):
    ds = _load(tmp_path)
    try:
        block = BlockReader(ds).read([])
        assert block.stats.variants == 0
        assert block.dense.shape == (0, ds.active_sample_ct)
        assert block.dense_rows.size == 0 and block.sparse_rows.size == 0
        assert block.sparse_sample_idxs == [] and block.sparse_values == []
    finally:
        ds.close()


def test_buffer_matches_active_sample_subset(tmp_path):
    ds = _load(tmp_path)
    try:
        assert ds.sample_ct == 20 and ds.active_sample_ct == 20
        subset = np.array([1, 3, 5, 7], dtype=np.uint32)
        ds_sub = _load(tmp_path, name="blocks_sub", sample_subset=subset)
        try:
            assert ds_sub.active_sample_ct == 4
            block = BlockReader(ds_sub).read([1, 3])
            assert block.dense.shape == (2, 4)
            full = BlockReader(ds, force_dense=True).read([1, 3]).dense
            np.testing.assert_array_equal(block.dense, full[:, subset])
        finally:
            ds_sub.close()
    finally:
        ds.close()


def test_dosage_reads_match_hardcalls_on_hardcall_data(tmp_path):
    ds = _load(tmp_path)
    try:
        reader_hard = BlockReader(ds, dosage=False)
        reader_dos = BlockReader(ds, dosage=True)
        hard = reader_hard.read([1, 3, 5])
        dos = reader_dos.read([1, 3, 5])
        np.testing.assert_allclose(dos.dense, hard.dense)
        assert dos.stats.mode == "dosage"
        assert reader_dos.arr_dtype == np.float64
        assert hard.dense.dtype == np.int8 and dos.dense.dtype == np.float64
    finally:
        ds.close()


def test_sparse_dosage_path_matches_dense(tmp_path):
    dosages = np.array(
        [
            [0.0] * 19 + [1.5],
            [0.5] + [0.0] * 19,
            [1.0, 1.0] + [0.0] * 16 + [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    variants = [
        ("1", f"1:{100 * (i + 1)}:A:G", 100 * (i + 1), "A", "G")
        for i in range(dosages.shape[0])
    ]
    prefix = make_dosage_pfile(tmp_path, "dos", dosages, SAMPLE_IDS, variants)
    ds = load_dataset(prefix, "pfile", np.array(SAMPLE_IDS), dosage=True)
    try:
        assert any(ds.reader.has_sparse_dosages(i) for i in range(ds.variant_ct))
        block = BlockReader(ds, dosage=True).read([0, 1, 2])
        assert block.stats.sparse_variants >= 1
        dense = BlockReader(ds, dosage=True, force_dense=True).read([0, 1, 2]).dense
        np.testing.assert_allclose(
            _reconstruct(block, ds.active_sample_ct), dense, atol=1e-6
        )
    finally:
        ds.close()


def test_stats_defaults_and_ordering(tmp_path):
    ds = _load(tmp_path)
    try:
        stats = BlockStats()
        assert stats.block_index == 1 and stats.block_total == 1
        assert stats.variants == 0 and stats.mode == "hardcall"
        block = BlockReader(ds).read([2, 0])
        # Requested order 2, 0 is preserved in the row mapping.
        assert list(block.sparse_rows) == [0, 1]
        assert list(block.sparse_sample_idxs[0]) == [0]
        assert list(block.sparse_sample_idxs[1]) == []
    finally:
        ds.close()
