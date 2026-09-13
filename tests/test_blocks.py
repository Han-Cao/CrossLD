"""Unit tests for the block-wise reader in :mod:`crossld.blocks`."""

from __future__ import annotations

import numpy as np
import pytest

from crossld.blocks import BlockReader
from crossld.plinkio import load_dataset, select_samples


@pytest.fixture(scope="module")
def loaded(sample1_prefix, sample2_prefix):
    ds1 = load_dataset(sample1_prefix, "dataset1", "pfile")
    ds2 = load_dataset(sample2_prefix, "dataset2", "pfile")
    subset1, subset2, _ = select_samples(ds1, ds2)
    ds1.set_sample_subset(subset1)
    ds2.set_sample_subset(subset2)
    yield ds1, ds2
    ds1.close()
    ds2.close()


def _sparse_eligible(dataset) -> list[int]:
    return [
        i
        for i in range(dataset.variant_ct)
        if dataset.reader.has_sparse(i)
    ]


def test_sparse_and_dense_agree_per_variant(loaded):
    ds1, _ = loaded
    eligible = _sparse_eligible(ds1)
    assert eligible, "fixture should contain at least one sparse-eligible variant"

    sparse_reader = BlockReader(ds1, sparse="auto")
    dense_reader = BlockReader(ds1, sparse="never")
    for index in eligible[:20]:
        sparse_matrix, sparse_stats = sparse_reader.read([index])
        dense_matrix, _ = dense_reader.read([index])
        assert sparse_stats.sparse_variants == 1
        assert sparse_stats.sparse_fallbacks == 0
        np.testing.assert_array_equal(sparse_matrix, dense_matrix)


def test_dense_fallback_on_non_sparse_variant(loaded):
    ds1, _ = loaded
    non_sparse = [i for i in range(ds1.variant_ct) if not ds1.reader.has_sparse(i)]
    assert non_sparse
    reader = BlockReader(ds1, sparse="auto")
    matrix, stats = reader.read([non_sparse[0]])
    assert stats.sparse_fallbacks == 1
    assert stats.sparse_variants == 0
    dense_matrix, _ = BlockReader(ds1, sparse="never").read([non_sparse[0]])
    np.testing.assert_array_equal(matrix, dense_matrix)


def test_sparse_require_raises_for_non_sparse_variant(loaded):
    ds1, _ = loaded
    non_sparse = [i for i in range(ds1.variant_ct) if not ds1.reader.has_sparse(i)]
    with pytest.raises(RuntimeError):
        BlockReader(ds1, sparse="require").read([non_sparse[0]])


def test_explicit_flip_applies_to_dense_reads(loaded):
    ds1, _ = loaded
    reader = BlockReader(ds1, sparse="never")
    plain, _ = reader.read([0])
    flipped, _ = reader.read([0], flip=np.array([True]))
    expected = np.where(plain >= 0, 2.0 - plain, plain)
    np.testing.assert_allclose(flipped, expected)


def test_empty_index_list_is_allowed(loaded):
    ds1, _ = loaded
    matrix, stats = BlockReader(ds1).read([])
    assert matrix.shape == (0, ds1.active_sample_ct)
    assert stats.variants == 0


def test_buffer_matches_active_sample_subset(loaded):
    ds1, _ = loaded
    matrix, _ = BlockReader(ds1).read([0])
    assert matrix.shape == (1, ds1.active_sample_ct) == (1, 100)


def test_dosage_reads_match_hardcalls_on_hardcall_data(loaded):
    ds1, _ = loaded
    hardcalls, _ = BlockReader(ds1, sparse="never", dosage=False).read([0, 1, 2])
    dosages, stats = BlockReader(ds1, sparse="never", dosage=True).read([0, 1, 2])
    np.testing.assert_allclose(dosages, hardcalls)
    assert stats.mode == "dosage"


def test_sparse_dosage_path_matches_dense(tmp_path):
    from crossld.plinkio import load_dataset
    from tests.conftest import make_dosage_pfile

    samples = [f"S{i}" for i in range(8)]
    variants = [("1", "1:100:A:G", 100, "A", "G"), ("1", "1:200:C:T", 200, "C", "T")]
    dosages = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    prefix = make_dosage_pfile(tmp_path, "dos", dosages, samples, variants)
    dataset = load_dataset(prefix, "dos", "pfile", dosage=True)
    try:
        sparse_reader = BlockReader(dataset, sparse="auto", dosage=True)
        sparse_matrix, stats = sparse_reader.read([0, 1])
        dense_matrix, _ = BlockReader(dataset, sparse="never", dosage=True).read([0, 1])
        np.testing.assert_allclose(sparse_matrix, dense_matrix)
        assert stats.sparse_variants + stats.sparse_fallbacks >= 1
    finally:
        dataset.close()


def test_sparse_flip_applies_to_sparse_reads(loaded):
    ds1, _ = loaded
    eligible = _sparse_eligible(ds1)
    index = eligible[0]
    plain, stats = BlockReader(ds1, sparse="auto").read([index])
    assert stats.sparse_variants == 1
    flipped, _ = BlockReader(ds1, sparse="auto").read([index], flip=np.array([True]))
    expected = np.where(plain >= 0, 2.0 - plain, plain)
    np.testing.assert_allclose(flipped, expected)


def test_runtime_error_falls_back_to_dense(loaded):
    ds1, _ = loaded
    eligible = _sparse_eligible(ds1)
    reader = BlockReader(ds1, sparse="auto")

    class Boom:
        """Proxy that advertises sparse support but fails on read."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def has_sparse(self, variant_idx, allele_idx=1):
            return True

        def read_sparse8(self, variant_idx, allele_idx=1):
            raise RuntimeError("no sparse representation")

    reader.reader = Boom(ds1.reader)
    matrix, stats = reader.read([eligible[0]])
    assert stats.sparse_fallbacks == 1
    assert stats.sparse_variants == 0
    dense, _ = BlockReader(ds1, sparse="never").read([eligible[0]])
    np.testing.assert_array_equal(matrix, dense)
