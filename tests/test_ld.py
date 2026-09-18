"""Unit tests for the LD math in :mod:`crossld.ld`.

Dense math is checked against the independent :func:`numpy.corrcoef` oracle in
``tests/conftest.py``; the sparse path is checked against the dense path on the
same data (unlisted samples are homozygous REF, listed negative values are
missing) and against the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from crossld.blocks import BlockMatrix, BlockStats
from crossld.ld import (
    alt_freq,
    alt_freq_sparse,
    major_allele_flipped,
    major_allele_flipped_sparse,
    pearson_pairwise,
    pearson_pairwise_block,
    pearson_pairwise_complete,
    pearson_pairwise_masked,
    pearson_pairwise_sparse,
    pearson_pairwise_sparse_complete,
    pearson_pairwise_sparse_masked,
)
from tests.conftest import pearson_reference, pearson_reference_stable

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _sparsify(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Represent a dense row the way ``pgenlib`` does: list every non-zero call.

    Missing calls are negative (non-zero), so they are listed too.
    """

    dense = np.asarray(dense, dtype=float)
    idx = np.flatnonzero(dense != 0).astype(np.uint32)
    return idx, dense[idx]


def _expand(idx: np.ndarray, values: np.ndarray, sample_ct: int) -> np.ndarray:
    out = np.zeros(sample_ct, dtype=float)
    out[idx] = values
    return out


def _blockmatrix(
    sample_ct: int,
    dense_rows: list[int],
    dense: list[np.ndarray],
    sparse_rows: list[int],
    sparse: list[tuple[np.ndarray, np.ndarray]],
) -> BlockMatrix:
    """Build a :class:`BlockMatrix` with the requested dense/sparse split."""

    variants = len(dense_rows) + len(sparse_rows)
    assert len(dense_rows) == len(dense)
    assert len(sparse_rows) == len(sparse)
    assert sorted([*dense_rows, *sparse_rows]) == list(range(variants))
    for idx, values in sparse:
        idx = np.asarray(idx)
        values = np.asarray(values)
        assert idx.shape == values.shape
        assert np.unique(idx).size == idx.size
        assert idx.size == 0 or int(idx.max()) < sample_ct
    dense_arr = (
        np.asarray(dense, dtype=float).reshape(-1, sample_ct)
        if dense
        else np.empty((0, sample_ct), dtype=float)
    )
    return BlockMatrix(
        dense=dense_arr,
        dense_rows=np.asarray(dense_rows, dtype=np.uint32),
        sparse_rows=np.asarray(sparse_rows, dtype=np.uint32),
        sparse_sample_idxs=[np.asarray(i, dtype=np.uint32) for i, _ in sparse],
        sparse_values=[np.asarray(v) for _, v in sparse],
        stats=BlockStats(variants=variants),
    )


# --------------------------------------------------------------------------- #
# dense pearson_pairwise
# --------------------------------------------------------------------------- #

TARGET = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "expected, row",
    [
        (1.0, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        (-1.0, [2.0, 1.0, 0.0, 2.0, 1.0, 0.0]),
    ],
)
def test_perfect_correlation(expected, row):
    r, n = pearson_pairwise(TARGET, np.array([row]))
    assert float(r[0]) == pytest.approx(expected)
    assert int(n[0]) == 6


def test_uncorrelated_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=5000).astype(float)
    b = rng.integers(0, 3, size=5000).astype(float)
    r, _ = pearson_pairwise(a, b.reshape(1, -1))
    assert abs(float(r[0])) < 0.05


def test_missing_values_are_pairwise_masked():
    a = np.array([0.0, 1.0, 2.0, -9.0, 1.0, 2.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 2.0, -9.0, 2.0, 0.0])
    r, n = pearson_pairwise(a, b.reshape(1, -1))
    assert float(r[0]) == pytest.approx(1.0)
    assert int(n[0]) == 5


def test_fewer_than_three_valid_samples_is_nan():
    a = np.array([0.0, 1.0, -9.0, -9.0])
    b = np.array([0.0, 1.0, 2.0, 2.0])
    r, n = pearson_pairwise(a, b.reshape(1, -1))
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 2


@pytest.mark.parametrize(
    "target, row",
    [
        (np.zeros(6), np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
        (TARGET, np.full(6, 2.0)),
    ],
    ids=["constant-target", "constant-row"],
)
def test_zero_variance_is_nan(target, row):
    r, _ = pearson_pairwise(target, row.reshape(1, -1))
    assert np.isnan(float(r[0]))


def test_pairwise_is_vectorised_over_blocks():
    block = np.array(
        [
            TARGET,
            2.0 - TARGET,
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            np.zeros(6),
        ]
    )
    r, n = pearson_pairwise(TARGET, block)
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(-1.0)
    assert np.isfinite(float(r[2]))
    assert np.isnan(float(r[3]))
    assert list(int(x) for x in n) == [6, 6, 6, 6]


def test_dense_matches_reference_oracle_with_missing():
    rng = np.random.default_rng(7)
    target = rng.integers(0, 3, size=200).astype(float)
    target[rng.random(200) < 0.1] = -9.0
    block = rng.integers(0, 3, size=(12, 200)).astype(float)
    block[rng.random(block.shape) < 0.1] = -9.0

    r, n = pearson_pairwise(target, block)
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        assert int(n[i]) == int(((target >= 0) & (block[i] >= 0)).sum())
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# sparse pearson_pairwise_sparse
# --------------------------------------------------------------------------- #

def test_sparse_row_matches_dense_row():
    """A sparse row equals the dense row with the unlisted samples set to 0."""

    dense = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    idx, values = _sparsify(dense)
    r_dense, n_dense = pearson_pairwise(TARGET, dense.reshape(1, -1))
    r_sparse, n_sparse = pearson_pairwise_sparse(TARGET, [idx], [values])
    assert float(r_sparse[0]) == pytest.approx(float(r_dense[0]))
    assert int(n_sparse[0]) == int(n_dense[0]) == 6


def test_sparse_listed_missing_reduces_n():
    target = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    idx = np.array([2, 3, 4], dtype=np.uint32)
    values = np.array([1.0, -9.0, 2.0])
    dense = _expand(idx, values, sample_ct=6)
    r_sparse, n_sparse = pearson_pairwise_sparse(target, [idx], [values])
    r_dense, n_dense = pearson_pairwise(target, dense.reshape(1, -1))
    assert int(n_sparse[0]) == int(n_dense[0]) == 5
    assert float(r_sparse[0]) == pytest.approx(float(r_dense[0]))


def test_sparse_empty_row_is_nan():
    """No listed samples -> constant zero row -> zero variance."""

    r, n = pearson_pairwise_sparse(
        TARGET, [np.empty(0, dtype=np.uint32)], [np.empty(0, dtype=float)]
    )
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 6


def test_sparse_all_missing_row_is_nan():
    idx = np.arange(6, dtype=np.uint32)
    values = np.full(6, -9.0)
    r, n = pearson_pairwise_sparse(TARGET, [idx], [values])
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 0


def test_sparse_constant_row_is_nan():
    idx = np.arange(6, dtype=np.uint32)
    values = np.ones(6)
    r, _ = pearson_pairwise_sparse(TARGET, [idx], [values])
    assert np.isnan(float(r[0]))


def test_sparse_zero_variance_target_is_nan():
    target = np.zeros(6)
    dense = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    idx, values = _sparsify(dense)
    r, _ = pearson_pairwise_sparse(target, [idx], [values])
    assert np.isnan(float(r[0]))


def test_sparse_multiple_rows_are_vectorised():
    rows = [
        TARGET.copy(),
        np.array([2.0, 1.0, 0.0, 2.0, 1.0, 0.0]),
        np.zeros(6),
    ]
    idxs, values = zip(*[_sparsify(row) for row in rows])
    r, n = pearson_pairwise_sparse(TARGET, list(idxs), list(values))
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(-1.0)
    assert np.isnan(float(r[2]))
    assert list(int(x) for x in n) == [6, 6, 6]


def test_sparse_matches_dense_and_oracle_with_missing():
    rng = np.random.default_rng(11)
    target = rng.integers(0, 3, size=150).astype(float)
    target[rng.random(150) < 0.08] = -9.0
    block = rng.integers(0, 3, size=(10, 150)).astype(float)
    block[rng.random(block.shape) < 0.08] = -9.0

    idxs, values = zip(*[_sparsify(row) for row in block])
    r, n = pearson_pairwise_sparse(target, list(idxs), list(values))
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)
        assert int(n[i]) == int(((target >= 0) & (block[i] >= 0)).sum())


# --------------------------------------------------------------------------- #
# allele-frequency / orientation helpers
# --------------------------------------------------------------------------- #

def test_alt_freq_ignores_missing():
    assert alt_freq(np.array([0.0, 1.0, 2.0, -9.0])) == pytest.approx(3.0 / (2 * 3))
    assert np.isnan(alt_freq(np.array([-9.0, -9.0])))
    assert np.isnan(alt_freq(np.array([])))


def test_alt_freq_sparse_treats_unlisted_as_reference():
    # Unlisted samples are 0; the one listed missing call is dropped.
    assert alt_freq_sparse(np.array([1.0, -9.0]), sample_ct=2) == pytest.approx(1.0 / (2 * 1))
    assert alt_freq_sparse(np.array([]), sample_ct=4) == 0.0
    assert np.isnan(alt_freq_sparse(np.array([-9.0, -9.0]), sample_ct=2))


def test_alt_freq_dense_and_sparse_agree():
    dense = np.array([0.0, 0.0, 1.0, -9.0, 2.0, 2.0])
    idx, values = _sparsify(dense)
    assert alt_freq_sparse(values, sample_ct=6) == pytest.approx(alt_freq(dense))


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([2.0, 2.0, 2.0, 2.0, 0.0]), True),  # ALT frequency 0.8
        (np.array([0.0, 0.0, 0.0, 0.0, 2.0]), False),  # ALT frequency 0.2
        (np.array([1.0, 1.0]), False),  # exact tie
        (np.array([]), False),
        (np.array([-9.0, -9.0]), False),
    ],
)
def test_major_allele_flipped_is_true_above_half(values, expected):
    assert major_allele_flipped(values) is expected


def test_major_allele_flipped_sparse_matches_dense():
    for dense in (
        np.array([2.0, 2.0, 2.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 2.0]),
        np.array([1.0, -9.0, 0.0]),
    ):
        idx, values = _sparsify(dense)
        assert major_allele_flipped_sparse(values, sample_ct=dense.size) is (
            major_allele_flipped(dense)
        )


# --------------------------------------------------------------------------- #
# block entry point
# --------------------------------------------------------------------------- #

def test_block_returns_results_in_requested_row_order():
    """Dense and sparse rows are interleaved back into the requested order."""

    target = TARGET
    block = _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 2],
        dense=[
            np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),  # r = 1
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),  # finite
        ],
        sparse_rows=[1, 3],
        sparse=[
            _sparsify(np.array([2.0, 1.0, 0.0, 2.0, 1.0, 0.0])),  # r = -1
            (np.empty(0, dtype=np.uint32), np.empty(0, dtype=float)),  # NaN
        ],
    )
    r, n = pearson_pairwise_block(target, block)
    assert len(r) == len(n) == 4
    # Output positions follow the requested order: dense row 0 -> r[0],
    # sparse row 1 -> r[1], dense row 2 -> r[2], sparse row 3 -> r[3].
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(-1.0)
    assert float(r[2]) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(float(r[3]))


def test_block_empty_read():
    empty = BlockMatrix(
        dense=np.empty((0, 6), dtype=float),
        dense_rows=np.empty(0, dtype=np.uint32),
        sparse_rows=np.empty(0, dtype=np.uint32),
        sparse_sample_idxs=[],
        sparse_values=[],
        stats=BlockStats(variants=0),
    )
    r, n = pearson_pairwise_block(TARGET, empty)
    assert r.shape == n.shape == (0,)


# --------------------------------------------------------------------------- #
# complete (no-missing) fast paths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.float64])
def test_dense_complete_matches_masked(dtype):
    rng = np.random.default_rng(21)
    target = rng.integers(0, 3, size=250).astype(dtype)
    block = rng.integers(0, 3, size=(9, 250)).astype(dtype)
    r_masked, n_masked = pearson_pairwise(target, block)
    r_fast, n_fast = pearson_pairwise_complete(target, block)
    assert np.array_equal(n_masked, n_fast)
    assert np.allclose(r_masked, r_fast, atol=1e-12, equal_nan=True)


def test_dense_complete_matches_oracle_and_nan_rules():
    rng = np.random.default_rng(22)
    target = rng.integers(0, 3, size=200).astype(float)
    block = rng.integers(0, 3, size=(6, 200)).astype(float)
    block[4] = 1.0  # constant row -> zero variance -> NaN
    r, n = pearson_pairwise_complete(target, block)
    assert list(int(x) for x in n) == [200] * 6
    for i in range(6):
        expected = pearson_reference(target, block[i])
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)


def test_dense_complete_zero_variance_target_is_nan():
    block = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    r, n = pearson_pairwise_complete(np.zeros(3), block)
    assert np.isnan(r).all()
    assert list(int(x) for x in n) == [3, 3]


def test_dense_complete_empty_block():
    r, n = pearson_pairwise_complete(TARGET, np.empty((0, TARGET.size)))
    assert r.shape == n.shape == (0,)


def test_sparse_complete_matches_masked_and_oracle():
    rng = np.random.default_rng(23)
    target = rng.integers(0, 3, size=180).astype(float)
    block = rng.integers(0, 3, size=(11, 180)).astype(float)
    idxs, values = zip(*[_sparsify(row) for row in block])
    r_masked, n_masked = pearson_pairwise_sparse(target, list(idxs), list(values))
    r_fast, n_fast = pearson_pairwise_sparse_complete(
        target, list(idxs), list(values)
    )
    assert np.array_equal(n_masked, n_fast)
    assert np.allclose(r_masked, r_fast, atol=1e-12, equal_nan=True)
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        if np.isnan(expected):
            assert np.isnan(float(r_fast[i]))
        else:
            assert float(r_fast[i]) == pytest.approx(expected, abs=1e-9)


def test_sparse_complete_empty_and_empty_rows():
    idxs = [np.empty(0, dtype=np.uint32), np.array([1, 2], dtype=np.uint32)]
    values = [np.empty(0), np.array([1.0, 2.0])]
    r, n = pearson_pairwise_sparse_complete(TARGET, idxs, values)
    assert np.isnan(float(r[0]))  # empty row -> zero variance
    assert int(n[0]) == 6
    assert np.isfinite(float(r[1]))
    assert int(n[1]) == 6


# --------------------------------------------------------------------------- #
# block routing + force_masked
# --------------------------------------------------------------------------- #

def _mixed_block() -> BlockMatrix:
    """2 clean dense rows, 1 dirty dense row, 1 clean sparse, 1 dirty sparse."""

    clean = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    dirty = clean.copy()
    dirty[3] = -9.0
    return _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 1, 4],
        dense=[clean, 2.0 - clean, dirty],
        sparse_rows=[2, 3],
        sparse=[
            _sparsify(np.array([2.0, 1.0, 0.0, 2.0, 0.0, 0.0])),
            (np.array([1, 2], dtype=np.uint32), np.array([1.0, -9.0])),
        ],
    )


def test_block_routing_splits_clean_and_missing_rows():
    fast_block = _mixed_block()
    masked_block = _mixed_block()
    r_fast, n_fast = pearson_pairwise_block(TARGET, fast_block)
    r_masked, n_masked = pearson_pairwise_block(TARGET, masked_block, force_masked=True)

    assert np.allclose(r_fast, r_masked, atol=1e-12, equal_nan=True)
    assert np.array_equal(n_fast, n_masked)
    # rows: 0 clean dense, 1 clean dense, 2 clean sparse, 3 dirty sparse, 4 dirty dense
    assert fast_block.stats.fast_rows == 3
    assert fast_block.stats.masked_rows == 2
    assert list(int(x) for x in n_fast) == [6, 6, 6, 5, 5]
    # force_masked takes the masked path for every row
    assert masked_block.stats.fast_rows == 0
    assert masked_block.stats.masked_rows == 5


def test_block_all_clean_uses_the_fast_path():
    block = _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 1],
        dense=[TARGET, 2.0 - TARGET],
        sparse_rows=[2],
        sparse=[_sparsify(np.array([2.0, 1.0, 0.0, 2.0, 0.0, 0.0]))],
    )
    r, _ = pearson_pairwise_block(TARGET, block)
    assert block.stats.fast_rows == 3
    assert block.stats.masked_rows == 0
    assert float(r[0]) == pytest.approx(1.0)
    assert float(r[1]) == pytest.approx(-1.0)
    assert np.isfinite(float(r[2]))


def test_block_no_missing_target_keeps_fast_path():
    """A dirty row in the block must not disable the fast path for clean rows."""

    block = _mixed_block()
    pearson_pairwise_block(TARGET, block)
    assert block.stats.fast_rows > 0


def test_block_force_masked_disables_fast_path():
    block = _blockmatrix(sample_ct=6, dense_rows=[0], dense=[TARGET], sparse_rows=[], sparse=[])
    r, n = pearson_pairwise_block(TARGET, block, force_masked=True)
    assert block.stats.fast_rows == 0
    assert block.stats.masked_rows == 1
    assert float(r[0]) == pytest.approx(1.0)
    assert int(n[0]) == 6


# --------------------------------------------------------------------------- #
# sample_mask on the complete fast paths
# --------------------------------------------------------------------------- #

TARGET_KEEP = np.array([0.0, 1.0, 2.0, 0.0, 1.0, -9.0])
KEEP = TARGET_KEEP >= 0  # first five samples


def test_dense_complete_sample_mask_matches_masked():
    rng = np.random.default_rng(41)
    size = 150
    target = rng.integers(0, 3, size=size).astype(float)
    keep = rng.random(size) > 0.3
    target[~keep] = -9.0
    block = rng.integers(0, 3, size=(7, size)).astype(float)
    block[:, ~keep] = -9.0  # complete on keep, missing elsewhere

    r_fast, n_fast = pearson_pairwise_complete(target, block, sample_mask=keep)
    r_masked, n_masked = pearson_pairwise(target, block)

    assert np.array_equal(n_fast, n_masked)
    assert list(int(x) for x in n_fast) == [int(keep.sum())] * block.shape[0]
    assert np.allclose(r_fast, r_masked, atol=1e-12, equal_nan=True)


def test_sparse_complete_sample_mask_matches_masked():
    rng = np.random.default_rng(42)
    size = 140
    target = rng.integers(0, 3, size=size).astype(float)
    keep = rng.random(size) > 0.3
    target[~keep] = -9.0
    block = rng.integers(0, 3, size=(9, size)).astype(float)
    block[:, ~keep] = -9.0
    idxs, values = zip(*[_sparsify(row) for row in block])

    r_fast, n_fast = pearson_pairwise_sparse_complete(
        target, list(idxs), list(values), sample_mask=keep
    )
    r_masked, n_masked = pearson_pairwise_sparse(target, list(idxs), list(values))

    assert np.array_equal(n_fast, n_masked)
    assert list(int(x) for x in n_fast) == [int(keep.sum())] * block.shape[0]
    assert np.allclose(r_fast, r_masked, atol=1e-12, equal_nan=True)


def test_sparse_complete_sample_mask_filters_target_missing_entries():
    target = np.array([0.0, 1.0, -9.0, 2.0, 1.0])
    keep = target >= 0
    idxs = [np.array([2], dtype=np.uint32), np.array([0], dtype=np.uint32)]
    values = [np.array([-9.0]), np.array([1.0])]

    r, n = pearson_pairwise_sparse_complete(
        target, idxs, values, sample_mask=keep
    )

    assert r.shape == n.shape == (2,)
    # Row 0 loses its only entry: constant -> NaN, but n is still the mask size.
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 4
    assert np.isfinite(float(r[1]))
    assert int(n[1]) == 4


# --------------------------------------------------------------------------- #
# memory-lean masked kernels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.float64])
def test_dense_masked_matches_pairwise(dtype):
    rng = np.random.default_rng(51)
    target = rng.integers(0, 3, size=200).astype(float)
    target[rng.random(200) < 0.3] = -9.0
    block = rng.integers(0, 3, size=(11, 200)).astype(dtype)
    block[rng.random((11, 200)) < 0.2] = dtype(-9)

    r_current, n_current = pearson_pairwise(target, block)
    r_lean, n_lean = pearson_pairwise_masked(target, block)

    assert np.array_equal(n_current, n_lean)
    assert np.array_equal(np.isnan(r_current), np.isnan(r_lean))
    assert np.allclose(r_current, r_lean, atol=1e-12, equal_nan=True)


def test_dense_masked_treats_nan_as_missing():
    target = np.array([0.0, 1.0, np.nan, 2.0, 1.0])
    block = np.array(
        [[0.0, 1.0, 2.0, np.nan, 1.0], [0.0, 1.0, 2.0, 0.0, 1.0]]
    )
    r_current, n_current = pearson_pairwise(target, block)
    r_lean, n_lean = pearson_pairwise_masked(target, block)
    assert np.array_equal(n_current, n_lean)
    assert np.array_equal(np.isnan(r_current), np.isnan(r_lean))


def test_sparse_masked_matches_sparse():
    rng = np.random.default_rng(52)
    target = rng.integers(0, 3, size=180).astype(float)
    target[rng.random(180) < 0.3] = -9.0
    block = rng.integers(0, 3, size=(13, 180)).astype(float)
    block[rng.random((13, 180)) < 0.25] = -9.0
    idxs, values = zip(*[_sparsify(row) for row in block])

    r_current, n_current = pearson_pairwise_sparse(target, list(idxs), list(values))
    r_lean, n_lean = pearson_pairwise_sparse_masked(
        target, list(idxs), list(values)
    )

    assert np.array_equal(n_current, n_lean)
    assert np.array_equal(np.isnan(r_current), np.isnan(r_lean))
    assert np.allclose(r_current, r_lean, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# incomplete-target routing
# --------------------------------------------------------------------------- #

def _incomplete_block() -> BlockMatrix:
    """Target with a missing call; two dense and two sparse candidate rows."""

    good = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -9.0])  # missing outside keep
    bad = np.array([0.0, 1.0, -9.0, 0.0, 1.0, 2.0])  # missing inside keep
    return _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 1],
        dense=[good, bad],
        sparse_rows=[2, 3],
        sparse=[
            (np.array([5], dtype=np.uint32), np.array([2.0])),
            (np.array([2], dtype=np.uint32), np.array([-9.0])),
        ],
    )


def test_block_incomplete_target_routing():
    block = _incomplete_block()
    forced = _incomplete_block()

    r, n = pearson_pairwise_block(TARGET_KEEP, block)
    r_forced, n_forced = pearson_pairwise_block(
        TARGET_KEEP, forced, force_masked=True
    )

    # Rows 0 and 2 are complete on keep, rows 1 and 3 are not.
    assert block.stats.fast_rows == 2
    assert block.stats.masked_rows == 2
    assert forced.stats.fast_rows == 0
    assert forced.stats.masked_rows == 4
    assert list(int(x) for x in n) == [5, 4, 5, 4]
    assert np.array_equal(n, n_forced)
    assert np.allclose(r, r_forced, atol=1e-12, equal_nan=True)


def test_block_incomplete_target_matches_oracle():
    block = _incomplete_block()
    r, n = pearson_pairwise_block(TARGET_KEEP, block)
    rows = [
        block.dense[0],
        block.dense[1],
        _expand(np.array([5], dtype=np.uint32), np.array([2.0]), 6),
        _expand(np.array([2], dtype=np.uint32), np.array([-9.0]), 6),
    ]
    assert int(block.stats.fast_rows + block.stats.masked_rows) == 4
    for i, row in enumerate(rows):
        expected = pearson_reference(TARGET_KEEP, row)
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)
        assert int(n[i]) == int(((TARGET_KEEP >= 0) & (row >= 0)).sum())


def test_block_few_valid_target_samples_is_nan():
    target = np.array([0.0, 1.0, -9.0, -9.0, -9.0, -9.0])  # n_keep == 2
    block = _blockmatrix(
        sample_ct=6,
        dense_rows=[0],
        dense=[np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])],
        sparse_rows=[],
        sparse=[],
    )
    r, n = pearson_pairwise_block(target, block)
    assert block.stats.fast_rows == 0
    assert block.stats.masked_rows == 1
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 2


def test_block_dense_nan_inside_keep_routes_masked():
    target = np.array([0.0, 1.0, 2.0, 0.0, -9.0])
    block = _blockmatrix(
        sample_ct=5,
        dense_rows=[0],
        dense=[np.array([0.0, 1.0, np.nan, 1.0, 2.0])],
        sparse_rows=[],
        sparse=[],
    )
    forced = _blockmatrix(
        sample_ct=5,
        dense_rows=[0],
        dense=[np.array([0.0, 1.0, np.nan, 1.0, 2.0])],
        sparse_rows=[],
        sparse=[],
    )
    r, n = pearson_pairwise_block(target, block)
    r_forced, n_forced = pearson_pairwise_block(target, forced, force_masked=True)
    assert block.stats.fast_rows == 0
    assert block.stats.masked_rows == 1
    assert np.array_equal(n, n_forced)
    assert np.allclose(r, r_forced, atol=1e-12, equal_nan=True)


def test_block_all_empty_sparse_rows_returns_nan():
    """Two empty sparse rows used to return zero-length arrays and crash."""

    empty = np.empty(0, dtype=np.uint32)
    block = _blockmatrix(
        sample_ct=6,
        dense_rows=[],
        dense=[],
        sparse_rows=[0, 1],
        sparse=[(empty, np.empty(0)), (empty, np.empty(0))],
    )
    r, n = pearson_pairwise_block(TARGET, block)
    assert r.shape == n.shape == (2,)
    assert np.isnan(r).all()
    assert list(int(x) for x in n) == [6, 6]
    assert block.stats.fast_rows == 2
    assert block.stats.masked_rows == 0


# --------------------------------------------------------------------------- #
# raw-moment cancellation on nearly constant fractional dosages
# --------------------------------------------------------------------------- #

def _near_constant(seed: int, size: int, spread: float = 1e-8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.0 + rng.normal(0.0, spread, size)


def test_near_constant_dense_uses_centered_formulation():
    # The raw-moment variance of this pair rounds to zero, but the data varies.
    x = np.array([1.0 - 1e-10, 1.0, 1.0 + 1e-10])
    r_masked, n_masked = pearson_pairwise_masked(x, x.reshape(1, -1))
    r_complete, n_complete = pearson_pairwise_complete(x, x.reshape(1, -1))

    assert float(r_masked[0]) == pytest.approx(1.0)
    assert float(r_complete[0]) == pytest.approx(1.0)
    assert int(n_masked[0]) == int(n_complete[0]) == 3


def test_near_constant_sparse_uses_centered_formulation():
    x = np.array([1.0 - 1e-10, 1.0, 1.0 + 1e-10])
    idx = np.array([0, 1, 2], dtype=np.uint32)

    r_legacy, _ = pearson_pairwise_sparse(x, [idx], [x])
    r_masked, _ = pearson_pairwise_sparse_masked(x, [idx], [x])
    r_complete, _ = pearson_pairwise_sparse_complete(x, [idx], [x])

    assert float(r_legacy[0]) == pytest.approx(1.0)
    assert float(r_masked[0]) == pytest.approx(1.0)
    assert float(r_complete[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kernel",
    [pearson_pairwise_masked, pearson_pairwise_complete],
    ids=["masked", "complete"],
)
def test_near_constant_dense_matches_centered_oracle(kernel):
    target = _near_constant(81, 200)
    block = np.vstack([_near_constant(82, 200), _near_constant(83, 200)])

    r_raw, n_raw = kernel(target, block)
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        assert not np.isnan(expected)
        assert float(r_raw[i]) == pytest.approx(expected, abs=1e-10)
    assert list(int(x) for x in n_raw) == [target.size, target.size]


@pytest.mark.parametrize(
    "kernel",
    [pearson_pairwise_sparse, pearson_pairwise_sparse_masked,
     pearson_pairwise_sparse_complete],
    ids=["legacy", "masked", "complete"],
)
def test_near_constant_sparse_matches_centered_oracle(kernel):
    target = _near_constant(84, 200)
    idx = np.arange(target.size, dtype=np.uint32)
    values = [_near_constant(85, 200)]

    r_raw, n_raw = kernel(target, [idx], values)
    expected = pearson_reference(target, values[0])
    assert not np.isnan(expected)
    assert float(r_raw[0]) == pytest.approx(expected, abs=1e-10)
    assert int(n_raw[0]) == target.size


def test_near_constant_masked_handles_missing_samples():
    target = _near_constant(86, 180)
    target[np.arange(0, 180, 11)] = -9.0
    row = _near_constant(87, 180)
    row[5:9] = -9.0  # block-missing inside the target's valid samples

    r_raw, n_raw = pearson_pairwise_masked(target, row.reshape(1, -1))
    expected = pearson_reference(target, row)
    assert int(n_raw[0]) == int(((target >= 0) & (row >= 0)).sum())
    assert float(r_raw[0]) == pytest.approx(expected, abs=1e-10)


def test_block_near_constant_matches_force_masked_and_oracle():
    target = _near_constant(88, 120)
    target[np.arange(0, 120, 7)] = -9.0  # incomplete target -> relaxed routing

    dense = np.vstack([_near_constant(89, 120), _near_constant(90, 120)])
    dense[1, 3:6] = -9.0  # missing inside keep -> masked lane

    sparse_rows = []
    for j, seed in enumerate((91, 92, 93)):
        row = _near_constant(seed, 120)
        if j == 2:
            row[2:4] = -9.0  # missing inside keep -> masked lane
        sparse_rows.append(_sparsify(row))

    def build() -> BlockMatrix:
        return _blockmatrix(
            sample_ct=120,
            dense_rows=[0, 1],
            dense=[dense[0], dense[1]],
            sparse_rows=[2, 3, 4],
            sparse=sparse_rows,
        )

    block = build()
    r, n = pearson_pairwise_block(target, block)
    forced = build()
    r_forced, n_forced = pearson_pairwise_block(target, forced, force_masked=True)

    assert np.array_equal(n, n_forced)
    assert np.allclose(r, r_forced, atol=1e-10, equal_nan=True)
    assert int(block.stats.fast_rows + block.stats.masked_rows) == 5

    rows = [dense[0], dense[1]]
    for idx, values in sparse_rows:
        rows.append(_expand(idx, values, 120))
    for i, row in enumerate(rows):
        expected = pearson_reference(target, row)
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-10)


def test_block_random_missing_matches_force_masked_and_oracle():
    rng = np.random.default_rng(61)
    for _ in range(40):
        size = int(rng.integers(5, 45))
        target = rng.integers(0, 3, size=size).astype(float)
        target[rng.random(size) < 0.35] = -9.0
        n_dense = int(rng.integers(0, 4))
        n_sparse = int(rng.integers(0, 4))
        variants = n_dense + n_sparse
        if variants == 0:
            continue
        dense = rng.integers(0, 3, size=(n_dense, size)).astype(float)
        dense[rng.random(dense.shape) < 0.25] = -9.0
        sparse = []
        for _ in range(n_sparse):
            row = rng.integers(0, 3, size=size).astype(float)
            row[rng.random(size) < 0.3] = -9.0
            sparse.append(_sparsify(row))

        block = _blockmatrix(
            size,
            list(range(n_dense)),
            [row for row in dense],
            list(range(n_dense, variants)),
            sparse,
        )
        forced = _blockmatrix(
            size,
            list(range(n_dense)),
            [row for row in dense],
            list(range(n_dense, variants)),
            sparse,
        )
        r, n = pearson_pairwise_block(target, block)
        r_forced, n_forced = pearson_pairwise_block(target, forced, force_masked=True)
        assert np.array_equal(n, n_forced)
        assert np.allclose(r, r_forced, atol=1e-10, equal_nan=True)
        assert int(block.stats.fast_rows + block.stats.masked_rows) == variants

        full = np.zeros((variants, size))
        full[:n_dense] = dense
        for k, (idx, values) in enumerate(sparse):
            full[n_dense + k][idx] = values
        for i in range(variants):
            expected = pearson_reference(target, full[i])
            if np.isnan(expected):
                assert np.isnan(float(r[i]))
            else:
                assert float(r[i]) == pytest.approx(expected, abs=1e-9)
            assert int(n[i]) == int(((target >= 0) & (full[i] >= 0)).sum())


# --------------------------------------------------------------------------- #
# numerical-stability regressions (stable oracle + cancellation hazards)
# --------------------------------------------------------------------------- #

def test_reference_stable_contract():
    r, n = pearson_reference_stable(np.zeros(3), np.zeros(3))
    assert np.isnan(r) and n == 3
    r, n = pearson_reference_stable(np.full(3, 0.1), np.full(3, 0.1))
    assert np.isnan(r) and n == 3
    r, n = pearson_reference_stable(np.zeros(2), np.zeros(2))
    assert np.isnan(r) and n == 2
    r, n = pearson_reference_stable(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    )
    assert r == pytest.approx(1.0) and n == 3
    # Extreme dosage: squaring the variance product would underflow.
    u = np.array([0.0, 1e-100, 2e-100])
    assert pearson_reference_stable(u, u)[0] == pytest.approx(1.0)


def test_fallback_handles_multiple_flagged_rows():
    """Several (but not all) flagged rows must scatter, not broadcast."""

    target = np.array([0, 1, 2], dtype=np.int8)
    block = np.array([[1, 1, 1], [2, 2, 2], [0, 1, 2]], dtype=np.int8)
    for kernel in (pearson_pairwise_complete, pearson_pairwise_masked):
        r, n = kernel(target, block)
        np.testing.assert_allclose(r, [np.nan, np.nan, 1.0], equal_nan=True)
        np.testing.assert_array_equal(n, [3, 3, 3])


def test_dropped_moment_cancellation_is_recovered():
    """The retained target moments come from a subtracting a larger sum."""

    target = np.array([2.0, 2.0, 2.0, 1e-9, 2e-9, 3e-9])
    row = np.array([-9.0, -9.0, -9.0, 0.0, 1.0, 2.0])
    idx = np.arange(6, dtype=np.uint32)

    for r, n in (
        pearson_pairwise_masked(target, row[None, :]),
        pearson_pairwise_sparse_masked(target, [idx], [row]),
        pearson_pairwise_sparse(target, [idx], [row]),
    ):
        assert int(n[0]) == 3
        assert float(r[0]) == pytest.approx(1.0)


def test_constant_dosage_is_nan_everywhere():
    c = np.full(3, 0.1)
    idx = np.arange(3, dtype=np.uint32)
    for r, _ in (
        pearson_pairwise(c, c[None, :]),
        pearson_pairwise_masked(c, c[None, :]),
        pearson_pairwise_complete(c, c[None, :]),
        pearson_pairwise_sparse(c, [idx], [c]),
        pearson_pairwise_sparse_masked(c, [idx], [c]),
        pearson_pairwise_sparse_complete(c, [idx], [c]),
    ):
        assert np.isnan(float(r[0]))


def test_tiny_dosage_variance_product_does_not_underflow():
    u = np.array([0.0, 1e-100, 2e-100])
    idx = np.arange(3, dtype=np.uint32)
    for r, _ in (
        pearson_pairwise(u, u[None, :]),
        pearson_pairwise_masked(u, u[None, :]),
        pearson_pairwise_complete(u, u[None, :]),
        pearson_pairwise_sparse(u, [idx], [u]),
        pearson_pairwise_sparse_masked(u, [idx], [u]),
        pearson_pairwise_sparse_complete(u, [idx], [u]),
    ):
        assert float(r[0]) == pytest.approx(1.0)


def test_constant_target_keeps_pairwise_n():
    target = np.ones(5)
    block = np.array([[1.0, 2.0, -9.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0, 2.0]])
    r, n = pearson_pairwise_masked(target, block)
    assert np.isnan(r).all()
    assert list(int(x) for x in n) == [4, 5]


def test_constant_on_pairwise_valid_subset_is_nan():
    # Constant target on the pairwise-valid subset only.
    target = np.array([-9.0, -9.0, -9.0, 5.0, 5.0, 5.0])
    row = np.array([0.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    r, n = pearson_pairwise_masked(target, row[None, :])
    assert int(n[0]) == 3
    assert np.isnan(float(r[0]))

    # Constant block row on the pairwise-valid subset only.
    target = np.array([0.0, 1.0, 2.0, -9.0, -9.0, -9.0])
    row = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    r, n = pearson_pairwise_masked(target, row[None, :])
    assert int(n[0]) == 3
    assert np.isnan(float(r[0]))


@pytest.mark.parametrize("k", [0, 1, 2])
def test_kernels_nan_for_too_few_valid_samples(k):
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    row = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    row[:] = -9.0
    if k:
        row[:k] = [0.0, 1.0, 2.0][:k]
    idx, values = _sparsify(row)
    for r, n in (
        pearson_pairwise_masked(target, row[None, :]),
        pearson_pairwise_sparse_masked(target, [idx], [values]),
    ):
        assert int(n[0]) == k
        assert np.isnan(float(r[0]))


@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.float64])
@pytest.mark.parametrize("lane", ["complete", "masked"])
def test_dense_kernels_match_stable_reference(dtype, lane):
    rng = np.random.default_rng(101)
    size = 180
    target = rng.integers(0, 3, size=size).astype(float)
    target[rng.random(size) < 0.15] = -9.0
    mask = target >= 0
    block = rng.integers(0, 3, size=(6, size)).astype(dtype)
    if lane == "complete":
        block[:, ~mask] = dtype(-9)
        r, n = pearson_pairwise_complete(target, block, sample_mask=mask)
    else:
        block[rng.random((6, size)) < 0.15] = dtype(-9)
        r, n = pearson_pairwise_masked(target, block)

    for i in range(block.shape[0]):
        expected, expected_n = pearson_reference_stable(target, block[i].astype(float))
        assert int(n[i]) == expected_n
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("lane", ["complete", "masked"])
def test_sparse_kernels_match_stable_reference(lane):
    rng = np.random.default_rng(102)
    size = 180
    target = rng.integers(0, 3, size=size).astype(float)
    target[rng.random(size) < 0.15] = -9.0
    mask = target >= 0
    block = rng.integers(0, 3, size=(7, size)).astype(float)
    if lane == "complete":
        block[:, ~mask] = -9.0
    else:
        block[rng.random((7, size)) < 0.15] = -9.0
    idxs, values = zip(*[_sparsify(row) for row in block])
    if lane == "complete":
        r, n = pearson_pairwise_sparse_complete(
            target, list(idxs), list(values), sample_mask=mask
        )
    else:
        r, n = pearson_pairwise_sparse_masked(target, list(idxs), list(values))

    for i in range(block.shape[0]):
        expected, expected_n = pearson_reference_stable(target, block[i])
        assert int(n[i]) == expected_n
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("center", [0.0, 2.0])
def test_near_constant_around_center_matches_stable_reference(center):
    rng = np.random.default_rng(103)
    target = center + np.abs(rng.normal(0.0, 1e-9, 150))
    row = center + np.abs(rng.normal(0.0, 1e-9, 150))
    expected, expected_n = pearson_reference_stable(target, row)
    r, n = pearson_pairwise_masked(target, row[None, :])
    assert int(n[0]) == expected_n
    assert float(r[0]) == pytest.approx(expected, abs=1e-9)


def test_complete_sample_mask_ignores_missing_outside_mask():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, -9.0, -9.0])
    mask = target >= 0
    block = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, -9.0, -9.0]])

    r, n = pearson_pairwise_complete(target, block, sample_mask=mask)
    assert int(n[0]) == 6
    assert float(r[0]) == pytest.approx(1.0)

    idx, values = _sparsify(block[0])
    r, n = pearson_pairwise_sparse_complete(
        target, [idx], [values], sample_mask=mask
    )
    assert int(n[0]) == 6
    assert float(r[0]) == pytest.approx(1.0)


def test_dense_sparse_equivalence_with_implicit_zeros():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    dense = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]])
    idxs, values = zip(*[_sparsify(row) for row in dense])
    r_dense, n_dense = pearson_pairwise_masked(target, dense)
    r_sparse, n_sparse = pearson_pairwise_sparse_masked(target, list(idxs), list(values))
    assert np.array_equal(n_dense, n_sparse)
    assert np.allclose(r_dense, r_sparse, atol=1e-12, equal_nan=True)


def test_row_chunks_respect_the_byte_budget():
    from crossld.ld import _row_chunks

    flagged = np.arange(100)
    chunks = _row_chunks(flagged, size=1000, budget=1000 * 8 * 7)
    assert sum(len(chunk) for chunk in chunks) == 100
    assert all(len(chunk) <= 7 for chunk in chunks)
    assert np.array_equal(np.concatenate(chunks), flagged)


def test_block_all_rows_flagged_matches_stable_reference():
    target = _near_constant(104, 150)
    block = np.vstack([_near_constant(seed, 150) for seed in range(105, 115)])
    r, n = pearson_pairwise_block(
        target,
        _blockmatrix(
            sample_ct=150,
            dense_rows=list(range(10)),
            dense=[row for row in block],
            sparse_rows=[],
            sparse=[],
        ),
    )
    for i in range(block.shape[0]):
        expected, expected_n = pearson_reference_stable(target, block[i])
        assert int(n[i]) == expected_n
        assert float(r[i]) == pytest.approx(expected, abs=1e-9)

