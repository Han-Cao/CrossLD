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
    pearson_pairwise_sparse,
)
from tests.conftest import pearson_reference


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
