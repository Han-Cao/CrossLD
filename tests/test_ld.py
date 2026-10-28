"""Unit tests for the LD math in :mod:`crossld.ld`.

The raw- and centered-moment kernels are checked against the independent
oracles in ``tests/conftest.py`` (``pearson_reference`` /
``pearson_reference_stable``), against each other, and dense against sparse on
the same data (unlisted samples are homozygous REF; listed negative values are
missing).
"""

from __future__ import annotations

import numpy as np
import pytest

from crossld.blocks import BlockMatrix, BlockStats
from crossld.ld import (
    SKIP_LOW_VARIANCE,
    SKIP_MISSING,
    _check_sparse_sample_ct_block_const,
    alt_freq,
    alt_freq_sparse,
    centered_moment_ld_dense,
    centered_moment_ld_sparse,
    major_allele_flipped,
    major_allele_flipped_sparse,
    pearson_pairwise_block,
    raw_moment_ld_dense,
    raw_moment_ld_sparse,
    target_qc,
)
from tests.conftest import pearson_reference, pearson_reference_stable

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

TARGET = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])


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
        sparse_values=[np.asarray(v, dtype=float) for _, v in sparse],
        stats=BlockStats(variants=variants),
    )


def _missing_row(rng: np.random.Generator, row: np.ndarray, rate: float) -> np.ndarray:
    row = np.asarray(row, dtype=float).copy()
    row[rng.random(row.size) < rate] = -9.0
    return row


# --------------------------------------------------------------------------- #
# allele-frequency / orientation helpers
# --------------------------------------------------------------------------- #

def test_alt_freq_ignores_missing():
    assert alt_freq(np.array([0.0, 1.0, 2.0, -9.0])) == pytest.approx(3.0 / (2 * 3))
    assert np.isnan(alt_freq(np.array([-9.0, -9.0])))
    assert np.isnan(alt_freq(np.array([])))


def test_alt_freq_sparse_treats_unlisted_as_reference():
    assert alt_freq_sparse(np.array([1.0, -9.0]), sample_ct=2) == pytest.approx(
        1.0 / (2 * 1)
    )
    assert alt_freq_sparse(np.array([]), sample_ct=4) == 0.0
    assert np.isnan(alt_freq_sparse(np.array([-9.0, -9.0]), sample_ct=2))


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.array([2.0, 2.0, 2.0, 2.0, 0.0]), True),
        (np.array([0.0, 0.0, 0.0, 0.0, 2.0]), False),
        (np.array([1.0, 1.0]), False),
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
        _idx, values = _sparsify(dense)
        assert major_allele_flipped_sparse(values, sample_ct=dense.size) is (
            major_allele_flipped(dense)
        )


# --------------------------------------------------------------------------- #
# dense centered moments
# --------------------------------------------------------------------------- #

def test_dense_centered_perfect_correlation():
    r, n = centered_moment_ld_dense(TARGET, TARGET.reshape(1, -1))
    assert float(r[0]) == pytest.approx(1.0)
    assert int(n[0]) == 6

    r, n = centered_moment_ld_dense(TARGET, (2.0 - TARGET).reshape(1, -1))
    assert float(r[0]) == pytest.approx(-1.0)
    assert int(n[0]) == 6


def test_dense_centered_pairwise_masking():
    a = np.array([0.0, 1.0, 2.0, -9.0, 1.0, 2.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 2.0, -9.0, 2.0, 0.0])
    r, n = centered_moment_ld_dense(a, b.reshape(1, -1), compute_mask=True)
    assert float(r[0]) == pytest.approx(1.0)
    assert int(n[0]) == 5


def test_dense_centered_matches_oracle_with_missing():
    rng = np.random.default_rng(7)
    target = rng.integers(0, 3, size=200).astype(float)
    target[rng.random(200) < 0.1] = -9.0
    block = rng.integers(0, 3, size=(12, 200)).astype(float)
    block[rng.random(block.shape) < 0.1] = -9.0

    r, n = centered_moment_ld_dense(target, block, compute_mask=True)
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        assert int(n[i]) == int(((target >= 0) & (block[i] >= 0)).sum())
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "target, row",
    [
        (np.zeros(6), np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
        (TARGET, np.full(6, 2.0)),
    ],
    ids=["constant-target", "constant-row"],
)
def test_dense_centered_zero_variance_is_nan(target, row):
    r, _ = centered_moment_ld_dense(target, row.reshape(1, -1))
    assert np.isnan(float(r[0]))


def test_dense_centered_fewer_than_three_valid_samples_is_nan():
    a = np.array([0.0, 1.0, -9.0, -9.0])
    b = np.array([0.0, 1.0, 2.0, 2.0])
    r, n = centered_moment_ld_dense(a, b.reshape(1, -1), compute_mask=True)
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 2


def test_dense_centered_constant_on_pairwise_subset_is_nan():
    # Target is non-constant overall but constant on the pairwise-valid samples.
    target = np.array([-9.0, -9.0, -9.0, 5.0, 5.0, 5.0])
    row = np.array([0.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    r, n = centered_moment_ld_dense(target, row.reshape(1, -1), compute_mask=True)
    assert int(n[0]) == 3
    assert np.isnan(float(r[0]))

    target = np.array([0.0, 1.0, 2.0, -9.0, -9.0, -9.0])
    row = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    r, n = centered_moment_ld_dense(target, row.reshape(1, -1), compute_mask=True)
    assert int(n[0]) == 3
    assert np.isnan(float(r[0]))


# --------------------------------------------------------------------------- #
# dense raw moments
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.float64])
def test_dense_raw_matches_centered(dtype):
    rng = np.random.default_rng(21)
    target = rng.integers(0, 3, size=250).astype(dtype)
    block = rng.integers(0, 3, size=(9, 250)).astype(dtype)
    r_centered, n_centered = centered_moment_ld_dense(target, block)
    r_raw, n_raw = raw_moment_ld_dense(target, block)
    assert np.array_equal(n_centered, n_raw)
    assert np.allclose(r_raw, r_centered, atol=1e-12, equal_nan=True)


def test_dense_raw_matches_centered_with_missing():
    rng = np.random.default_rng(22)
    target = _missing_row(rng, rng.integers(0, 3, size=200), 0.25)
    block = rng.integers(0, 3, size=(11, 200)).astype(float)
    block[rng.random(block.shape) < 0.2] = -9.0
    r_centered, n_centered = centered_moment_ld_dense(target, block, compute_mask=True)
    r_raw, n_raw = raw_moment_ld_dense(target, block, compute_mask=True)
    assert np.array_equal(n_centered, n_raw)
    assert np.allclose(r_raw, r_centered, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# mask contract
# --------------------------------------------------------------------------- #

def test_compute_mask_true_derives_masks_from_input():
    target = np.array([0.0, 1.0, 2.0, -9.0, 1.0, 2.0])
    block = np.array([[0.0, 1.0, 2.0, 2.0, -9.0, 2.0]])
    r, n = centered_moment_ld_dense(target, block, compute_mask=True)
    assert int(n[0]) == 4
    assert np.isfinite(float(r[0]))


def test_compute_mask_false_none_means_no_missing():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    block = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]])
    r_auto, n_auto = centered_moment_ld_dense(target, block, compute_mask=True)
    r_none, n_none = centered_moment_ld_dense(
        target, block, compute_mask=False
    )
    assert np.array_equal(n_auto, n_none)
    assert np.allclose(r_auto, r_none)


def test_explicit_masks_match_computed_masks():
    rng = np.random.default_rng(31)
    target = _missing_row(rng, rng.integers(0, 3, size=180), 0.3)
    block = rng.integers(0, 3, size=(7, 180)).astype(float)
    block[rng.random(block.shape) < 0.25] = -9.0

    target_valid = target >= 0
    block_valid = block >= 0

    r_auto, n_auto = centered_moment_ld_dense(target, block, compute_mask=True)
    r_mask, n_mask = centered_moment_ld_dense(
        target, block, target_valid, block_valid, compute_mask=False
    )
    assert np.array_equal(n_auto, n_mask)
    assert np.allclose(r_auto, r_mask, atol=1e-12, equal_nan=True)

    r_raw_mask, n_raw_mask = raw_moment_ld_dense(
        target, block, target_valid, block_valid, compute_mask=False
    )
    assert np.array_equal(n_auto, n_raw_mask)
    assert np.allclose(r_auto, r_raw_mask, atol=1e-12, equal_nan=True)


def test_explicit_masks_ignore_missing_outside_the_mask():
    """Values marked invalid outside the mask must not affect the result."""

    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, -9.0, -9.0])
    block = np.array([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, -9.0, -9.0]])
    target_valid = target >= 0
    block_valid = block >= 0
    r, n = centered_moment_ld_dense(
        target, block, target_valid, block_valid, compute_mask=False
    )
    assert int(n[0]) == 6
    assert float(r[0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# sparse kernels
# --------------------------------------------------------------------------- #

def test_sparse_centered_matches_dense():
    dense = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    idx, values = _sparsify(dense)
    r_dense, n_dense = centered_moment_ld_dense(TARGET, dense.reshape(1, -1))
    r_sparse, n_sparse = centered_moment_ld_sparse(TARGET, [idx], [values])
    assert float(r_sparse[0]) == pytest.approx(float(r_dense[0]))
    assert int(n_sparse[0]) == int(n_dense[0]) == 6


def test_sparse_raw_matches_centered():
    rng = np.random.default_rng(41)
    target = rng.integers(0, 3, size=160).astype(float)
    block = rng.integers(0, 3, size=(9, 160)).astype(float)
    idxs, values = zip(*[_sparsify(row) for row in block])
    r_centered, n_centered = centered_moment_ld_sparse(
        target, list(idxs), list(values)
    )
    r_raw, n_raw = raw_moment_ld_sparse(target, list(idxs), list(values))
    assert np.array_equal(n_centered, n_raw)
    assert np.allclose(r_raw, r_centered, atol=1e-12, equal_nan=True)


def test_sparse_matches_oracle_with_missing():
    rng = np.random.default_rng(42)
    target = rng.integers(0, 3, size=150).astype(float)
    target[rng.random(150) < 0.08] = -9.0
    block = rng.integers(0, 3, size=(10, 150)).astype(float)
    block[rng.random(block.shape) < 0.08] = -9.0
    idxs, values = zip(*[_sparsify(row) for row in block])

    r, n = centered_moment_ld_sparse(
        target, list(idxs), list(values), compute_mask=True
    )
    r_raw, n_raw = raw_moment_ld_sparse(
        target, list(idxs), list(values), compute_mask=True
    )
    for i in range(block.shape[0]):
        expected = pearson_reference(target, block[i])
        assert int(n[i]) == int(n_raw[i])
        assert int(n[i]) == int(((target >= 0) & (block[i] >= 0)).sum())
        if np.isnan(expected):
            assert np.isnan(float(r[i]))
            assert np.isnan(float(r_raw[i]))
        else:
            assert float(r[i]) == pytest.approx(expected, abs=1e-9)
            assert float(r_raw[i]) == pytest.approx(expected, abs=1e-9)


def test_sparse_empty_row_is_nan():
    r, n = centered_moment_ld_sparse(
        TARGET, [np.empty(0, dtype=np.uint32)], [np.empty(0, dtype=float)]
    )
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 6


def test_sparse_all_missing_row_is_nan():
    idx = np.arange(6, dtype=np.uint32)
    values = np.full(6, -9.0)
    r, n = centered_moment_ld_sparse(
        TARGET, [idx], [values], compute_mask=True
    )
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 0


def test_sparse_constant_row_is_nan():
    idx = np.arange(6, dtype=np.uint32)
    values = np.ones(6)
    r, _ = centered_moment_ld_sparse(TARGET, [idx], [values])
    assert np.isnan(float(r[0]))


def test_sparse_implicit_zeros_are_valid():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    dense = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]])
    idxs, values = zip(*[_sparsify(row) for row in dense])
    r_dense, n_dense = centered_moment_ld_dense(target, dense)
    r_sparse, n_sparse = centered_moment_ld_sparse(target, list(idxs), list(values))
    assert np.array_equal(n_dense, n_sparse)
    assert np.allclose(r_dense, r_sparse, atol=1e-12, equal_nan=True)


def test_sparse_masks_are_sparse_and_match_computed():
    rng = np.random.default_rng(43)
    target = _missing_row(rng, rng.integers(0, 3, size=140), 0.25)
    block = rng.integers(0, 3, size=(8, 140)).astype(float)
    block[rng.random(block.shape) < 0.2] = -9.0
    idxs, values = zip(*[_sparsify(row) for row in block])

    target_valid = target >= 0
    value_valid = [v >= 0 for v in values]

    r_auto, n_auto = centered_moment_ld_sparse(
        target, list(idxs), list(values), compute_mask=True
    )
    r_mask, n_mask = centered_moment_ld_sparse(
        target, list(idxs), list(values), target_valid, value_valid, compute_mask=False
    )
    assert np.array_equal(n_auto, n_mask)
    assert np.allclose(r_auto, r_mask, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# numerical stability (raw cancellation -> centered fallback)
# --------------------------------------------------------------------------- #

def _near_constant(seed: int, size: int, spread: float = 1e-8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.0 + rng.normal(0.0, spread, size)


@pytest.mark.parametrize("center", [0.0, 2.0])
def test_near_constant_centered_matches_stable_oracle(center):
    rng = np.random.default_rng(103)
    target = center + np.abs(rng.normal(0.0, 1e-9, 150))
    row = center + np.abs(rng.normal(0.0, 1e-9, 150))
    expected, expected_n = pearson_reference_stable(target, row)
    r, n = centered_moment_ld_dense(target, row[None, :])
    assert int(n[0]) == expected_n
    assert float(r[0]) == pytest.approx(expected, abs=1e-9)


def test_near_constant_raw_is_nan_but_fallback_recovers():
    target = _near_constant(81, 200)
    block = np.vstack([_near_constant(82, 200), _near_constant(83, 200)])
    r_raw, _ = raw_moment_ld_dense(target, block)
    r_centered, _ = centered_moment_ld_dense(target, block)
    for i in range(block.shape[0]):
        expected, _ = pearson_reference_stable(target, block[i])
        # The raw kernel flags these rows as unreliable ...
        assert np.isnan(float(r_raw[i]))
        # ... while the centered kernel reproduces the stable oracle.
        assert float(r_centered[i]) == pytest.approx(expected, abs=1e-10)


def test_tiny_dosage_variance_product_does_not_underflow():
    u = np.array([0.0, 1e-100, 2e-100])
    r, _ = centered_moment_ld_dense(u, u[None, :])
    assert float(r[0]) == pytest.approx(1.0)


def test_constant_dosage_is_nan():
    c = np.full(3, 0.1)
    idx = np.arange(3, dtype=np.uint32)
    for r, _ in (
        raw_moment_ld_dense(c, c[None, :]),
        centered_moment_ld_dense(c, c[None, :]),
        raw_moment_ld_sparse(c, [idx], [c]),
        centered_moment_ld_sparse(c, [idx], [c]),
    ):
        assert np.isnan(float(r[0]))


# --------------------------------------------------------------------------- #
# block entry point
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


def test_block_returns_results_in_requested_row_order():
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
    r, n = pearson_pairwise_block(TARGET, block)
    assert len(r) == len(n) == 4
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


def test_block_routing_splits_clean_and_missing_rows():
    block = _mixed_block()
    _r, n = pearson_pairwise_block(TARGET, block)
    assert block.stats.complete_pair == 3
    assert block.stats.incomplete_pair == 2
    assert block.stats.raw_moment_pair == 5
    assert block.stats.centered_moment_pair == 0
    assert [int(x) for x in n] == [6, 6, 6, 5, 5]


def test_block_force_centered_matches_default_and_skips_raw_method():
    default = _mixed_block()
    forced = _mixed_block()
    r_default, n_default = pearson_pairwise_block(TARGET, default)
    r_forced, n_forced = pearson_pairwise_block(TARGET, forced, force_centered=True)
    assert np.array_equal(n_default, n_forced)
    assert np.allclose(r_default, r_forced, atol=1e-12, equal_nan=True)
    # The completeness split describes the input, so it is unchanged; only the
    # method split changes when centered moments are forced.
    assert forced.stats.complete_pair == 3
    assert forced.stats.incomplete_pair == 2
    assert forced.stats.raw_moment_pair == 0
    assert forced.stats.centered_moment_pair == 5


def test_block_incomplete_target_routing():
    target = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -9.0])  # missing outside keep
    good = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -9.0])
    bad = np.array([0.0, 1.0, -9.0, 0.0, 1.0, 2.0])  # missing inside keep
    block = _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 1],
        dense=[good, bad],
        sparse_rows=[2, 3],
        sparse=[
            (np.array([5], dtype=np.uint32), np.array([2.0])),
            (np.array([2], dtype=np.uint32), np.array([-9.0])),
        ],
    )
    forced = _blockmatrix(
        sample_ct=6,
        dense_rows=[0, 1],
        dense=[good, bad],
        sparse_rows=[2, 3],
        sparse=[
            (np.array([5], dtype=np.uint32), np.array([2.0])),
            (np.array([2], dtype=np.uint32), np.array([-9.0])),
        ],
    )
    r, n = pearson_pairwise_block(target, block)
    r_forced, n_forced = pearson_pairwise_block(target, forced, force_centered=True)
    assert block.stats.complete_pair == 2
    assert block.stats.incomplete_pair == 2
    assert block.stats.raw_moment_pair + block.stats.centered_moment_pair == 4
    assert [int(x) for x in n] == [5, 4, 5, 4]
    assert np.array_equal(n, n_forced)
    assert np.allclose(r, r_forced, atol=1e-12, equal_nan=True)


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
    assert np.isnan(float(r[0]))
    assert int(n[0]) == 2


def test_block_random_missing_matches_stable_oracle():
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
        r_forced, n_forced = pearson_pairwise_block(target, forced, force_centered=True)
        assert np.array_equal(n, n_forced)
        assert np.allclose(r, r_forced, atol=1e-10, equal_nan=True)
        # some rows are includedd in both skip_* and *complte* | *moment*
        # so this variant count test is no longer valid
        # valid_pair = variants - block.stats.skip_missing
        # assert block.stats.complete_pair + block.stats.incomplete_pair == valid_pair
        # assert block.stats.raw_moment_pair + block.stats.centered_moment_pair == valid_pair

        full = np.zeros((variants, size))
        full[:n_dense] = dense
        for k, (idx, values) in enumerate(sparse):
            full[n_dense + k][idx] = values
        for i in range(variants):
            expected, expected_n = pearson_reference_stable(target, full[i])
            assert int(n[i]) == expected_n
            if np.isnan(expected):
                assert np.isnan(float(r[i]))
            else:
                assert float(r[i]) == pytest.approx(expected, abs=1e-9)


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


def test_dense_sparse_raw_equivalence_with_implicit_zeros():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    dense = np.array(
        [[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]]
    )
    idxs, values = zip(*[_sparsify(row) for row in dense])
    r_dense, n_dense = raw_moment_ld_dense(target, dense)
    r_sparse, n_sparse = raw_moment_ld_sparse(target, list(idxs), list(values))
    assert np.array_equal(n_dense, n_sparse)
    assert np.allclose(r_dense, r_sparse, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# target-level QC and the lean sparse sample-count check
# ---------------------------------------------------------------------------

def test_target_qc_reports_missing_low_variance_and_ok():
    assert target_qc(np.array([-9.0, 1.0, -9.0, -9.0, -9.0, -9.0])) == SKIP_MISSING
    assert target_qc(np.full(6, -9.0)) == SKIP_MISSING
    assert target_qc(np.array([])) == SKIP_MISSING
    assert target_qc(np.full(6, 0.1)) == SKIP_LOW_VARIANCE
    # constant over the valid samples only
    assert target_qc(np.array([2.0, 2.0, 2.0, -9.0, -9.0, -9.0])) == SKIP_LOW_VARIANCE
    assert target_qc(TARGET) is None
    assert target_qc(np.array([0.0, 1.0, 2.0])) is None


def test_check_sample_ct_sparse_counts_pairwise_valid_and_block_constancy():
    target = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    idx = np.arange(6, dtype=np.uint32)

    values = [np.array([0.0, 1.0, -9.0, 1.0, 2.0, 2.0])]
    n, bad = _check_sparse_sample_ct_block_const(target, [idx], values, None, [values[0] >= 0])
    assert int(n[0]) == 5
    assert not bad[0]

    values = [np.full(6, 2.0), np.array([-9.0, -9.0, -9.0, -9.0, 1.0, 2.0])]
    n, bad = _check_sparse_sample_ct_block_const(
        target, [idx, idx], values, None, [v >= 0 for v in values]
    )
    assert [int(x) for x in n] == [6, 2]  # constant row, then n < 3
    assert list(bad) == [True, True]


def test_check_sample_ct_sparse_ignores_pairwise_constant_target():
    # The target is constant only on the pairwise-valid samples; this check no
    # longer flags it (the centered kernel does), but n is still correct.
    target = np.array([0.1, 0.2, 0.3, 500.0, 500.0, 500.0])
    idx = np.arange(6, dtype=np.uint32)
    values = [np.array([-9.0, -9.0, -9.0, 4.0, 5.0, 6.0])]
    n, bad = _check_sparse_sample_ct_block_const(target, [idx], values, None, [values[0] >= 0])
    assert int(n[0]) == 3
    assert not bad[0]


# --------------------------------------------------------------------------- #
# sparse centered moments (native, no densification)
# --------------------------------------------------------------------------- #

def test_sparse_centered_constant_target_on_pairwise_subset_is_nan():
    target = np.array([0.1, 0.2, 0.3, 500.0, 500.0, 500.0])
    idx = np.arange(6, dtype=np.uint32)
    values = np.array([-9.0, -9.0, -9.0, 4.0, 5.0, 6.0])
    r, n = centered_moment_ld_sparse(target, [idx], [values], compute_mask=True)
    assert int(n[0]) == 3
    assert np.isnan(float(r[0]))


@pytest.mark.parametrize(
    "target, row",
    [
        (np.full(6, 2.0), np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
        (TARGET, np.full(6, 2.0)),
        (np.array([0.0, 1e-100, 2e-100]), np.array([0.0, 1e-100, 2e-100])),
        (np.array([0.0, 1e-100, 2e-100]), np.array([0.0, -9.0, 2e-100])),
    ],
    ids=["constant-target", "constant-row", "tiny-variance", "tiny-variance-missing"],
)
def test_sparse_centered_matches_dense_on_degenerate_cases(target, row):
    idx, values = _sparsify(row)
    r_dense, n_dense = centered_moment_ld_dense(target, row[None, :], compute_mask=True)
    r_sparse, n_sparse = centered_moment_ld_sparse(
        target, [idx], [values], compute_mask=True
    )
    assert int(n_sparse[0]) == int(n_dense[0])
    if np.isnan(r_dense[0]):
        assert np.isnan(float(r_sparse[0]))
    else:
        assert float(r_sparse[0]) == pytest.approx(float(r_dense[0]), abs=1e-12)


def test_sparse_centered_matches_stable_oracle_on_hard_cases():
    rng = np.random.default_rng(97)
    near_target = 1e-8 + rng.normal(0.0, 1e-11, 60)
    near_rows = 1e-8 + rng.normal(0.0, 1e-12, (5, 60))
    extreme_target = rng.random(40) * 1e120
    extreme_rows = rng.random((4, 40)) * 1e-120
    missing_target = rng.integers(0, 3, 80).astype(float)
    missing_target[rng.random(80) < 0.4] = -9.0
    missing_rows = rng.integers(0, 3, (6, 80)).astype(float)
    missing_rows[rng.random(missing_rows.shape) < 0.4] = -9.0

    for target, block in (
        (near_target, near_rows),
        (extreme_target, extreme_rows),
        (missing_target, missing_rows),
    ):
        for row in block:
            idx, values = _sparsify(row)
            r, n = centered_moment_ld_sparse(target, [idx], [values], compute_mask=True)
            expected, expected_n = pearson_reference_stable(target, row)
            assert int(n[0]) == expected_n
            if np.isnan(expected):
                assert np.isnan(float(r[0]))
            else:
                assert float(r[0]) == pytest.approx(expected, abs=1e-9)
