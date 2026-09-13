"""Unit tests for the LD math in :mod:`crossld.ld`."""

from __future__ import annotations

import numpy as np
import pytest

from crossld.ld import (
    compute_pair,
    major_allele_flipped,
    orient,
    pearson_pairwise,
)


def test_perfect_positive_correlation():
    a = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    r, r2 = compute_pair(a, a.copy())
    assert r == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_perfect_negative_correlation():
    a = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    b = 2.0 - a
    r, r2 = compute_pair(a, b)
    assert r == pytest.approx(-1.0)
    assert r2 == pytest.approx(1.0)


def test_uncorrelated_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=5000).astype(float)
    b = rng.integers(0, 3, size=5000).astype(float)
    r, r2 = compute_pair(a, b)
    assert abs(r) < 0.05
    assert r2 == pytest.approx(r * r)


def test_missing_values_are_pairwise_masked():
    a = np.array([0.0, 1.0, 2.0, -9.0, 1.0, 2.0, 0.0])
    b = np.array([0.0, 1.0, 2.0, 2.0, -9.0, 2.0, 0.0])
    r, r2 = compute_pair(a, b)
    # Only 5 samples are usable and they still form a perfectly ordered pair.
    assert r2 == pytest.approx(1.0)
    _, n = pearson_pairwise(a, b.reshape(1, -1))
    assert int(n[0]) == 5


def test_fewer_than_three_samples_is_nan():
    a = np.array([0.0, 1.0, -9.0, -9.0])
    b = np.array([0.0, 1.0, 2.0, 2.0])
    r, r2 = compute_pair(a, b)
    assert np.isnan(r) and np.isnan(r2)


def test_zero_variance_is_nan():
    a = np.zeros(6)
    b = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    r, r2 = compute_pair(a, b)
    assert np.isnan(r) and np.isnan(r2)


def test_pairwise_is_vectorised_over_blocks():
    target = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    block = np.array(
        [
            target,
            2.0 - target,
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            np.zeros(6),
        ]
    )
    r, n = pearson_pairwise(target, block)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-1.0)
    assert np.isfinite(r[2])
    assert np.isnan(r[3])
    assert list(n) == [6, 6, 6, 6]


def test_major_allele_flipped_uses_alt_frequency():
    # ALT copies 2 out of 20 -> ALT is minor -> major allele is REF -> flip.
    minor = np.array([0.0] * 9 + [2.0])
    assert major_allele_flipped(minor) is True
    major = np.array([2.0] * 9 + [0.0])
    assert major_allele_flipped(major) is False
    # A tie (ALT frequency exactly 0.5) resolves to REF, like PLINK 2.
    assert major_allele_flipped(np.array([1.0, 1.0])) is True
    assert major_allele_flipped(np.array([0.0, 2.0])) is True


def test_orient_mirrors_only_non_missing():
    values = np.array([0.0, 1.0, 2.0, -9.0])
    flipped = orient(values, True)
    assert flipped.tolist() == [2.0, 1.0, 0.0, -9.0]
    assert orient(values, False) is values


def test_major_allele_flipped_on_empty_input():
    assert major_allele_flipped(np.array([], dtype=float)) is False


def test_orient_returns_input_when_not_flipped():
    values = np.array([0.0, 1.0, 2.0, -9.0])
    assert orient(values, False) is values


def test_orient_flips_sign_of_r():
    a = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    b = 2.0 - a
    base, _ = compute_pair(a, b)
    # orient() maps an ALT-count vector to the reference representation. The
    # centred vector merely changes sign, so mirroring *both* sides leaves the
    # correlation unchanged (this is why --ref-based only matters for the sign
    # when the two datasets code opposite alleles).
    flipped, _ = compute_pair(orient(a, True), orient(b, True))
    assert base == pytest.approx(-1.0)
    assert flipped == pytest.approx(-1.0)


def test_single_sided_orient_flips_sign_of_r():
    a = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    b = 2.0 - a
    base, _ = compute_pair(a, b)
    # Mirroring only one side reverses the sign, which is exactly the effect
    # --ref-based has when one dataset's REF is the other's ALT.
    flipped, _ = compute_pair(orient(a, True), b)
    assert base == pytest.approx(-1.0)
    assert flipped == pytest.approx(1.0)
