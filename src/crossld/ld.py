"""Linkage-disequilibrium math: pairwise unphased r / r² between variant vectors.

Vectors are additive allele dosages on the selected shared samples, with missing
calls encoded as negative values. Correlations are computed pairwise over the
non-missing samples of each pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pair-level skip reasons, reported through logging.
SKIP_MISSING = "n<3"
SKIP_ZERO_VARIANCE = "zero_variance"
SKIP_SAME_ID = "same_id"


@dataclass(frozen=True)
class PairResult:
    """One LD result for ``target`` × ``other``."""

    index_a: int
    index_b: int
    r: float
    r2: float


def major_allele_flipped(values: np.ndarray) -> bool:
    """True when the ALT-coded vector must be mirrored to represent the major allele.

    Returns ``True`` when the ALT allele frequency is **at most** 0.5, i.e. the
    major allele is REF and the caller should recode with ``2 - x``. Ties (ALT
    frequency exactly 0.5) resolve to REF, matching PLINK 2's convention; the
    curated ``tests/data/ld_results_r_major.vcor`` reference pins this choice.
    """

    if values.size == 0:
        return False
    alt_freq = float(values.sum()) / (2.0 * values.size)
    return alt_freq <= 0.5


def orient(values: np.ndarray, flipped: bool) -> np.ndarray:
    """Recode an ALT-count vector to the REF allele when ``flipped`` is true."""

    if not flipped:
        return values
    out = np.where(values < 0, values, 2.0 - values)
    return out


def pearson_pairwise(target: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between ``target`` and every row of ``block``.

    Returns ``(r, n)`` where ``n`` is the per-pair non-missing sample count.
    Pairs with fewer than 3 usable samples or zero variance on either side come
    back as ``NaN``.
    """

    block = np.atleast_2d(block)
    valid_a = target >= 0
    valid_b = block >= 0
    valid = valid_a[None, :] & valid_b

    n = valid.sum(axis=1)
    a = np.where(valid, target[None, :], 0.0)
    b = np.where(valid, block, 0.0)

    denom_n = np.maximum(n, 1)
    mean_a = a.sum(axis=1) / denom_n
    mean_b = b.sum(axis=1) / denom_n

    da = a - mean_a[:, None]
    db = b - mean_b[:, None]
    da = np.where(valid, da, 0.0)
    db = np.where(valid, db, 0.0)

    cov = (da * db).sum(axis=1)
    var_a = (da * da).sum(axis=1)
    var_b = (db * db).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(var_a * var_b)

    bad = (n < 3) | (var_a <= 0) | (var_b <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)
    return r, n


def compute_pair(target: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """Scalar helper returning ``(r, r2)`` for a single pair (``NaN`` when invalid)."""

    r, _ = pearson_pairwise(target, other.reshape(1, -1))
    value = float(r[0])
    if np.isnan(value):
        return float("nan"), float("nan")
    clipped = max(-1.0, min(1.0, value))
    return clipped, clipped * clipped


__all__ = [
    "SKIP_MISSING",
    "SKIP_SAME_ID",
    "SKIP_ZERO_VARIANCE",
    "PairResult",
    "compute_pair",
    "major_allele_flipped",
    "orient",
    "pearson_pairwise",
]
