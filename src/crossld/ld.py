"""Linkage-disequilibrium math: pairwise unphased r / r² between variant vectors.

Vectors are additive allele dosages on the selected shared samples, with missing
calls encoded as negative values. Correlations are computed pairwise over the
non-missing samples of each pair.

Sparse rows follow :mod:`crossld.blocks`: ``sample_idxs`` are indexes into the
active sample subset and ``values`` are the corresponding ALT counts (missing
``< 0``); samples absent from ``sample_idxs`` are exactly 0 (homozygous REF) and
are valid observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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


def alt_freq(values: npt.NDArray) -> float:
    """ALT allele frequency of a dense ALT-count vector (missing values ignored)."""

    values = np.asarray(values)
    valid = values >= 0
    count = int(valid.sum())
    if count == 0:
        return float("nan")
    return float(values[valid].sum()) / (2.0 * count)


def alt_freq_sparse(values: npt.NDArray, sample_ct: int) -> float:
    """ALT allele frequency of one sparse row (unlisted samples are valid 0s)."""

    values = np.asarray(values)
    missing = int((values < 0).sum())
    count = int(sample_ct) - missing
    if count <= 0:
        return float("nan")
    return float(values[values >= 0].sum()) / (2.0 * count)


def major_allele_flipped(values: npt.NDArray) -> bool:
    """True when the ALT-coded vector must be mirrored to represent the major allele.

    Returns ``True`` when the ALT allele frequency is **at most** 0.5 (missing
    samples ignored), matching PLINK 2's convention; ties resolve to REF.
    """

    af = alt_freq(values)
    if not np.isfinite(af):
        return False
    return af <= 0.5


def major_allele_flipped_sparse(values: npt.NDArray, sample_ct: int) -> bool:
    """Sparse counterpart of :func:`major_allele_flipped` for one block row."""

    af = alt_freq_sparse(values, sample_ct)
    if not np.isfinite(af):
        return False
    return af <= 0.5


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


def pearson_pairwise_sparse(
    target: np.ndarray,
    sample_idxs: list[npt.NDArray[np.uint32]] | npt.NDArray[np.uint32],
    values: list[npt.NDArray] | npt.NDArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between ``target`` and one or more sparse block rows.

    ``sample_idxs``/``values`` are the per-row sparse arrays (a single row may be
    passed directly). Returns ``(r, n)`` per row with the same NaN rules as
    :func:`pearson_pairwise`; the per-row masked sums are computed without
    densifying.
    """

    single = isinstance(sample_idxs, np.ndarray)
    if single:
        sample_idxs = [sample_idxs]
        values = [values]

    target = np.asarray(target)
    t_valid = target >= 0
    t0 = float(t_valid.sum())
    t_valid_values = target[t_valid]
    t1 = float(t_valid_values.sum())
    t2 = float((t_valid_values * t_valid_values).sum())

    n_rows = len(sample_idxs)
    r_out = np.full(n_rows, np.nan, dtype=np.float64)
    n_out = np.zeros(n_rows, dtype=np.int64)

    for k in range(n_rows):
        idx = np.asarray(sample_idxs[k], dtype=np.int64)
        v = np.asarray(values[k], dtype=np.float64)

        if idx.size == 0:
            n = int(t0)
            sum_x = 0.0
            sum_x2 = 0.0
            sum_xt = 0.0
            sum_t = t1
            sum_t2 = t2
        else:
            t_listed = target[idx]
            t_ok = t_listed >= 0
            good = (v >= 0) & t_ok

            # Unlisted samples are valid, so start from every valid target
            # sample and drop only listed samples with a missing value.
            v_missing = int(((v < 0) & t_ok).sum())
            n = int(t0) - v_missing

            drop = (~good) & t_ok
            sum_t = t1 - float(t_listed[drop].sum())
            sum_t2 = t2 - float((t_listed[drop] ** 2).sum())

            good_v = v[good]
            good_t = t_listed[good]
            sum_x = float(good_v.sum())
            sum_x2 = float((good_v * good_v).sum())
            sum_xt = float((good_v * good_t).sum())

        n_out[k] = n
        if n < 3:
            continue

        numerator = n * sum_xt - sum_x * sum_t
        var_x = n * sum_x2 - sum_x * sum_x
        var_t = n * sum_t2 - sum_t * sum_t
        if var_x <= 0 or var_t <= 0:
            continue
        denom = np.sqrt(var_x * var_t)
        if denom <= 0:
            continue
        r = numerator / denom
        if np.isfinite(r):
            r_out[k] = r

    return r_out, n_out


def pearson_pairwise_block(target: np.ndarray, block) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between ``target`` and a :class:`crossld.blocks.BlockMatrix`.

    Returns ``(r, n)`` aligned with the block's requested row order (dense and
    sparse rows interleaved as they were requested). ``block`` is duck-typed
    (``dense``, ``dense_rows``, ``sparse_rows``, ``sparse_sample_idxs``,
    ``sparse_values``, ``stats``) so this module does not import ``blocks``.
    """

    n_rows = int(block.stats.variants)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.int64)

    if block.dense_rows.size:
        r_dense, n_dense = pearson_pairwise(target, block.dense)
        r[block.dense_rows] = r_dense
        n[block.dense_rows] = n_dense

    if block.sparse_rows.size:
        r_sparse, n_sparse = pearson_pairwise_sparse(
            target, block.sparse_sample_idxs, block.sparse_values
        )
        r[block.sparse_rows] = r_sparse
        n[block.sparse_rows] = n_sparse

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
    "alt_freq",
    "alt_freq_sparse",
    "compute_pair",
    "major_allele_flipped",
    "major_allele_flipped_sparse",
    "orient",
    "pearson_pairwise",
    "pearson_pairwise_block",
    "pearson_pairwise_sparse",
]
