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

import numpy as np
import numpy.typing as npt

from crossld.blocks import BlockMatrix

# Pair-level skip reasons, reported through logging.
SKIP_MISSING = "n<3"
SKIP_ZERO_VARIANCE = "zero_variance"


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

    Returns ``True`` when the ALT allele frequency > 0.5.
    """

    af = alt_freq(values)
    if not np.isfinite(af):
        return False
    return af > 0.5


def major_allele_flipped_sparse(values: npt.NDArray, sample_ct: int) -> bool:
    """Sparse counterpart of :func:`major_allele_flipped` for one block row."""

    af = alt_freq_sparse(values, sample_ct)
    if not np.isfinite(af):
        return False
    return af > 0.5


def pearson_pairwise(target: npt.NDArray, block: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
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
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between ``target`` and one or more sparse block rows.

    ``sample_idxs``/``values`` are the per-row sparse arrays (a single row may be
    passed directly). Returns ``(r, n)`` per row with the same NaN rules as
    :func:`pearson_pairwise`; the per-row masked sums are computed without
    densifying.
    """

    t_valid = target >= 0
    t0 = int(t_valid.sum())
    t_valid_values = target[t_valid]
    t1 = float(t_valid_values.sum())
    t2 = float((t_valid_values * t_valid_values).sum())

    n_rows = len(sample_idxs)
    r_out = np.full(n_rows, np.nan, dtype=np.float64)
    n_out = np.zeros(n_rows, dtype=np.int64)

    for k in range(n_rows):
        idx = sample_idxs[k]
        v = values[k]

        if idx.size == 0:
            n = t0
            sum_x = 0.0
            sum_x2 = 0.0
            sum_xt = 0.0
            sum_t = t1
            sum_t2 = t2
        else:
            t_listed = target[idx]
            t_listed_valid = t_listed >= 0
            vt_valid = (v >= 0) & t_listed_valid

            # t0 already handles invalid target
            # so check missing in v & t_listed_valid v missing only
            v_missing = int(((v < 0) & t_listed_valid).sum())
            n = t0 - v_missing

            if n < 3:
                n_out[k] = n
                continue

            t_drop_by_v = (~vt_valid) & t_listed_valid
            sum_t = t1 - float(t_listed[t_drop_by_v].sum())
            sum_t2 = t2 - float((t_listed[t_drop_by_v] ** 2).sum())

            good_v = v[vt_valid]
            good_t = t_listed[vt_valid]
            sum_x = float(good_v.sum())
            sum_x2 = float((good_v * good_v).sum())
            sum_xt = float((good_v * good_t).sum())

        n_out[k] = n

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


def pearson_pairwise_block(target: npt.NDArray, block: BlockMatrix) -> tuple[npt.NDArray, npt.NDArray]:
    """Pearson r between ``target`` and a :class:`crossld.blocks.BlockMatrix`.

    Returns ``(r, n)`` aligned with the block's requested row order (dense and
    sparse rows interleaved as they were requested). ``block`` is duck-typed
    (``dense``, ``dense_rows``, ``sparse_rows``, ``sparse_sample_idxs``,
    ``sparse_values``, ``stats``) so this module does not import ``blocks``.
    """

    n_rows = int(block.stats.variants)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.uint32)

    # TODO:
    # faster path for data without NAs, this is more common case for LD analysis
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


__all__ = [
    "SKIP_MISSING",
    "SKIP_ZERO_VARIANCE",
    "alt_freq",
    "alt_freq_sparse",
    "major_allele_flipped",
    "major_allele_flipped_sparse",
    "pearson_pairwise",
    "pearson_pairwise_block",
    "pearson_pairwise_sparse",
]
