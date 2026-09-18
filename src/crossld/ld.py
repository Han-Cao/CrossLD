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
    """True when the ALT allele is the major.

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

    n = valid.sum(axis=1, dtype=np.uint32)
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


def pearson_pairwise_complete(
    target: npt.NDArray, block: npt.NDArray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Pearson r between ``target`` and every row of ``block`` with no missing values.

    Sum-based formulation (``n*sum_ab - sum_a*sum_b``) evaluated in float64
    without materialising a float64 copy of ``block``, so a large block does not
    allocate several dense temporaries the way :func:`pearson_pairwise` does.
    Every pair uses all samples, so ``n == target.size`` for every row. The
    ``n < 3`` and zero-variance NaN rules match :func:`pearson_pairwise`.
    """

    block = np.atleast_2d(block)
    n_rows = int(block.shape[0])
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)
    n = int(target.size)

    t = np.asarray(target, dtype=np.float64)
    sum_t = float(t.sum())
    sum_t2 = float(t @ t)

    sum_b = block.sum(axis=1, dtype=np.float64)
    sum_b2 = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    sum_tb = np.einsum("ij,j->i", block, t, dtype=np.float64)

    numerator = n * sum_tb - sum_b * sum_t
    var_b = n * sum_b2 - sum_b * sum_b
    var_t = n * sum_t2 - sum_t * sum_t

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / np.sqrt(var_b * var_t)

    bad = (n < 3) | (var_b <= 0) | (var_t <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)
    return r, np.full(n_rows, n, dtype=np.uint32)


def pearson_pairwise_sparse(
    target: np.ndarray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r between ``target`` and one or more sparse block rows.

    ``sample_idxs`` and ``values`` are lists containing one sparse array per row.
    To process one row, pass one-element lists: ``[idx]`` and ``[values]``.

    Returns ``(r, n)`` per row with the same NaN rules as
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
    n_out = np.zeros(n_rows, dtype=np.uint32)

    for k in range(n_rows):
        idx = sample_idxs[k]
        v = values[k]

        # idx.size == 0 -> var_x == 0, skip
        if idx.size == 0:
            n_out[k] = t0
            continue

        t_listed = target[idx]
        t_listed_valid = t_listed >= 0
        vt_valid = (v >= 0) & t_listed_valid

        # t0 already handles invalid target
        # so check missing in v & t_listed_valid v missing only
        v_missing = int(((v < 0) & t_listed_valid).sum())
        n = t0 - v_missing
        n_out[k] = n

        if n < 3:
            continue

        t_drop_by_v = (~vt_valid) & t_listed_valid
        sum_t = t1 - float(t_listed[t_drop_by_v].sum())
        sum_t2 = t2 - float((t_listed[t_drop_by_v] ** 2).sum())

        good_v = v[vt_valid]
        good_t = t_listed[vt_valid]
        sum_x = float(good_v.sum())
        sum_x2 = float((good_v * good_v).sum())
        sum_xt = float((good_v * good_t).sum())

        numerator = n * sum_xt - sum_x * sum_t
        var_x = n * sum_x2 - sum_x * sum_x
        var_t = n * sum_t2 - sum_t * sum_t
        if var_x <= 0 or var_t <= 0:
            continue
        denom = np.sqrt(var_x * var_t)
        # this only helps with var_x * var_t underflows to 0.0
        # to avoid dividing by 0.0, we do one more check
        if denom <= 0:
            continue
        r = numerator / denom
        if np.isfinite(r):
            r_out[k] = r

    return r_out, n_out


def pearson_pairwise_sparse_complete(
    target: np.ndarray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r for sparse rows that contain no missing calls.

    Every listed value is non-negative and ``target`` is complete, so unlisted
    samples (exactly 0) are valid and each pair uses all samples (target.size).
    Rows are processed together with :func:`numpy.bincount` instead of a Python
    loop. Empty rows have zero variance and come back as ``NaN``; the ``n < 3``
    and zero-variance rules match :func:`pearson_pairwise_sparse`.
    """

    n_rows = len(sample_idxs)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)
    n = int(target.size)

    t = np.asarray(target, dtype=np.float64)
    sum_t = float(t.sum())
    sum_t2 = float(t @ t)

    lengths = np.fromiter(
        (idx.size for idx in sample_idxs), dtype=np.uint32, count=n_rows
    )
    row_ids = np.repeat(np.arange(n_rows, dtype=np.uint32), lengths)
    if row_ids.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)

    v_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    i_all = np.concatenate([np.asarray(idx, dtype=np.uint32) for idx in sample_idxs])

    sum_x = np.bincount(row_ids, weights=v_all, minlength=n_rows)
    sum_x2 = np.bincount(row_ids, weights=v_all * v_all, minlength=n_rows)
    sum_xt = np.bincount(row_ids, weights=v_all * t[i_all], minlength=n_rows)

    numerator = n * sum_xt - sum_x * sum_t
    var_x = n * sum_x2 - sum_x * sum_x
    var_t = n * sum_t2 - sum_t * sum_t

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / np.sqrt(var_x * var_t)

    bad = (n < 3) | (var_x <= 0) | (var_t <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)
    return r, np.full(n_rows, n, dtype=np.uint32)


def pearson_pairwise_block(
    target: npt.NDArray,
    block: BlockMatrix,
    *,
    force_masked: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Pearson r between ``target`` and a :class:`crossld.blocks.BlockMatrix`.

    Returns ``(r, n)`` aligned with the block's requested row order (dense and
    sparse rows interleaved as they were requested).

    Rows without missing calls (and a complete ``target``) take a sum-based fast
    path; rows holding missing values keep the pairwise-masked path. Pass
    ``force_masked=True`` to disable the fast path (routing only, identical
    results) for differential tests and debugging. ``block.stats.fast_rows`` /
    ``block.stats.masked_rows`` record how many rows took each path.
    """

    n_rows = int(block.stats.variants)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.uint32)

    target_complete = (not force_masked) and bool(np.all(target >= 0))
    fast_rows = 0
    masked_rows = 0

    if block.dense_rows.size:
        dense = block.dense
        rows = block.dense_rows
        row_complete = (
            np.all(dense >= 0, axis=1)
            if target_complete
            else np.zeros(dense.shape[0], dtype=bool)
        )
        if row_complete.all():
            r[rows], n[rows] = pearson_pairwise_complete(target, dense)
            fast_rows += int(dense.shape[0])
        elif not row_complete.any():
            r[rows], n[rows] = pearson_pairwise(target, dense)
            masked_rows += int(dense.shape[0])
        else:
            fast_idx = np.flatnonzero(row_complete)
            masked_idx = np.flatnonzero(~row_complete)
            r[rows[fast_idx]], n[rows[fast_idx]] = pearson_pairwise_complete(
                target, dense[fast_idx]
            )
            r[rows[masked_idx]], n[rows[masked_idx]] = pearson_pairwise(
                target, dense[masked_idx]
            )
            fast_rows += int(fast_idx.size)
            masked_rows += int(masked_idx.size)

    if block.sparse_rows.size:
        sample_idxs = block.sparse_sample_idxs
        values = block.sparse_values
        rows = block.sparse_rows
        row_complete = (
            np.fromiter(
                (v.size == 0 or bool(np.all(v >= 0)) for v in values),
                dtype=bool,
                count=len(values),
            )
            if target_complete
            else np.zeros(len(values), dtype=bool)
        )
        if row_complete.all():
            r[rows], n[rows] = pearson_pairwise_sparse_complete(
                target, sample_idxs, values
            )
            fast_rows += len(values)
        elif not row_complete.any():
            r[rows], n[rows] = pearson_pairwise_sparse(
                target, sample_idxs, values
            )
            masked_rows += len(values)
        else:
            fast_idx = np.flatnonzero(row_complete)
            masked_idx = np.flatnonzero(~row_complete)
            r[rows[fast_idx]], n[rows[fast_idx]] = pearson_pairwise_sparse_complete(
                target,
                [sample_idxs[k] for k in fast_idx],
                [values[k] for k in fast_idx],
            )
            r[rows[masked_idx]], n[rows[masked_idx]] = pearson_pairwise_sparse(
                target,
                [sample_idxs[k] for k in masked_idx],
                [values[k] for k in masked_idx],
            )
            fast_rows += int(fast_idx.size)
            masked_rows += int(masked_idx.size)

    block.stats.fast_rows += fast_rows
    block.stats.masked_rows += masked_rows
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
    "pearson_pairwise_complete",
    "pearson_pairwise_sparse",
    "pearson_pairwise_sparse_complete",
]
