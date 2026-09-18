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

    # Exactly constant input must be NaN even though mean-subtracting a constant
    # float vector is not exact: it can leave tiny nonzero residuals, which would
    # make the centered variance positive and report a spurious correlation.
    if valid.shape[1]:
        a_min = np.min(np.where(valid, target[None, :], np.inf), axis=1)
        a_max = np.max(np.where(valid, target[None, :], -np.inf), axis=1)
        b_min = np.min(np.where(valid, block, np.inf), axis=1)
        b_max = np.max(np.where(valid, block, -np.inf), axis=1)
        constant = (a_min == a_max) | (b_min == b_max)
    else:
        constant = np.zeros(block.shape[0], dtype=bool)

    # sqrt(var_a) * sqrt(var_b), not sqrt(var_a * var_b): the product of two
    # representable variances can underflow even when the correlation is fine.
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (np.sqrt(var_a) * np.sqrt(var_b))

    bad = constant | (n < 3) | (var_a <= 0) | (var_b <= 0) | ~np.isfinite(r)
    r = np.clip(np.where(bad, np.nan, r), -1.0, 1.0)
    return r, n


# The raw-moment variance ``n*sum(x**2) - sum(x)**2`` subtracts two nearly equal
# terms for nearly constant (e.g. imputed fractional dosage) rows, so the result
# can carry a large relative error or even round to <= 0. Pairs whose variance is
# small enough for that are recomputed with the centered formulation below.
_RAW_MOMENT_MAX_REL_ERROR = 1e-6
_RAW_MOMENT_LIMIT = float(np.finfo(np.float64).eps) / _RAW_MOMENT_MAX_REL_ERROR
# Rows recomputed with the centered formulation are processed in chunks bounded
# by this budget, so a block with many flagged rows cannot undo the memory-lean
# design by densifying everything at once.
_CENTERED_BUDGET_BYTES = 64 * 1024 * 1024
_OUT_OF_RANGE_TOL = 1e-9


def _raw_moment_unstable(
    var: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64] | int,
    sum_sq: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Heuristic flag for a raw-moment variance that cancellation may dominate.

    ``_RAW_MOMENT_LIMIT`` is a heuristic threshold (``eps / 1e-6``), not a
    guaranteed error bound: it does not account for accumulation error, for
    subtraction of dropped samples (see :func:`_subtraction_unstable`), or for
    covariance cancellation. ``sum_sq`` is the second-moment sum over the same
    valid set as ``var``, so ``n*sum_sq`` bounds the magnitudes being subtracted.
    """

    term = np.asarray(n, dtype=np.float64) * sum_sq
    limit = _RAW_MOMENT_LIMIT * term
    with np.errstate(invalid="ignore"):
        return (term > 0.0) & (var <= limit)


def _subtraction_unstable(
    kept: npt.NDArray[np.float64] | np.float64,
    total: npt.NDArray[np.float64] | np.float64,
    dropped: npt.NDArray[np.float64] | np.float64,
) -> npt.NDArray[np.bool_]:
    """Heuristic flag for ``kept == total - dropped`` losing the kept moment.

    The retained sum is compared against the magnitude of the operands that were
    subtracted, so an inaccurate but *nonzero* result is still flagged; testing
    ``kept != 0`` would not be enough.
    """

    scale = np.maximum(np.abs(total), np.abs(dropped))
    with np.errstate(invalid="ignore"):
        return (scale > 0.0) & (np.abs(kept) <= _RAW_MOMENT_LIMIT * scale)


def _target_is_constant(
    target: npt.NDArray, mask: npt.NDArray[np.bool_] | None = None
) -> bool:
    """True when the target is exactly constant over ``mask`` (or everywhere)."""

    values = np.asarray(target if mask is None else target[mask])
    if values.size == 0:
        return True
    return bool(values.min() == values.max())


def _out_of_range(r: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Finite correlations materially outside ``[-1, 1]`` (rounding artefacts)."""

    with np.errstate(invalid="ignore"):
        return np.isfinite(r) & (np.abs(r) > 1.0 + _OUT_OF_RANGE_TOL)


def _row_chunks(
    flagged: npt.NDArray[np.intp], size: int, budget: int = _CENTERED_BUDGET_BYTES
) -> list[npt.NDArray[np.intp]]:
    """Split flagged row indexes into chunks of at most ``budget`` float64 bytes."""

    per_row = max(1, int(size)) * 8
    step = max(1, budget // per_row)
    return [flagged[start : start + step] for start in range(0, len(flagged), step)]


def _centered_dense(
    target: npt.NDArray,
    block_rows: npt.NDArray,
    sample_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Centered (mean-subtracted) pairwise r for a small set of dense rows.

    Samples outside ``sample_mask`` are forced missing so the centered oracle
    uses exactly the samples the raw-moment caller used.
    """

    t = np.asarray(target, dtype=np.float64)
    rows = block_rows
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        t = np.where(mask, t, -1.0)
        rows = np.where(mask[None, :], rows, -1.0)
    return pearson_pairwise(t, rows)


def _centered_sparse(
    target: npt.NDArray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    sample_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Centered pairwise r for a small set of sparse rows.

    Only the flagged rows are densified, so :func:`pearson_pairwise` can apply
    the stable formulation without a full-block float64 temporary.
    """

    size = int(np.asarray(target).size)
    dense = np.zeros((len(sample_idxs), size), dtype=np.float64)
    for row, (idx, vals) in enumerate(zip(sample_idxs, values)):
        dense[row, np.asarray(idx)] = np.asarray(vals)
    return _centered_dense(target, dense, sample_mask)


def pearson_pairwise_masked(
    target: npt.NDArray, block: npt.NDArray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Memory-lean pairwise-masked Pearson r between ``target`` and ``block``.

    Same contract as :func:`pearson_pairwise` (``n`` is the per-pair non-missing
    sample count, and the ``n < 3`` / zero-variance / non-finite rules produce
    ``NaN``), but the sums are accumulated with :func:`numpy.einsum` and
    :func:`numpy.bincount` over each pair's valid set instead of materialising
    ``(rows, samples)`` float64 temporaries. Peak extra memory is one
    native-dtype copy of ``block`` (zeroed at missing calls) plus the index
    arrays of the invalid cells.

    Missing calls are detected as ``not (value >= 0)``, so a ``NaN`` dosage also
    counts as missing rather than as a valid observation. Nearly constant rows,
    whose variance the raw-moment identity cannot resolve, are recomputed with
    the centered formulation.
    """

    block = np.atleast_2d(block)
    n_rows = int(block.shape[0])
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)

    t = np.asarray(target, dtype=np.float64)
    target_valid = t >= 0
    n_keep = int(target_valid.sum())
    tw = np.where(target_valid, t, 0.0)
    sum_t = float(tw.sum())
    sum_t2 = float(tw @ tw)

    # Enumerate the invalid cells before ``zeroed`` exists so only one full-size
    # temporary is live at a time.
    rows, cols = np.nonzero(~(block >= 0))
    if rows.size:
        keep_entry = target_valid[cols]
        rows = rows[keep_entry]
        cols = cols[keep_entry]

    # Zeroing the invalid cells keeps the block-side sums exact and avoids
    # subtracting the (potentially huge) negative missing markers back out.
    zeroed = np.where(block >= 0, block, 0)
    weight = target_valid.astype(np.float64)

    sum_b = np.einsum("ij,j->i", zeroed, weight, dtype=np.float64)
    sum_b2 = np.einsum("ij,ij,j->i", zeroed, zeroed, weight, dtype=np.float64)
    sum_ab = np.einsum("ij,j->i", zeroed, tw, dtype=np.float64)

    n = np.full(n_rows, n_keep, dtype=np.uint32)
    sum_a = np.full(n_rows, sum_t, dtype=np.float64)
    sum_a2 = np.full(n_rows, sum_t2, dtype=np.float64)
    bad_sum = np.zeros(n_rows, dtype=np.float64)
    bad_sum2 = np.zeros(n_rows, dtype=np.float64)
    if rows.size:
        n = (n_keep - np.bincount(rows, minlength=n_rows)).astype(np.uint32)
        t_bad = t[cols]
        bad_sum = np.bincount(rows, weights=t_bad, minlength=n_rows)
        bad_sum2 = np.bincount(rows, weights=t_bad * t_bad, minlength=n_rows)
        sum_a = sum_a - bad_sum
        sum_a2 = sum_a2 - bad_sum2

    nf = n.astype(np.float64)
    numerator = nf * sum_ab - sum_a * sum_b
    var_a = nf * sum_a2 - sum_a * sum_a
    var_b = nf * sum_b2 - sum_b * sum_b

    target_constant = _target_is_constant(t, target_valid)

    # sqrt(var_a) * sqrt(var_b), not sqrt(var_a * var_b): the product can
    # underflow even when both variances are representable.
    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / (np.sqrt(var_a) * np.sqrt(var_b))

    bad = (
        target_constant
        | (n < 3)
        | (var_a <= 0)
        | (var_b <= 0)
        | ~np.isfinite(r)
    )
    r = np.where(bad, np.nan, r)

    unstable = (
        _raw_moment_unstable(var_a, nf, sum_a2)
        | _raw_moment_unstable(var_b, nf, sum_b2)
        | _subtraction_unstable(sum_a, sum_t, bad_sum)
        | _subtraction_unstable(sum_a2, sum_t2, bad_sum2)
        | _out_of_range(r)
    )
    unstable &= (n >= 3) & (not target_constant)
    for chunk in _row_chunks(np.flatnonzero(unstable), t.size):
        r_centered, _ = _centered_dense(t, block[chunk])
        r[chunk] = r_centered
    r = np.clip(r, -1.0, 1.0)
    return r, n


def pearson_pairwise_complete(
    target: npt.NDArray,
    block: npt.NDArray,
    *,
    sample_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Pearson r between ``target`` and every row of ``block`` with no missing values.

    Every row of ``block`` must hold a valid call on all samples selected by
    ``sample_mask``; ``sample_mask=None`` (the default) selects every sample.
    ``target`` values outside the mask are ignored and may be missing, so callers
    can restrict each pair to the target's non-missing samples.

    Sum-based formulation (``n*sum_ab - sum_a*sum_b``) evaluated in float64
    without materialising a float64 copy of ``block``, so a large block does not
    allocate several dense temporaries the way :func:`pearson_pairwise` does.
    Every pair uses the selected samples, so ``n`` is the mask size for every
    row. The ``n < 3`` and zero-variance NaN rules match
    :func:`pearson_pairwise`. Nearly constant rows, whose variance the raw-moment
    identity cannot resolve, are recomputed with the centered formulation.
    """

    block = np.atleast_2d(block)
    n_rows = int(block.shape[0])
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)

    t = np.asarray(target, dtype=np.float64)
    if sample_mask is None:
        mask = None
        n = int(target.size)
        tw = t
        weight = None
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        n = int(mask.sum())
        tw = np.where(mask, t, 0.0)
        weight = mask.astype(np.float64)

    sum_t = float(tw.sum())
    sum_t2 = float(tw @ tw)

    if weight is None:
        sum_b = block.sum(axis=1, dtype=np.float64)
        sum_b2 = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    else:
        sum_b = np.einsum("ij,j->i", block, weight, dtype=np.float64)
        sum_b2 = np.einsum("ij,ij,j->i", block, block, weight, dtype=np.float64)
    sum_tb = np.einsum("ij,j->i", block, tw, dtype=np.float64)

    numerator = n * sum_tb - sum_b * sum_t
    var_b = n * sum_b2 - sum_b * sum_b
    var_t = n * sum_t2 - sum_t * sum_t

    target_constant = _target_is_constant(t, mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / (np.sqrt(var_b) * np.sqrt(var_t))

    bad = target_constant | (n < 3) | (var_b <= 0) | (var_t <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)

    unstable = (
        _raw_moment_unstable(var_b, n, sum_b2)
        | _raw_moment_unstable(var_t, n, sum_t2)
        | _out_of_range(r)
    )
    unstable &= (n >= 3) & (not target_constant)
    for chunk in _row_chunks(np.flatnonzero(unstable), t.size):
        r_centered, _ = _centered_dense(t, block[chunk], mask)
        r[chunk] = r_centered
    r = np.clip(r, -1.0, 1.0)
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
    densifying. Nearly constant rows, whose variance the raw-moment identity
    would lose to cancellation, are recomputed with the centered formulation.
    """

    t_valid = target >= 0
    t0 = int(t_valid.sum())
    t_valid_values = target[t_valid]
    t1 = float(t_valid_values.sum())
    t2 = float((t_valid_values * t_valid_values).sum())

    n_rows = len(sample_idxs)
    r_out = np.full(n_rows, np.nan, dtype=np.float64)
    n_out = np.zeros(n_rows, dtype=np.uint32)
    unstable = np.zeros(n_rows, dtype=bool)

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
        dropped_t = float(t_listed[t_drop_by_v].sum())
        dropped_t2 = float((t_listed[t_drop_by_v] ** 2).sum())
        sum_t = t1 - dropped_t
        sum_t2 = t2 - dropped_t2

        good_v = v[vt_valid]
        good_t = t_listed[vt_valid]
        sum_x = float(good_v.sum())
        sum_x2 = float((good_v * good_v).sum())
        sum_xt = float((good_v * good_t).sum())

        numerator = n * sum_xt - sum_x * sum_t
        var_x = n * sum_x2 - sum_x * sum_x
        var_t = n * sum_t2 - sum_t * sum_t
        # Raw moments lose precision for nearly constant data, and the retained
        # target moments are themselves a difference of larger sums; flag both
        # for a centered recomputation after the loop.
        if (
            bool(_raw_moment_unstable(np.float64(var_x), n, np.float64(sum_x2)))
            or bool(_raw_moment_unstable(np.float64(var_t), n, np.float64(sum_t2)))
            or bool(
                _subtraction_unstable(np.float64(sum_t), np.float64(t1), dropped_t)
            )
            or bool(
                _subtraction_unstable(np.float64(sum_t2), np.float64(t2), dropped_t2)
            )
        ):
            unstable[k] = True
        if var_x <= 0 or var_t <= 0:
            continue
        # sqrt(var_x) * sqrt(var_t) avoids underflow of the variance product.
        denom = np.sqrt(var_x) * np.sqrt(var_t)
        if denom <= 0:
            continue
        r = numerator / denom
        if np.isfinite(r):
            r_out[k] = r

    if _target_is_constant(target, t_valid):
        r_out[:] = np.nan
        unstable[:] = False

    flagged = np.flatnonzero(unstable)
    for chunk in _row_chunks(flagged, int(target.size)):
        r_centered, n_centered = _centered_sparse(
            target,
            [sample_idxs[k] for k in chunk],
            [values[k] for k in chunk],
        )
        r_out[chunk] = r_centered
        n_out[chunk] = n_centered

    return np.clip(r_out, -1.0, 1.0), n_out


def pearson_pairwise_sparse_masked(
    target: np.ndarray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-lean pairwise-masked sparse Pearson r for one or more block rows.

    Same contract as :func:`pearson_pairwise_sparse` (``sample_idxs`` and
    ``values`` hold one sparse array per row, unlisted samples are exactly 0 and
    valid, and the ``n < 3`` / zero-variance / non-finite rules produce ``NaN``),
    but the per-row Python loop is replaced by vectorised :func:`numpy.bincount`
    accumulation over every row at once.

    Missing calls are detected as ``not (value >= 0)`` for both sides, so a
    ``NaN`` dosage counts as missing. Nearly constant rows, whose variance the
    raw-moment identity cannot resolve, are recomputed with the centered
    formulation.
    """

    n_rows = len(sample_idxs)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)

    t = np.asarray(target, dtype=np.float64)
    target_valid = t >= 0
    n_keep = int(target_valid.sum())
    tw = np.where(target_valid, t, 0.0)
    sum_t = float(tw.sum())
    sum_t2 = float(tw @ tw)

    lengths = np.fromiter(
        (idx.size for idx in sample_idxs), dtype=np.uint32, count=n_rows
    )
    row_ids = np.repeat(np.arange(n_rows, dtype=np.uint32), lengths)
    v_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    i_all = np.concatenate([np.asarray(idx, dtype=np.uint32) for idx in sample_idxs])

    n = np.full(n_rows, n_keep, dtype=np.uint32)
    sum_a = np.full(n_rows, sum_t, dtype=np.float64)
    sum_a2 = np.full(n_rows, sum_t2, dtype=np.float64)
    bad_sum = np.zeros(n_rows, dtype=np.float64)
    bad_sum2 = np.zeros(n_rows, dtype=np.float64)
    sum_x = np.zeros(n_rows, dtype=np.float64)
    sum_x2 = np.zeros(n_rows, dtype=np.float64)
    sum_xt = np.zeros(n_rows, dtype=np.float64)

    if row_ids.size:
        listed_target_valid = target_valid[i_all]
        listed_block_valid = v_all >= 0

        both = listed_target_valid & listed_block_valid
        if both.any():
            ids = row_ids[both]
            vals = v_all[both]
            sum_x = np.bincount(ids, weights=vals, minlength=n_rows)
            sum_x2 = np.bincount(ids, weights=vals * vals, minlength=n_rows)
            sum_xt = np.bincount(
                ids, weights=vals * t[i_all[both]], minlength=n_rows
            )

        # Samples the target observes but this row does not: they leave the pair
        # and shrink both ``n`` and the target-side sums.
        dropped = listed_target_valid & ~listed_block_valid
        if dropped.any():
            ids = row_ids[dropped]
            t_drop = t[i_all[dropped]]
            n = (n_keep - np.bincount(ids, minlength=n_rows)).astype(np.uint32)
            bad_sum = np.bincount(ids, weights=t_drop, minlength=n_rows)
            bad_sum2 = np.bincount(ids, weights=t_drop * t_drop, minlength=n_rows)
            sum_a = sum_a - bad_sum
            sum_a2 = sum_a2 - bad_sum2

    nf = n.astype(np.float64)
    numerator = nf * sum_xt - sum_a * sum_x
    var_a = nf * sum_a2 - sum_a * sum_a
    var_x = nf * sum_x2 - sum_x * sum_x

    target_constant = _target_is_constant(t, target_valid)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / (np.sqrt(var_a) * np.sqrt(var_x))

    bad = target_constant | (n < 3) | (var_a <= 0) | (var_x <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)

    unstable = (
        _raw_moment_unstable(var_a, nf, sum_a2)
        | _raw_moment_unstable(var_x, nf, sum_x2)
        | _subtraction_unstable(sum_a, sum_t, bad_sum)
        | _subtraction_unstable(sum_a2, sum_t2, bad_sum2)
        | _out_of_range(r)
    )
    unstable &= (n >= 3) & (not target_constant)
    for chunk in _row_chunks(np.flatnonzero(unstable), t.size):
        r_centered, _ = _centered_sparse(
            target,
            [sample_idxs[k] for k in chunk],
            [values[k] for k in chunk],
        )
        r[chunk] = r_centered
    r = np.clip(r, -1.0, 1.0)
    return r, n


def pearson_pairwise_sparse_complete(
    target: np.ndarray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    *,
    sample_mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r for sparse rows that contain no missing calls.

    Every listed value must be non-negative on the samples selected by
    ``sample_mask`` (``None`` selects every sample). Listed values outside the
    mask are dropped, so block-missing calls that coincide with target-missing
    samples are ignored. Unlisted samples are exactly 0 and are valid, and each
    pair uses the selected samples, so ``n`` is the mask size for every row.
    Rows are processed together with :func:`numpy.bincount` instead of a Python
    loop. Empty rows have zero variance and come back as ``NaN``; the ``n < 3``
    and zero-variance rules match :func:`pearson_pairwise_sparse`. Nearly constant
    rows, whose variance the raw-moment identity cannot resolve, are recomputed
    with the centered formulation.
    """

    n_rows = len(sample_idxs)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint32)

    t = np.asarray(target, dtype=np.float64)
    if sample_mask is None:
        mask = None
        n = int(target.size)
        sum_t = float(t.sum())
        sum_t2 = float(t @ t)
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        n = int(mask.sum())
        tw = np.where(mask, t, 0.0)
        sum_t = float(tw.sum())
        sum_t2 = float(tw @ tw)

    lengths = np.fromiter(
        (idx.size for idx in sample_idxs), dtype=np.uint32, count=n_rows
    )
    row_ids = np.repeat(np.arange(n_rows, dtype=np.uint32), lengths)
    v_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    i_all = np.concatenate([np.asarray(idx, dtype=np.uint32) for idx in sample_idxs])

    if mask is not None and row_ids.size:
        keep_entry = mask[i_all]
        row_ids = row_ids[keep_entry]
        v_all = v_all[keep_entry]
        i_all = i_all[keep_entry]

    if row_ids.size == 0:
        # No listed entries remain, so every row is constant (zero variance).
        return (
            np.full(n_rows, np.nan, dtype=np.float64),
            np.full(n_rows, n, dtype=np.uint32),
        )

    sum_x = np.bincount(row_ids, weights=v_all, minlength=n_rows)
    sum_x2 = np.bincount(row_ids, weights=v_all * v_all, minlength=n_rows)
    sum_xt = np.bincount(row_ids, weights=v_all * t[i_all], minlength=n_rows)

    numerator = n * sum_xt - sum_x * sum_t
    var_x = n * sum_x2 - sum_x * sum_x
    var_t = n * sum_t2 - sum_t * sum_t

    target_constant = _target_is_constant(t, mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = numerator / (np.sqrt(var_x) * np.sqrt(var_t))

    bad = target_constant | (n < 3) | (var_x <= 0) | (var_t <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)

    unstable = (
        _raw_moment_unstable(var_x, n, sum_x2)
        | _raw_moment_unstable(var_t, n, sum_t2)
        | _out_of_range(r)
    )
    unstable &= (n >= 3) & (not target_constant)
    for chunk in _row_chunks(np.flatnonzero(unstable), t.size):
        r_centered, _ = _centered_sparse(
            target,
            [sample_idxs[k] for k in chunk],
            [values[k] for k in chunk],
            mask,
        )
        r[chunk] = r_centered
    r = np.clip(r, -1.0, 1.0)
    return r, np.full(n_rows, n, dtype=np.uint32)


def _dense_complete_on_keep(
    dense: npt.NDArray, keep: npt.NDArray[np.bool_] | None
) -> npt.NDArray[np.bool_]:
    """Rows of ``dense`` with a valid call on every sample selected by ``keep``."""

    if keep is None:
        return np.all(dense >= 0, axis=1)
    # ``min`` over the selected samples: a ``NaN`` inside ``keep`` propagates and
    # marks the row ineligible, matching ``np.all(dense >= 0)`` for NaN.
    return np.min(np.where(keep[None, :], dense, 0), axis=1) >= 0


def _sparse_complete_on_keep(
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    keep: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Sparse rows with a valid call on every sample selected by ``keep``.

    Block-missing calls that fall on samples the target does not observe
    (``~keep``) are irrelevant and keep the row on the fast path.
    """

    if keep is None:
        return np.fromiter(
            (bool(np.all(v >= 0)) for v in values), dtype=bool, count=len(values)
        )
    return np.fromiter(
        (bool(np.all((v >= 0) | ~keep[idx])) for idx, v in zip(sample_idxs, values)),
        dtype=bool,
        count=len(values),
    )


def pearson_pairwise_block(
    target: npt.NDArray,
    block: BlockMatrix,
    *,
    force_masked: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Pearson r between ``target`` and a :class:`crossld.blocks.BlockMatrix`.

    Returns ``(r, n)`` aligned with the block's requested row order (dense and
    sparse rows interleaved as they were requested).

    Rows complete on all of the target's non-missing samples take a sum-based
    fast path (``n`` is then the target's non-missing sample count); the
    remaining rows take the pairwise-masked path. Pass ``force_masked=True`` to
    route every row through the original masked functions
    (:func:`pearson_pairwise` / :func:`pearson_pairwise_sparse`) for differential
    tests and debugging. ``block.stats.fast_rows`` / ``block.stats.masked_rows``
    record how many rows took each path.
    """

    n_rows = int(block.stats.variants)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.uint32)

    target_valid = target >= 0
    target_complete = bool(target_valid.all())
    n_keep = int(target_valid.sum())

    # The complete-on-keep lane only helps when the target has missing samples
    # and enough of them survive for a pair to clear the n >= 3 rule.
    use_fast = not force_masked
    keep: npt.NDArray[np.bool_] | None = None
    if use_fast and not target_complete:
        if n_keep < 3:
            use_fast = False
        else:
            keep = target_valid

    dense_masked = pearson_pairwise if force_masked else pearson_pairwise_masked
    sparse_masked = (
        pearson_pairwise_sparse if force_masked else pearson_pairwise_sparse_masked
    )

    fast_rows = 0
    masked_rows = 0

    if block.dense_rows.size:
        dense = block.dense
        rows = block.dense_rows
        if not use_fast:
            r[rows], n[rows] = dense_masked(target, dense)
            masked_rows += int(dense.shape[0])
        else:
            row_ok = _dense_complete_on_keep(dense, keep)
            if row_ok.all():
                r[rows], n[rows] = pearson_pairwise_complete(
                    target, dense, sample_mask=keep
                )
                fast_rows += int(dense.shape[0])
            elif not row_ok.any():
                r[rows], n[rows] = dense_masked(target, dense)
                masked_rows += int(dense.shape[0])
            else:
                fast_idx = np.flatnonzero(row_ok)
                masked_idx = np.flatnonzero(~row_ok)
                r[rows[fast_idx]], n[rows[fast_idx]] = pearson_pairwise_complete(
                    target, dense[fast_idx], sample_mask=keep
                )
                r[rows[masked_idx]], n[rows[masked_idx]] = dense_masked(
                    target, dense[masked_idx]
                )
                fast_rows += int(fast_idx.size)
                masked_rows += int(masked_idx.size)

    if block.sparse_rows.size:
        sample_idxs = block.sparse_sample_idxs
        values = block.sparse_values
        rows = block.sparse_rows
        if not use_fast:
            r[rows], n[rows] = sparse_masked(target, sample_idxs, values)
            masked_rows += len(values)
        else:
            row_ok = _sparse_complete_on_keep(sample_idxs, values, keep)
            if row_ok.all():
                r[rows], n[rows] = pearson_pairwise_sparse_complete(
                    target, sample_idxs, values, sample_mask=keep
                )
                fast_rows += len(values)
            elif not row_ok.any():
                r[rows], n[rows] = sparse_masked(target, sample_idxs, values)
                masked_rows += len(values)
            else:
                fast_idx = np.flatnonzero(row_ok)
                masked_idx = np.flatnonzero(~row_ok)
                r[rows[fast_idx]], n[rows[fast_idx]] = (
                    pearson_pairwise_sparse_complete(
                        target,
                        [sample_idxs[k] for k in fast_idx],
                        [values[k] for k in fast_idx],
                        sample_mask=keep,
                    )
                )
                r[rows[masked_idx]], n[rows[masked_idx]] = sparse_masked(
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
    "pearson_pairwise_masked",
    "pearson_pairwise_sparse",
    "pearson_pairwise_sparse_complete",
    "pearson_pairwise_sparse_masked",
]
