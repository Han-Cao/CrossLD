"""Linkage-disequilibrium math: pairwise unphased r / r² between variants.

Vectors are additive genotype or dosage values on the shared sample axis.
Genotypes are int8 hardcalls of 0, 1, 2, and dosages are float64 in [0, 2];
missing calls are encoded as ``-9`` (never ``NaN``). Correlations are computed
over the samples that are non-missing on *both* sides of a pair.

A block is a matrix with one variant per row and one sample per column. Blocks
may be dense (``np.ndarray``) or sparse (see :mod:`crossld.blocks`): a sparse
row lists the sample indexes whose value is nonzero and the values themselves;
samples absent from ``sample_idxs`` are exactly 0 (homozygous REF) and are valid
observations.

Two equivalent formulations are provided:

* the **raw-moment** method (``raw_moment_ld_*``) evaluates
  ``n·Σxy − Σx·Σy`` without subtracting the mean, which is fast but loses
  precision when the two terms nearly cancel (nearly constant rows);
* the **centered-moment** method (``centered_moment_ld_*``) subtracts the
  pairwise mean first and is used as the fallback for the raw method.

Both methods take optional per-side **valid** masks (``True`` = usable
observation) so missingness is inspected only once. When no mask is supplied,
the ``compute_mask`` flag decides what ``None`` means: with ``compute_mask=False``
a ``None`` mask means "nothing is missing", while with ``compute_mask=True`` a
``None`` mask is derived from the input (``value >= 0``).

Before any moment math a **sanity check** runs per pair: fewer than three
non-missing samples, or a target / block row that is constant over the
pairwise-valid samples, yields ``NaN`` directly. Any **suspicious** raw result
(raw variance ≤ 0, non-finite raw moment, ``|r| > 1``, or a cancellation
heuristic) is recomputed with the centered method, and is only ``NaN`` if the
centered result is degenerate as well.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from crossld.blocks import BlockMatrix

# Pair-level skip reasons, reported through logging.
SKIP_MISSING = "n<3"
SKIP_LOW_VARIANCE = "low_variance"

_MIN_SAMPLES = 3

# The raw-moment variance ``n*sum(x**2) - sum(x)**2`` subtracts two nearly equal
# terms for nearly constant (e.g. imputed fractional dosage) rows, so the result
# can carry a large relative error or even round to <= 0. Such pairs are
# recomputed with the centered formulation.
_RAW_MOMENT_MAX_REL_ERROR = 1e-6
_RAW_MOMENT_LIMIT = float(np.finfo(np.float64).eps) / _RAW_MOMENT_MAX_REL_ERROR
_OUT_OF_RANGE_TOL = 1e-9


# --------------------------------------------------------------------------- #
# allele-frequency helpers
# --------------------------------------------------------------------------- #

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
    missing = int((~(values >= 0)).sum())
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


# --------------------------------------------------------------------------- #
# mask resolution
# --------------------------------------------------------------------------- #

def _resolve_target_mask(
    target: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    compute_mask: bool,
) -> npt.NDArray[np.bool_] | None:
    """Resolve the target valid mask (``None`` = every sample valid)."""

    if target_valid is not None:
        return np.asarray(target_valid, dtype=bool)
    if compute_mask:
        return np.asarray(target) >= 0
    return None


def _resolve_dense_block_mask(
    block: npt.NDArray,
    block_valid: npt.NDArray[np.bool_] | None,
    compute_mask: bool,
) -> npt.NDArray[np.bool_] | None:
    """Resolve a dense block valid mask of shape ``(rows, samples)``."""

    if block_valid is not None:
        return np.asarray(block_valid, dtype=bool)
    if compute_mask:
        return np.asarray(block) >= 0
    return None


def _resolve_sparse_value_masks(
    values: list[npt.NDArray],
    value_valid: list[npt.NDArray[np.bool_]] | None,
    compute_mask: bool,
) -> list[npt.NDArray[np.bool_]] | None:
    """Resolve the per-row sparse validity masks aligned with ``values``."""

    if value_valid is not None:
        return [np.asarray(mask, dtype=bool) for mask in value_valid]
    if compute_mask:
        return [np.asarray(row) >= 0 for row in values]
    return None


def _sparse_value_masks(values: list[npt.NDArray]) -> list[npt.NDArray[np.bool_]] | None:
    """Per-row validity masks for sparse values, or ``None`` when all valid."""

    masks = [np.asarray(row) >= 0 for row in values]
    if all(bool(mask.all()) for mask in masks):
        return None
    return masks


# --------------------------------------------------------------------------- #
# cancellation heuristics
# --------------------------------------------------------------------------- #

def _raw_moment_unstable(
    var: npt.NDArray[np.float64],
    n: npt.NDArray[np.float64] | float,
    sum_sq: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.bool_]:
    """Heuristic flag for a raw-moment variance that cancellation may dominate.

    ``_RAW_MOMENT_LIMIT`` is a heuristic threshold (``eps / 1e-6``), not a
    guaranteed error bound: it does not account for accumulation error, for
    subtraction of dropped samples (see :func:`_subtraction_unstable`), or for
    covariance cancellation. ``sum_sq`` is the second-moment sum over the same
    valid set as ``var``, so ``n*sum_sq`` bounds the magnitudes being subtracted.
    """

    term = np.asarray(n, dtype=np.float64) * np.asarray(sum_sq, dtype=np.float64)
    limit = _RAW_MOMENT_LIMIT * term
    with np.errstate(invalid="ignore"):
        return (term > 0.0) & (np.asarray(var) <= limit)


def _subtraction_unstable(
    kept: npt.NDArray[np.float64] | float,
    total: npt.NDArray[np.float64] | float,
    dropped: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.bool_]:
    """Heuristic flag for ``kept == total - dropped`` losing the kept moment.

    The retained sum is compared against the magnitude of the operands that were
    subtracted, so an inaccurate but *nonzero* result is still flagged; testing
    ``kept != 0`` would not be enough.
    """

    total = np.asarray(total, dtype=np.float64)
    dropped = np.asarray(dropped, dtype=np.float64)
    kept = np.asarray(kept, dtype=np.float64)
    scale = np.maximum(np.abs(total), np.abs(dropped))
    with np.errstate(invalid="ignore"):
        return (scale > 0.0) & (np.abs(kept) <= _RAW_MOMENT_LIMIT * scale)


def _out_of_range(r: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Finite correlations materially outside ``[-1, 1]`` (rounding artefacts)."""

    with np.errstate(invalid="ignore"):
        return np.isfinite(r) & (np.abs(r) > 1.0 + _OUT_OF_RANGE_TOL)


# --------------------------------------------------------------------------- #
# sanity checks (run before any moment math)
# --------------------------------------------------------------------------- #

def _dense_pairwise_constant(
    target: npt.NDArray[np.float64],
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    both_valid: npt.NDArray[np.bool_] | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Per-row constancy of target and block over the pairwise-valid samples.
    
       Return (per-row target constancy, per-row block constancy).
    """

    n_rows = block.shape[0]
    t_const = np.zeros(n_rows, dtype=bool)
    b_const = np.zeros(n_rows, dtype=bool)

    if target_valid is None:
        t_values = target
    else:
        t_values = target[target_valid]
    # if target is const regardless block missingness
    t_global_const = bool(t_values.size == 0 or t_values.min() == t_values.max())

    if both_valid is None:
        # No missing anywhere, so the target's constancy is already global.
        t_const[:] = t_global_const
        b_const[:] = block.min(axis=1) == block.max(axis=1)
        return t_const, b_const

    if t_global_const:
        t_const[:] = True
    else:
        t_mins = np.where(both_valid, target[None, :], np.inf).min(axis=1)
        t_maxs = np.where(both_valid, target[None, :], -np.inf).max(axis=1)
        t_const[:] = t_mins == t_maxs

    b_mins = np.where(both_valid, block, np.inf).min(axis=1)
    b_maxs = np.where(both_valid, block, -np.inf).max(axis=1)
    b_const[:] = b_mins == b_maxs
    return t_const, b_const


def _small_sample_dense(
    target: npt.NDArray[np.float64],
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    block_valid: npt.NDArray[np.bool_] | None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.bool_]]:
    """Pairwise sample count and per-row ``NaN`` decision for a dense block.
       
       Return (per-row non-missing sample count, per-row < _MIN_SAMPLES).
    """

    n_rows = block.shape[0]
    n_keep = int(target_valid.sum() if target_valid is not None else target.size)

    if block_valid is None:
        n = np.full(n_rows, n_keep, dtype=np.uint32)
    elif target_valid is None:
        n = block_valid.sum(axis=1)
    else:
        # if both have missingness, the do dot product
        n = target_valid.astype(np.uint32) @ block_valid.T.astype(np.uint32)
    bad = n < _MIN_SAMPLES

    return n, bad


def _sanity_sparse(
    target: npt.NDArray[np.float64],
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None,
    value_valid: list[npt.NDArray[np.bool_]] | None,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.bool_]]:
    """Pairwise sample count and per-row ``NaN`` decision for sparse rows."""

    n_rows = len(sample_idxs)
    if target_valid is None:
        n_keep = int(target.size)
        t_global_const = bool(target.size == 0 or target.min() == target.max())
    else:
        n_keep = int(target_valid.sum())
        t_seen = target[target_valid]
        t_global_const = bool(t_seen.size == 0 or t_seen.min() == t_seen.max())

    # np.empty is safe here since every row are processed in loop without continue
    n_out = np.empty(n_rows, dtype=np.uint32)
    t_const = np.empty(n_rows, dtype=bool)
    b_const = np.empty(n_rows, dtype=bool)

    for row in range(n_rows):
        idx = np.asarray(sample_idxs[row])
        vals = np.asarray(values[row], dtype=np.float64)
        mask = None if value_valid is None else np.asarray(value_valid[row], dtype=bool)

        if target_valid is None:
            in_target = np.ones(idx.size, dtype=bool)
        else:
            in_target = target_valid[idx]

        if mask is None:
            dropped_mask = np.zeros(idx.size, dtype=bool)
        else:
            dropped_mask = in_target & ~mask
        dropped = int(dropped_mask.sum())
        n = n_keep - dropped
        n_out[row] = n

        # Block-row values over the pairwise-valid set are the listed valid
        # values plus one implicit 0 per unlisted valid sample.
        listed_valid = vals[in_target & (mask if mask is not None else True)]
        if listed_valid.size == 0:
            b_const[row] = True
        elif n > listed_valid.size:
            b_const[row] = bool(listed_valid.min() == 0.0 and listed_valid.max() == 0.0)
        else:
            b_const[row] = bool(listed_valid.min() == listed_valid.max())

        if t_global_const or dropped == 0:
            t_const[row] = t_global_const
        else:
            observed = (
                np.ones(target.size, dtype=bool)
                if target_valid is None
                else target_valid.copy()
            )
            observed[idx[dropped_mask]] = False
            seen = target[observed]
            t_const[row] = bool(seen.size == 0 or seen.min() == seen.max())

    bad = (n_out < _MIN_SAMPLES) | t_const | b_const
    return n_out, bad


# --------------------------------------------------------------------------- #
# centered-moment kernels
# --------------------------------------------------------------------------- #

def _centered_dense(
    target: npt.NDArray[np.float64],
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    block_valid: npt.NDArray[np.bool_] | None,
    n: npt.NDArray[np.uint32],
    bad: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Centered pairwise r for a dense block."""

    r = np.full(block.shape[0], np.nan, dtype=np.float64)
    if block.shape[0] == 0:
        return r

    if block_valid is None:
        valid = None if target_valid is None else target_valid[None, :]
    else:
        valid = block_valid
        if target_valid is not None:
            valid = valid & target_valid[None, :]

    t_const, b_const = _dense_pairwise_constant(target, block, target_valid, valid)

    count = n.astype(np.float64)
    if valid is None:
        a = np.broadcast_to(target[None, :], block.shape)
        b = block
    else:
        a = np.where(valid, target[None, :], 0.0)
        b = np.where(valid, block, 0.0)

    denom = np.maximum(count, 1.0)
    da = a - (a.sum(axis=1) / denom)[:, None]
    db = b - (b.sum(axis=1) / denom)[:, None]
    if valid is not None:
        da = np.where(valid, da, 0.0)
        db = np.where(valid, db, 0.0)

    var_a = (da * da).sum(axis=1)
    var_b = (db * db).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # sqrt(var_a) * sqrt(var_b): the product of two representable variances
        # can underflow even when the correlation is fine.
        r = (da * db).sum(axis=1) / (np.sqrt(var_a) * np.sqrt(var_b))

    good = ~(bad | t_const | b_const | ~np.isfinite(r))
    return np.where(good, r, np.nan)


def centered_moment_ld_dense(
    target: npt.NDArray,
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None = None,
    block_valid: npt.NDArray[np.bool_] | None = None,
    *,
    compute_mask: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Centered pairwise r between ``target`` and every row of a dense ``block``.

    Returns ``(r, n)``. ``n`` is the per-pair non-missing sample count and
    ``r`` is ``NaN`` for pairs with fewer than three usable samples or a
    constant target / block row over the pairwise-valid samples.
    """

    t = np.asarray(target, dtype=np.float64)
    block = np.atleast_2d(block)
    tv = _resolve_target_mask(target, target_valid, compute_mask)
    bv = _resolve_dense_block_mask(block, block_valid, compute_mask)
    n, bad = _small_sample_dense(t, block, tv, bv)
    r = _centered_dense(t, block, tv, bv, n, bad)
    return np.clip(r, -1.0, 1.0), n


def _centered_sparse(
    target: npt.NDArray[np.float64],
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None,
    value_valid: list[npt.NDArray[np.bool_]] | None,
    n: npt.NDArray[np.uint32],
    bad: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Centered pairwise r for sparse rows (densified as one block)."""

    n_rows = len(sample_idxs)
    if n_rows == 0:
        return np.full(0, np.nan, dtype=np.float64)

    sample_ct = int(target.size)
    dense = np.zeros((n_rows, sample_ct), dtype=np.float64)
    dense_valid = np.ones((n_rows, sample_ct), dtype=bool)
    for row in range(n_rows):
        idx = np.asarray(sample_idxs[row])
        dense[row, idx] = np.asarray(values[row], dtype=np.float64)
        if value_valid is not None:
            dense_valid[row, idx] = np.asarray(value_valid[row], dtype=bool)
    return _centered_dense(target, dense, target_valid, dense_valid, n, bad)


def centered_moment_ld_sparse(
    target: npt.NDArray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None = None,
    value_valid: list[npt.NDArray[np.bool_]] | None = None,
    *,
    compute_mask: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Centered pairwise r between ``target`` and sparse block rows.

    ``sample_idxs`` and ``values`` hold one sparse array per row; unlisted
    samples are exactly 0 and valid. Same ``(r, n)`` contract as
    :func:`centered_moment_ld_dense`.
    """

    t = np.asarray(target, dtype=np.float64)
    tv = _resolve_target_mask(target, target_valid, compute_mask)
    vv = _resolve_sparse_value_masks(values, value_valid, compute_mask)
    n, bad = _sanity_sparse(t, sample_idxs, values, tv, vv)
    r = _centered_sparse(t, sample_idxs, values, tv, vv, n, bad)
    return np.clip(r, -1.0, 1.0), n


# --------------------------------------------------------------------------- #
# raw-moment kernels
# --------------------------------------------------------------------------- #

def _raw_moment_result(
    n: npt.NDArray[np.uint32],
    bad: npt.NDArray[np.bool_],
    sum_a0: float,
    sum_a20: float,
    sum_a: npt.NDArray[np.float64],
    sum_a2: npt.NDArray[np.float64],
    sum_b: npt.NDArray[np.float64],
    sum_b2: npt.NDArray[np.float64],
    sum_ab: npt.NDArray[np.float64],
    bad_sum: npt.NDArray[np.float64],
    bad_sum2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Raw-moment r plus the non-``bad`` rows that need a centered retry.

    ``a`` is the target side and ``b`` the block side. ``sum_a0`` / ``sum_a20``
    are the target moments over every valid sample and ``bad_sum`` / ``bad_sum2``
    the part of them that the block's missing cells remove.
    """

    nf = n.astype(np.float64)
    numerator = nf * sum_ab - sum_a * sum_b
    var_a = nf * sum_a2 - sum_a * sum_a
    var_b = nf * sum_b2 - sum_b * sum_b

    with np.errstate(invalid="ignore", divide="ignore"):
        # sqrt(var_a) * sqrt(var_b): the product of two representable variances
        # can underflow even when the correlation is fine.
        r = numerator / (np.sqrt(var_a) * np.sqrt(var_b))

    finite = (
        np.isfinite(numerator)
        & np.isfinite(var_a)
        & np.isfinite(var_b)
        & np.isfinite(r)
    )
    suspicious = (
        (var_a <= 0.0)
        | (var_b <= 0.0)
        | ~finite
        | _raw_moment_unstable(var_a, nf, sum_a2)
        | _raw_moment_unstable(var_b, nf, sum_b2)
        | _subtraction_unstable(sum_a, sum_a0, bad_sum)
        | _subtraction_unstable(sum_a2, sum_a20, bad_sum2)
        | _out_of_range(r)
    )
    retry = ~bad & suspicious
    r = np.where(retry, np.nan, r)
    r = np.where(bad, np.nan, r)
    return r, retry


def _raw_dense_block_sums(
    target_v: npt.NDArray[np.float64],
    block: npt.NDArray,
    weight: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Raw-moment sums for a target (valid samples only) X dense block.

       Return (sum_b, sum_b2, sum_tb)
       
       target_v = target[target_valid]
       Weight = np.asarray(target_valid, dtype=np.float64), this is used
       to filter block columns to match the target's valid samples.
       Weight is None means all samples in origain target are valid.
    """

    if weight is None:
        sum_b = block.sum(axis=1, dtype=np.float64)
        sum_b2 = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    else:
        sum_b = np.einsum("ij,j->i", block, weight, dtype=np.float64)
        sum_b2 = np.einsum("ij,ij,j->i", block, block, weight, dtype=np.float64)
    sum_tb = np.einsum("ij,j->i", block, target_v, dtype=np.float64)

    return sum_b, sum_b2, sum_tb


def _raw_dense_sums(
    target: npt.NDArray[np.float64],
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    block_valid: npt.NDArray[np.bool_] | None,
    n: npt.NDArray[np.uint32],
    bad: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Raw-moment r for a dense block, plus the rows that need a retry.

    Returns ``(r, retry)`` where ``r`` is ``NaN`` for ``bad`` rows and for every
    suspicious row. ``retry`` excludes ``bad`` rows, which are permanently
    ``NaN``.
    """

    n_rows = block.shape[0]
    r = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows == 0:
        return r, np.zeros(0, dtype=bool)

    # target_valid/weight is None -> all valid
    # weight[j] == 0 means remove block[,j] from the sum
    if target_valid is None:
        t_values = target
        weight = None
        sum_t0 = float(target.sum())
        sum_t20 = float(target @ target)
    else:
        t_values = np.where(target_valid, target, 0.0)
        # convert to float to calculate r
        weight = target_valid.astype(np.float64)
        sum_t0 = float(t_values.sum())
        sum_t20 = float(t_values @ t_values)

    # sum_t0 / sum_t20 are the target moments over every valid target sample
    # ban_sum / bad_sum2 for block's missing samples to be removed later
    sum_t = np.full(n_rows, sum_t0, dtype=np.float64)
    sum_t2 = np.full(n_rows, sum_t20, dtype=np.float64)
    bad_sum = np.zeros(n_rows, dtype=np.float64)
    bad_sum2 = np.zeros(n_rows, dtype=np.float64)

    if block_valid is None:
        if weight is None:
            sum_b = block.sum(axis=1, dtype=np.float64)
            sum_b2 = np.einsum("ij,ij->i", block, block, dtype=np.float64)
        else:
            sum_b = np.einsum("ij,j->i", block, weight, dtype=np.float64)
            sum_b2 = np.einsum("ij,ij,j->i", block, block, weight, dtype=np.float64)
        sum_tb = np.einsum("ij,j->i", block, t_values, dtype=np.float64)
    else:
        # Missing cells inside the target's valid region leave the pair. Cells
        # outside it are irrelevant (weight 0), so the "complete" rows can be
        # summed directly while incomplete rows are zeroed first.
        inv_rows, inv_cols = np.nonzero(~block_valid)
        if target_valid is not None and inv_rows.size:
            inside = target_valid[inv_cols]
            inv_rows = inv_rows[inside]
            inv_cols = inv_cols[inside]

        if inv_rows.size:
            complete = np.bincount(inv_rows, minlength=n_rows) == 0
        else:
            complete = np.ones(n_rows, dtype=bool)

        sum_b = np.zeros(n_rows, dtype=np.float64)
        sum_b2 = np.zeros(n_rows, dtype=np.float64)
        sum_tb = np.zeros(n_rows, dtype=np.float64)

        if complete.any():
            subset = block[complete]
            if weight is None:
                sum_b[complete] = subset.sum(axis=1, dtype=np.float64)
                sum_b2[complete] = np.einsum(
                    "ij,ij->i", subset, subset, dtype=np.float64
                )
            else:
                sum_b[complete] = np.einsum(
                    "ij,j->i", subset, weight, dtype=np.float64
                )
                sum_b2[complete] = np.einsum(
                    "ij,ij,j->i", subset, subset, weight, dtype=np.float64
                )
            sum_tb[complete] = np.einsum(
                "ij,j->i", subset, t_values, dtype=np.float64
            )

        incomplete = ~complete
        if incomplete.any():
            subset = block[incomplete]
            zeroed = np.where(subset >= 0, subset, 0.0)
            if weight is None:
                sum_b[incomplete] = zeroed.sum(axis=1, dtype=np.float64)
                sum_b2[incomplete] = np.einsum(
                    "ij,ij->i", zeroed, zeroed, dtype=np.float64
                )
            else:
                sum_b[incomplete] = np.einsum(
                    "ij,j->i", zeroed, weight, dtype=np.float64
                )
                sum_b2[incomplete] = np.einsum(
                    "ij,ij,j->i", zeroed, zeroed, weight, dtype=np.float64
                )
            sum_tb[incomplete] = np.einsum(
                "ij,j->i", zeroed, t_values, dtype=np.float64
            )
            dropped_t = target[inv_cols]
            bad_sum = np.bincount(inv_rows, weights=dropped_t, minlength=n_rows)
            bad_sum2 = np.bincount(
                inv_rows, weights=dropped_t * dropped_t, minlength=n_rows
            )
            sum_t = sum_t - bad_sum
            sum_t2 = sum_t2 - bad_sum2

    return _raw_moment_result(
        n, bad, sum_t0, sum_t20, sum_t, sum_t2, sum_b, sum_b2, sum_tb, bad_sum, bad_sum2
    )


def raw_moment_ld_dense(
    target: npt.NDArray,
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None = None,
    block_valid: npt.NDArray[np.bool_] | None = None,
    *,
    compute_mask: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Raw-moment pairwise r between ``target`` and every row of a dense ``block``.

    Same ``(r, n)`` contract as :func:`centered_moment_ld_dense`, except that
    ``r`` is ``NaN`` for every row where the raw-moment identity is unreliable
    (variance ≤ 0, non-finite moment, ``|r| > 1``, or a cancellation heuristic).
    Such rows can be recovered with the centered method on the same masks.
    """

    t = np.asarray(target, dtype=np.float64)
    block = np.atleast_2d(block)
    tv = _resolve_target_mask(target, target_valid, compute_mask)
    bv = _resolve_dense_block_mask(block, block_valid, compute_mask)
    n, bad = _small_sample_dense(t, block, tv, bv)
    r, _ = _raw_dense_sums(t, block, tv, bv, n, bad)
    return np.clip(r, -1.0, 1.0), n


def _raw_sparse_sums(
    target: npt.NDArray[np.float64],
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None,
    value_valid: list[npt.NDArray[np.bool_]] | None,
    n: npt.NDArray[np.uint32],
    bad: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Raw-moment r for sparse rows, plus the rows that need a retry."""

    n_rows = len(sample_idxs)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows == 0:
        return r, np.zeros(0, dtype=bool)

    if target_valid is None:
        weighted = target
        sum_t0 = float(target.sum())
        sum_t20 = float(target @ target)
    else:
        weighted = np.where(target_valid, target, 0.0)
        sum_t0 = float(weighted.sum())
        sum_t20 = float(weighted @ weighted)

    lengths = np.fromiter(
        (int(np.asarray(idx).size) for idx in sample_idxs),
        dtype=np.uint32,
        count=n_rows,
    )
    row_ids = np.repeat(np.arange(n_rows, dtype=np.uint32), lengths)
    if row_ids.size:
        v_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
        i_all = np.concatenate(
            [np.asarray(idx, dtype=np.uint32) for idx in sample_idxs]
        )
        if value_valid is None:
            listed_valid = np.ones(row_ids.size, dtype=bool)
        else:
            listed_valid = np.concatenate(
                [np.asarray(mask, dtype=bool) for mask in value_valid]
            )
    else:
        v_all = np.empty(0, dtype=np.float64)
        i_all = np.empty(0, dtype=np.uint32)
        listed_valid = np.empty(0, dtype=bool)

    if target_valid is None:
        in_target = np.ones(row_ids.size, dtype=bool)
    else:
        in_target = target_valid[i_all]

    sum_t = np.full(n_rows, sum_t0, dtype=np.float64)
    sum_t2 = np.full(n_rows, sum_t20, dtype=np.float64)
    bad_sum = np.zeros(n_rows, dtype=np.float64)
    bad_sum2 = np.zeros(n_rows, dtype=np.float64)
    sum_x = np.zeros(n_rows, dtype=np.float64)
    sum_x2 = np.zeros(n_rows, dtype=np.float64)
    sum_xt = np.zeros(n_rows, dtype=np.float64)

    if row_ids.size:
        both = in_target & listed_valid
        if both.any():
            ids = row_ids[both]
            vals = v_all[both]
            sum_x = np.bincount(ids, weights=vals, minlength=n_rows)
            sum_x2 = np.bincount(ids, weights=vals * vals, minlength=n_rows)
            sum_xt = np.bincount(
                ids, weights=vals * weighted[i_all[both]], minlength=n_rows
            )

        dropped = in_target & ~listed_valid
        if dropped.any():
            ids = row_ids[dropped]
            dropped_t = target[i_all[dropped]]
            bad_sum = np.bincount(ids, weights=dropped_t, minlength=n_rows)
            bad_sum2 = np.bincount(
                ids, weights=dropped_t * dropped_t, minlength=n_rows
            )
            sum_t = sum_t - bad_sum
            sum_t2 = sum_t2 - bad_sum2

    return _raw_moment_result(
        n, bad, sum_t0, sum_t20, sum_t, sum_t2, sum_x, sum_x2, sum_xt, bad_sum, bad_sum2
    )


def raw_moment_ld_sparse(
    target: npt.NDArray,
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None = None,
    value_valid: list[npt.NDArray[np.bool_]] | None = None,
    *,
    compute_mask: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Raw-moment pairwise r between ``target`` and sparse block rows.

    Same ``(r, n)`` contract as :func:`centered_moment_ld_sparse`, with the
    unreliable-row rule of :func:`raw_moment_ld_dense`.
    """

    t = np.asarray(target, dtype=np.float64)
    tv = _resolve_target_mask(target, target_valid, compute_mask)
    vv = _resolve_sparse_value_masks(values, value_valid, compute_mask)
    n, bad = _sanity_sparse(t, sample_idxs, values, tv, vv)
    r, _ = _raw_sparse_sums(t, sample_idxs, values, tv, vv, n, bad)
    return np.clip(r, -1.0, 1.0), n


# --------------------------------------------------------------------------- #
# raw-with-centered-fallback orchestration
# --------------------------------------------------------------------------- #

def _ld_dense(
    target: npt.NDArray[np.float64],
    block: npt.NDArray,
    target_valid: npt.NDArray[np.bool_] | None,
    block_valid: npt.NDArray[np.bool_] | None,
    force_centered: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32], int]:
    """Raw-moment r for a dense block, retrying suspicious rows centered.

    Returns ``(r, n, centered_rows)`` where ``centered_rows`` is how many rows
    were evaluated with the centered method.
    """

    n_rows = int(block.shape[0])
    n, bad = _small_sample_dense(target, block, target_valid, block_valid)
    if force_centered:
        r = _centered_dense(target, block, target_valid, block_valid, n, bad)
        return np.clip(r, -1.0, 1.0), n, n_rows

    r, retry = _raw_dense_sums(target, block, target_valid, block_valid, n, bad)
    centered_rows = int(retry.sum())
    if centered_rows:
        rows = np.flatnonzero(retry)
        sub_valid = None if block_valid is None else block_valid[rows]
        r[rows] = _centered_dense(
            target, block[rows], target_valid, sub_valid, n[rows], bad[rows]
        )
    return np.clip(r, -1.0, 1.0), n, centered_rows


def _ld_sparse(
    target: npt.NDArray[np.float64],
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    target_valid: npt.NDArray[np.bool_] | None,
    value_valid: list[npt.NDArray[np.bool_]] | None,
    force_centered: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32], int]:
    """Raw-moment r for sparse rows, retrying suspicious rows centered.

    Returns ``(r, n, centered_rows)`` where ``centered_rows`` is how many rows
    were evaluated with the centered method.
    """

    n_rows = len(sample_idxs)
    n, bad = _sanity_sparse(target, sample_idxs, values, target_valid, value_valid)
    if force_centered:
        r = _centered_sparse(
            target, sample_idxs, values, target_valid, value_valid, n, bad
        )
        return np.clip(r, -1.0, 1.0), n, n_rows

    r, retry = _raw_sparse_sums(
        target, sample_idxs, values, target_valid, value_valid, n, bad
    )
    centered_rows = int(retry.sum())
    if centered_rows:
        rows = np.flatnonzero(retry)
        sub_idxs = [sample_idxs[row] for row in rows]
        sub_values = [values[row] for row in rows]
        sub_valid = None if value_valid is None else [value_valid[row] for row in rows]
        r[rows] = _centered_sparse(
            target, sub_idxs, sub_values, target_valid, sub_valid, n[rows], bad[rows]
        )
    return np.clip(r, -1.0, 1.0), n, centered_rows


# --------------------------------------------------------------------------- #
# block dispatcher
# --------------------------------------------------------------------------- #

def _dense_complete_rows(
    dense: npt.NDArray,
    block_valid: npt.NDArray[np.bool_] | None,
    target_valid: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Rows with a valid call on every sample selected by ``target_valid``."""

    n_rows = int(dense.shape[0])
    if block_valid is None:
        return np.ones(n_rows, dtype=bool)
    inv_rows, inv_cols = np.nonzero(~block_valid)
    if target_valid is not None and inv_rows.size:
        inv_rows = inv_rows[target_valid[inv_cols]]
    if inv_rows.size == 0:
        return np.ones(n_rows, dtype=bool)
    return np.bincount(inv_rows, minlength=n_rows) == 0


def _sparse_complete_rows(
    sample_idxs: list[npt.NDArray[np.uint32]],
    values: list[npt.NDArray],
    value_valid: list[npt.NDArray[np.bool_]] | None,
    target_valid: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Sparse rows with no missing call on the target's valid samples."""

    n_rows = len(sample_idxs)
    out = np.ones(n_rows, dtype=bool)
    if value_valid is None:
        return out
    for row in range(n_rows):
        mask = np.asarray(value_valid[row], dtype=bool)
        if mask.all():
            continue
        idx = np.asarray(sample_idxs[row])
        if target_valid is None or bool((target_valid[idx] & ~mask).any()):
            out[row] = False
    return out


def pearson_pairwise_block(
    target: npt.NDArray,
    block: BlockMatrix,
    *,
    force_centered: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32]]:
    """Pearson r between ``target`` and a :class:`crossld.blocks.BlockMatrix`.

    Returns ``(r, n)`` aligned with the block's requested row order (dense and
    sparse rows interleaved as they were requested).

    Missingness is inspected once, per side, and the resulting valid masks are
    reused by the raw attempt and any centered fallback. Rows complete on the
    target's valid samples take the vectorized raw path; the remaining rows
    take the pairwise-masked path. Pass ``force_centered=True`` to skip the
    raw-moment method entirely and evaluate every row with centered moments.

    The block's :class:`crossld.blocks.BlockStats` pair counters are updated:
    ``complete_pair`` / ``incomplete_pair`` split the pairs by whether the block
    row is valid on every sample the target observes, and ``raw_moment_pair`` /
    ``centered_moment_pair`` split them by which LD method produced the result.
    """

    n_rows = int(block.stats.variants)
    r = np.full(n_rows, np.nan, dtype=np.float64)
    n = np.zeros(n_rows, dtype=np.uint32)

    target = np.asarray(target)
    target_checked = target >= 0
    target_valid = None if bool(target_checked.all()) else target_checked

    # check if target_valid < _MIN_SAMPLES
    n_keep = int(target_checked.sum())
    if n_keep < _MIN_SAMPLES:
        block.stats.skip_missing += n_rows
        return r, np.full(n_rows, n_keep, dtype=np.uint32)

    complete_pair = 0
    incomplete_pair = 0
    raw_moment_pair = 0
    centered_moment_pair = 0

    if block.dense_rows.size:
        dense = block.dense
        rows = block.dense_rows
        block_checked = dense >= 0
        block_valid = None if bool(block_checked.all()) else block_checked
        r[rows], n[rows], centered = _ld_dense(
            target.astype(np.float64), dense, target_valid, block_valid, force_centered
        )
        # LD pair summary
        row_count = len(rows)
        n_miss = n_keep - n[rows]
        assert np.all(n_miss >= 0)
        complete = n_miss == 0
        complete_count = int(complete.sum())
        complete_pair += complete_count
        incomplete_pair += row_count - complete_count
        centered_moment_pair += centered
        raw_moment_pair += row_count - centered

    if block.sparse_rows.size:
        sample_idxs = block.sparse_sample_idxs
        values = block.sparse_values
        rows = block.sparse_rows
        value_valid = _sparse_value_masks(values)
        complete = _sparse_complete_rows(
            sample_idxs, values, value_valid, target_valid
        )
        row_count = len(values)
        complete_count = int(complete.sum())
        complete_pair += complete_count
        incomplete_pair += row_count - complete_count
        r[rows], n[rows], centered = _ld_sparse(
            target.astype(np.float64),
            sample_idxs,
            values,
            target_valid,
            value_valid,
            force_centered,
        )
        centered_moment_pair += centered
        raw_moment_pair += row_count - centered

    missing_pair = int(np.sum(n < _MIN_SAMPLES))
    low_variance_pair = int(np.sum(np.isnan(r))) - missing_pair
    block.stats.complete_pair += complete_pair
    block.stats.incomplete_pair += incomplete_pair
    block.stats.raw_moment_pair += raw_moment_pair
    block.stats.centered_moment_pair += centered_moment_pair
    block.stats.skip_missing += missing_pair
    block.stats.skip_low_variance += low_variance_pair
    return r, n


__all__ = [
    "SKIP_MISSING",
    "SKIP_LOW_VARIANCE",
    "alt_freq",
    "alt_freq_sparse",
    "centered_moment_ld_dense",
    "centered_moment_ld_sparse",
    "major_allele_flipped",
    "major_allele_flipped_sparse",
    "pearson_pairwise_block",
    "raw_moment_ld_dense",
    "raw_moment_ld_sparse",
]
