"""Block-wise genotype/dosage reading through :mod:`pgenlib`.

Reads dataset variants in blocks, using sparse representations when the underlying
``.pgen`` data is compatibly sparse and falling back to dense reads otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from crossld.logging_setup import LOGGER_NAME
from crossld.plinkio import PlinkDataset

logger = logging.getLogger(f"{LOGGER_NAME}.blocks")


@dataclass(slots=True)
class BlockStats:
    """Per-block statistics emitted at DEBUG level.

    The pair counters describe two independent partitions of the block's
    evaluated variant pairs:

    * ``complete_pair`` / ``incomplete_pair`` -- whether the block row has a
      valid call on every sample the target observes (complete) or not
      (incomplete).
    * ``raw_moment_pair`` / ``centered_moment_pair`` -- which LD method produced
      the result: the raw-moment method, or the centered-moment method used as
      the precision fallback (or for every pair under ``force_centered``).
    """

    block_index: int = 1
    block_total: int = 1
    variants: int = 0
    dense_variants: int = 0
    sparse_variants: int = 0
    complete_pair: int = 0
    incomplete_pair: int = 0
    raw_moment_pair: int = 0
    centered_moment_pair: int = 0
    # pairs_evaluated: int = 0
    kept: int = 0
    below_r2: int = 0
    # skipped: int = 0
    skip_missing: int = 0
    skip_low_variance: int = 0
    mode: str = "hardcall"


@dataclass
class BlockMatrix:
    """One requested block, split into dense and sparse rows.

    ``dense_rows``/``sparse_rows`` are 0-based row positions within the requested
    variant list, so results can be mapped back in the original order. Rows in
    ``dense`` follow the order of ``dense_rows``; ``sparse_sample_idxs`` and
    ``sparse_values`` are aligned with ``sparse_rows``. Sparse sample indexes are
    relative to the dataset's active sample subset; unlisted samples are exactly
    0 (homozygous REF), while listed values may be missing (``< 0``).
    """

    dense: npt.NDArray
    dense_rows: npt.NDArray[np.uint32]
    sparse_rows: npt.NDArray[np.uint32]
    sparse_sample_idxs: list[npt.NDArray[np.uint32]]
    sparse_values: list[npt.NDArray]
    stats: BlockStats


@dataclass(slots=True)
class _CachedRow:
    """One cached dataset-2 variant row.

    A row is either dense (``values`` is the 1-D genotype/dosage vector and
    ``sample_idxs`` is ``None``) or sparse (``sample_idxs``/``values`` are the
    listed sample indexes and values). Cached arrays are never mutated by the
    LD code, so they are safe to share across blocks.
    """

    values: npt.NDArray
    sample_idxs: npt.NDArray[np.uint32] | None = None


@dataclass
class BlockReader:
    """Reads variant blocks as int8 hardcalls or float64 dosages on the active subset.

    For sparse variants, the reader always read the alt alelle (pgenlib's default).

    Notes for precision:
    Based on tests, PLINK2's LD is estimated in float64, to replicate the same precision,
    we also use np.float64 for all dosages and downstream LD calaculation.
    
    TODO: consider make --float32 as a optional flag

    Rows decoded for a block can be kept for later reads through ``keep_next``
    (see :meth:`read`), bounded by ``cache_limit`` (``0`` disables caching).
    """

    dataset: PlinkDataset
    dosage: bool = False
    force_dense: bool = False
    cache_limit: int = 0
    cache_hits: int = field(default=0, init=False)
    _cache: dict[int, _CachedRow] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.reader = self.dataset.reader
        self.fmt = self.dataset.fmt
        if self.dosage:
            self.arr_dtype = np.float64
        else:
            self.arr_dtype = np.int8

    @property
    def sample_ct(self) -> int:
        """Samples the reader returns per variant (the active subset size)."""

        return self.dataset.active_sample_ct

    @property
    def cache_size(self) -> int:
        """Number of variant rows currently cached."""

        return len(self._cache)

    def cached_mask(self, indices: list[int] | npt.NDArray[np.uint32]) -> npt.NDArray[np.bool_]:
        """Boolean mask: which requested variant indexes are already cached."""

        idx_array = np.asarray(indices, dtype=np.uint32)
        cache = self._cache
        return np.array([int(idx) in cache for idx in idx_array], dtype=bool)

    def retain(self, keep) -> None:
        """Drop every cached row whose variant index is not in ``keep``."""

        keep_set = {int(idx) for idx in np.asarray(keep).tolist()}
        for key in [key for key in self._cache if key not in keep_set]:
            del self._cache[key]

    def read(
        self,
        indices: list[int] | npt.NDArray[np.uint32],
        keep_next: set[int] | None = None,
    ) -> BlockMatrix:
        """Read ``indices`` as one :class:`BlockMatrix` (dense and sparse rows).

        Rows already in the cache are served from memory; the remaining rows go
        through the regular pgenlib path. Passing ``keep_next`` keeps the rows
        listed there in the cache (up to :attr:`cache_limit`) and drops the
        consumed rows that are not listed, so the cache only ever holds rows a
        later call can reuse.
        """

        idx_array = np.asarray(indices, dtype=np.uint32)
        stats = BlockStats(
            variants=int(idx_array.size),
            mode="dosage" if self.dosage else "hardcall",
        )
        if idx_array.size == 0:
            return self._empty(stats)

        cached_mask = self.cached_mask(idx_array)
        self.cache_hits += int(cached_mask.sum())
        new_mask = ~cached_mask
        new_positions = np.flatnonzero(new_mask)

        # classify only the rows that still have to be read (dense vs sparse)
        new_idx = idx_array[new_positions]
        if new_idx.size and not (self.force_dense or self.fmt == "bfile"):
            idx_sparse = self._find_sparse(new_idx)
        else:
            idx_sparse = np.empty(0, dtype=np.uint32)
        if idx_sparse.size:
            new_sparse_mask = np.isin(new_idx, idx_sparse)
        else:
            new_sparse_mask = np.zeros(new_idx.size, dtype=bool)

        new_dense_positions = new_positions[~new_sparse_mask]
        new_sparse_positions = new_positions[new_sparse_mask]

        dense_mask = new_mask.copy()
        dense_mask[new_sparse_positions] = False
        sparse_mask = np.zeros(idx_array.size, dtype=bool)
        sparse_mask[new_sparse_positions] = True
        # cached rows take the dense/sparse kind they were stored with
        for pos in np.flatnonzero(cached_mask).tolist():
            if self._cache[int(idx_array[pos])].sample_idxs is None:
                dense_mask[pos] = True
            else:
                sparse_mask[pos] = True

        dense_rows = np.flatnonzero(dense_mask).astype(np.uint32)
        sparse_rows = np.flatnonzero(sparse_mask).astype(np.uint32)

        if new_dense_positions.size:
            new_dense = self._read_dense(idx_array[new_dense_positions])
        else:
            new_dense = np.empty((0, self.sample_ct), dtype=self.arr_dtype)

        if new_sparse_positions.size:
            new_sparse_idxs, new_sparse_values = self._read_sparse(
                idx_array[new_sparse_positions]
            )
        else:
            new_sparse_idxs, new_sparse_values = [], []

        # assemble dense rows in the exact requested order
        if dense_rows.size:
            dense = np.empty((dense_rows.size, self.sample_ct), dtype=self.arr_dtype)
            cached_in_dense = np.flatnonzero(cached_mask[dense_rows])
            if cached_in_dense.size:
                dense[cached_in_dense] = np.stack(
                    [
                        self._cache[int(idx_array[dense_rows[pos]])].values
                        for pos in cached_in_dense.tolist()
                    ]
                )
            new_in_dense = np.flatnonzero(new_mask[dense_rows])
            if new_in_dense.size:
                dense[new_in_dense] = new_dense
        else:
            dense = np.empty((0, self.sample_ct), dtype=self.arr_dtype)

        # assemble sparse rows in the exact requested order
        sparse_sample_idxs: list[npt.NDArray[np.uint32]] = []
        sparse_values: list[npt.NDArray] = []
        new_sparse_cursor = 0
        for pos in sparse_rows.tolist():
            if cached_mask[pos]:
                row = self._cache[int(idx_array[pos])]
                sparse_sample_idxs.append(row.sample_idxs)
                sparse_values.append(row.values)
            else:
                sparse_sample_idxs.append(new_sparse_idxs[new_sparse_cursor])
                sparse_values.append(new_sparse_values[new_sparse_cursor])
                new_sparse_cursor += 1

        self._update_cache(
            idx_array,
            cached_mask,
            new_dense_positions,
            new_dense,
            new_sparse_positions,
            new_sparse_idxs,
            new_sparse_values,
            keep_next,
        )

        stats.dense_variants = int(dense_rows.size)
        stats.sparse_variants = int(sparse_rows.size)

        return BlockMatrix(
            dense=dense,
            dense_rows=dense_rows,
            sparse_rows=sparse_rows,
            sparse_sample_idxs=sparse_sample_idxs,
            sparse_values=sparse_values,
            stats=stats,
        )

    def _update_cache(
        self,
        idx_array: npt.NDArray[np.uint32],
        cached_mask: npt.NDArray[np.bool_],
        new_dense_positions: npt.NDArray[np.intp],
        new_dense: npt.NDArray,
        new_sparse_positions: npt.NDArray[np.intp],
        new_sparse_idxs: list[npt.NDArray[np.uint32]],
        new_sparse_values: list[npt.NDArray],
        keep_next: set[int] | None,
    ) -> None:
        """Drop consumed rows not needed later, then retain newly read ones.

        Dropping happens first so the freed slots are available to the rows
        just read; the ``cache_limit`` invariant holds throughout.
        """

        keep = keep_next if keep_next is not None else set()

        # 1. consumed cached rows that no later call will reuse
        for pos in np.flatnonzero(cached_mask).tolist():
            key = int(idx_array[pos])
            if key not in keep:
                del self._cache[key]

        # 2. newly read rows that a later call will reuse
        if self.cache_limit <= 0:
            return
        for row_pos, pos in enumerate(new_dense_positions.tolist()):
            if len(self._cache) >= self.cache_limit:
                return
            key = int(idx_array[pos])
            if key in keep:
                self._cache[key] = _CachedRow(values=new_dense[row_pos].copy())
        for row_pos, pos in enumerate(new_sparse_positions.tolist()):
            if len(self._cache) >= self.cache_limit:
                return
            key = int(idx_array[pos])
            if key in keep:
                self._cache[key] = _CachedRow(
                    values=new_sparse_values[row_pos],
                    sample_idxs=new_sparse_idxs[row_pos],
                )

    def _empty(self, stats: BlockStats) -> BlockMatrix:
        """Return an empty BlockMatrix."""

        return BlockMatrix(
            dense=np.empty((0, self.sample_ct), dtype=self.arr_dtype),
            dense_rows=np.empty(0, dtype=np.uint32),
            sparse_rows=np.empty(0, dtype=np.uint32),
            sparse_sample_idxs=[],
            sparse_values=[],
            stats=stats,
        )

    def _read_dense(self, idx_array: npt.NDArray[np.uint32]) -> npt.NDArray:
        """Read a dense genotype/dosage matrix (one row per variant)."""

        # this will be overwritten, safe for np.empty
        out = np.empty((idx_array.size, self.sample_ct), dtype=self.arr_dtype)
        if self.dosage:
            self.reader.read_dosages_list(idx_array, out)
        else:
            self.reader.read_list(idx_array, out)
        return out

    def _read_sparse(
        self, idx_array: npt.NDArray[np.uint32]
    ) -> tuple[list[npt.NDArray[np.uint32]], list[npt.NDArray]]:
        """Read sparse rows, returning per-variant (sample indexes, values)."""

        lst_sample_idxs: list[npt.NDArray[np.uint32]] = []
        lst_values: list[npt.NDArray] = []
        for idx in idx_array.tolist():
            if self.dosage:
                sample_idxs, values = self.reader.read_sparse_dosages64(idx)
            else:
                sample_idxs, values = self.reader.read_sparse8(idx)
            lst_sample_idxs.append(np.asarray(sample_idxs, dtype=np.uint32))
            lst_values.append(values)
        return lst_sample_idxs, lst_values

    def _find_sparse(
        self, idx_array: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        """Find indeces of sparse variants."""

        idx_sparse: list[int] = []
        for idx in idx_array.tolist():
            if self._has_sparse(idx):
                idx_sparse.append(idx)
        return  np.asarray(idx_sparse, dtype=np.uint32)

    def _has_sparse(self, variant_idx: int) -> bool:
        """Check if a variant has a sparse representation for dosage or hardcalls."""

        if self.dosage:
            return self.reader.has_sparse_dosages(variant_idx)
        return self.reader.has_sparse(variant_idx)


__all__ = ["BlockMatrix", "BlockReader", "BlockStats"]
