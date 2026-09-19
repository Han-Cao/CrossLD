"""Block-wise genotype/dosage reading through :mod:`pgenlib`.

Reads dataset variants in blocks, using sparse representations when the underlying
``.pgen`` data is compatibly sparse and falling back to dense reads otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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


def slice_block(block: BlockMatrix, start: int, end: int) -> BlockMatrix:
    """Rows ``[start, end)`` of ``block`` as a **view** with fresh stats.

    ``start``/``end`` are 0-based positions in the block's requested variant
    list. The dense rows are a C-contiguous 2D slice (a numpy view, never a
    copy) and the sparse per-row arrays are shared; only the tiny position
    arrays and the :class:`BlockStats` object are new. The LD code never mutates
    these arrays, so views are safe to share with a cached block.
    """

    n_variants = int(block.stats.variants)
    if not 0 <= start <= end <= n_variants:
        raise ValueError(
            f"slice [{start}, {end}) out of range for a {n_variants}-variant block"
        )

    # ``dense_rows``/``sparse_rows`` are ascending row positions, so the rows
    # inside ``[start, end)`` are a contiguous slice of each array.
    d0 = int(np.searchsorted(block.dense_rows, start, side="left"))
    d1 = int(np.searchsorted(block.dense_rows, end, side="left"))
    s0 = int(np.searchsorted(block.sparse_rows, start, side="left"))
    s1 = int(np.searchsorted(block.sparse_rows, end, side="left"))

    stats = BlockStats(
        variants=end - start,
        dense_variants=d1 - d0,
        sparse_variants=s1 - s0,
        mode=block.stats.mode,
    )
    return BlockMatrix(
        dense=block.dense[d0:d1],
        dense_rows=block.dense_rows[d0:d1] - np.uint32(start),
        sparse_rows=block.sparse_rows[s0:s1] - np.uint32(start),
        sparse_sample_idxs=block.sparse_sample_idxs[s0:s1],
        sparse_values=block.sparse_values[s0:s1],
        stats=stats,
    )


@dataclass(slots=True)
class _CachedBlock:
    """One cached target block: its request-order indexes and decoded matrix."""

    indices: npt.NDArray[np.uint32]
    block: BlockMatrix


class BlockCache:
    """Whole-block dataset-2 read cache across overlapping target windows.

    A cache entry is a ``(variant_indices, BlockMatrix)`` pair in the dataset-2
    position order produced by :func:`~crossld.plinkio.candidate_indices`.
    :meth:`plan` reuses entries by intersecting them with a target's candidates
    and returning contiguous slices (numpy views for dense rows, shared per-row
    arrays for sparse rows); :meth:`restamp` rebuilds the cache from the blocks
    actually used, keeping the rightmost ``limit`` whose overlap with the next
    target's candidates is strictly greater than ``min_overlap``.

    The cache never changes results: ``limit <= 0`` leaves it empty and
    :meth:`plan` degenerates to plain ``block_size`` chunking.
    """

    def __init__(self, limit: int = 0, min_overlap: float = 0.5) -> None:
        self.limit = int(limit)
        self.min_overlap = float(min_overlap)
        self._entries: list[_CachedBlock] = []
        self.reused_variants = 0

    @property
    def size(self) -> int:
        """Number of blocks currently cached."""

        return len(self._entries)

    def clear(self) -> None:
        """Drop every cached block."""

        self._entries.clear()

    def plan(
        self,
        candidates: npt.NDArray[np.uint32],
        block_size: int,
    ) -> list[tuple[npt.NDArray[np.uint32], BlockMatrix | None]]:
        """Blocks for one target, in candidate order, each ``<= block_size``.

        Each cached entry is intersected with ``candidates``; with every array
        in the fixed position order the intersection is one contiguous run,
        emitted as a :func:`slice_block` view. Candidate rows not covered by any
        cached run are emitted as ``(index_chunk, None)`` for the caller to read.
        The two kinds are never merged (merging would copy). A defensive check
        emits the block fresh when an intersection is not a single contiguous
        run.
        """

        idx = np.asarray(candidates, dtype=np.uint32)
        n = int(idx.size)
        block_size = max(1, int(block_size))

        run_id = np.full(n, -1, dtype=np.int64)
        # (candidate_start, candidate_end, entry, block_start, block_end)
        runs: list[tuple[int, int, _CachedBlock, int, int]] = []

        if self.limit > 0 and n:
            for entry in self._entries:
                if entry.indices.size == 0:
                    continue
                _common, ib, ic = np.intersect1d(
                    entry.indices, idx, assume_unique=True, return_indices=True
                )
                if ic.size == 0:
                    continue
                order = np.argsort(ic, kind="stable")
                ic = ic[order]
                ib = ib[order]
                if not (np.all(np.diff(ic) == 1) and np.all(np.diff(ib) == 1)):
                    # Defensive: not a single contiguous run -> read it fresh.
                    continue
                cand_start, cand_end = int(ic[0]), int(ic[-1]) + 1
                blk_start, blk_end = int(ib[0]), int(ib[-1]) + 1
                if np.any(run_id[cand_start:cand_end] >= 0):
                    continue  # entries are disjoint; guard against overlap
                run_id[cand_start:cand_end] = len(runs)
                runs.append((cand_start, cand_end, entry, blk_start, blk_end))

        out: list[tuple[npt.NDArray[np.uint32], BlockMatrix | None]] = []
        i = 0
        while i < n:
            rid = int(run_id[i])
            if rid >= 0:
                cand_start, cand_end, entry, blk_start, blk_end = runs[rid]
                out.append(
                    (
                        idx[cand_start:cand_end].copy(),
                        slice_block(entry.block, blk_start, blk_end),
                    )
                )
                self.reused_variants += cand_end - cand_start
                i = cand_end
            else:
                j = i
                while j < n and run_id[j] < 0:
                    j += 1
                for a in range(i, j, block_size):
                    b = min(a + block_size, j)
                    out.append((idx[a:b].copy(), None))
                i = j

        return out

    def restamp(
        self,
        used: list[tuple[npt.NDArray[np.uint32], BlockMatrix]],
        next_candidates: npt.NDArray[np.uint32],
    ) -> None:
        """Rebuild the cache from this target's blocks for the next target.

        Keep every ``(indices, block)`` whose overlap with ``next_candidates``
        is strictly greater than ``min_overlap`` and retain the rightmost
        ``limit``. Blocks that are themselves views are stored as-is (cheap; the
        numpy view keeps the base array alive).
        """

        self._entries = []
        if self.limit <= 0:
            return
        nxt = np.asarray(next_candidates, dtype=np.uint32)
        if nxt.size == 0:
            return

        selected: list[_CachedBlock] = []
        for indices, block in used:
            arr = np.asarray(indices, dtype=np.uint32)
            if arr.size == 0:
                continue
            overlap = int(np.intersect1d(arr, nxt, assume_unique=True).size)
            if overlap / arr.size > self.min_overlap:
                selected.append(_CachedBlock(indices=arr.copy(), block=block))

        if len(selected) > self.limit:
            selected = selected[-self.limit:]
        self._entries = selected


@dataclass
class BlockReader:
    """Reads variant blocks as int8 hardcalls or float64 dosages on the active subset.

    For sparse variants, the reader always read the alt alelle (pgenlib's default).

    Notes for precision:
    Based on tests, PLINK2's LD is estimated in float64, to replicate the same precision,
    we also use np.float64 for all dosages and downstream LD calaculation.
    
    TODO: consider make --float32 as a optional flag
    """

    dataset: PlinkDataset
    dosage: bool = False
    force_dense: bool = False

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

    def read(self, indices: list[int] | npt.NDArray[np.uint32]) -> BlockMatrix:
        """Read ``indices`` as one :class:`BlockMatrix` (dense and sparse rows)."""

        idx_array = np.asarray(indices, dtype=np.uint32)
        stats = BlockStats(
            variants=int(idx_array.size),
            mode="dosage" if self.dosage else "hardcall",
        )
        if idx_array.size == 0:
            return self._empty(stats)

        if self.force_dense or self.fmt == "bfile":
            idx_sparse = np.empty(0, dtype=np.uint32)
        else:
            idx_sparse = self._find_sparse(idx_array)

        if idx_sparse.size:
            sparse_mask = np.isin(idx_array, idx_sparse)
        else:
            sparse_mask = np.zeros(idx_array.size, dtype=bool)
        dense_rows = np.flatnonzero(~sparse_mask).astype(np.uint32)
        sparse_rows = np.flatnonzero(sparse_mask).astype(np.uint32)

        if dense_rows.size:
            dense = self._read_dense(idx_array[dense_rows])
        else:
            dense = np.empty((0, self.sample_ct), dtype=self.arr_dtype)

        if sparse_rows.size:
            sparse_sample_idxs, sparse_values = self._read_sparse(idx_array[sparse_rows])
        else:
            sparse_sample_idxs, sparse_values = [], []

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


__all__ = ["BlockCache", "BlockMatrix", "BlockReader", "BlockStats", "slice_block"]
