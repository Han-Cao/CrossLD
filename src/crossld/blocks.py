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
    pairs_evaluated: int = 0
    kept: int = 0
    below_r2: int = 0
    skipped: int = 0
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
