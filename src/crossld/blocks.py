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

MISSING = -9.0


@dataclass(slots=True)
class BlockStats:
    """Per-block statistics emitted at DEBUG level."""

    block_index: int = 1
    block_total: int = 1
    variants: int = 0
    sparse_variants: int = 0
    pairs_evaluated: int = 0
    kept: int = 0
    below_r2: int = 0
    skipped: int = 0
    mode: str = "hardcall"


@dataclass()
class BlockReader:
    """Reads variant blocks as int8 or float32 matrices on the selected sample subset."""

    dataset: PlinkDataset
    dosage: bool = False
    r_major: bool = False

    def __post_init__(self) -> None:
        self.reader = self.dataset._reader
        self.fmt = self.dataset.fmt
        if self.dosage:
            self.arr_dtype = np.float32
        else:
            self.arr_dtype = np.int8

    @property
    def sample_ct(self) -> int:
        """Samples the reader returns per variant (the active subset size)."""

        return self.dataset.active_sample_ct

    def read(
        self, 
        indices: list[int] | npt.NDArray[np.uint32]
    ) -> tuple[npt.NDArray, list[npt.NDArray[np.uint32]], list[npt.NDArray], BlockStats]:
        """Return ``(dense_matrix, sparse_sample_idx, sparse_geno_arr, stats)`` with one row per variant."""

        idx_array = np.asarray(indices, dtype=np.uint32)
        stats = BlockStats(
            variants=int(idx_array.size),
            mode="dosage" if self.dosage else "hardcall",
        )
        # TODO: see if need this
        if idx_array.size == 0:
            matrix = np.zeros((0, self.sample_ct), dtype=self.arr_dtype)
            return matrix, stats

        idx_dense, idx_sparse = self._find_dense_sparse(idx_array)
        if len(idx_dense) > 0:
            mat_dense = self._read_dense(idx_dense)
        else:
            mat_dense = np.empty((0, self.sample_ct))
        if len(idx_sparse) > 0:
            sparse_sample_ids, sparse_geno_arr,  = self._read_sparse(idx_sparse)
        else:
            sparse_sample_ids = []
            sparse_geno_arr = []

        stats.sparse_variants = len(idx_sparse)

        return mat_dense, sparse_sample_ids, sparse_geno_arr, stats

    def _read_dense(self, idx_array: npt.NDArray[np.uint32]) -> npt.NDArray:
        """Read a dense genotype/dosage matrix"""
        sample_ct = self.sample_ct
        if self.dosage:
            out = np.empty((idx_array.size, sample_ct), dtype=self.arr_dtype)
            self.reader.read_dosages_list(idx_array, out)
            return out
        else:
            out = np.empty((idx_array.size, sample_ct), dtype=self.arr_dtype)
            self.reader.read_list(idx_array, out)
            return out

    def _read_sparse(self, idx_array: npt.NDArray[np.uint32]) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
        """Read sparse genotype/dosage matrix, return list of sample indices and genotype/dosage arrays"""

        lst_sample_idxs: list[npt.NDArray[np.uint32]] = []
        lst_geno_arr: list[npt.NDArray] = []
        for idx in idx_array.tolist():
            if self.dosage:
                sample_idxs, geno_arr = self.reader.read_sparse_dosages32(idx)
            else:
                sample_idxs, geno_arr = self.reader.read_sparse8(idx)
            lst_sample_idxs.append(sample_idxs)
            lst_geno_arr.append(geno_arr)
        return lst_sample_idxs, lst_geno_arr

    def _find_dense_sparse(
        self, 
        idx_array: npt.NDArray[np.uint32]
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        """Return indices of dense and sparse variants."""

        if self.fmt == "bfile":
            return idx_array, np.empty(0, dtype=np.uint32)

        idx_dense: list[int] = []
        idx_sparse: list[int] = []
        for idx in idx_array.tolist():
            if self._has_sparse(idx):
                idx_sparse.append(idx)
            else:
                idx_dense.append(idx)

        return np.asarray(idx_dense, dtype=np.uint32), np.asarray(idx_sparse, dtype=np.uint32)

    def _has_sparse(self, variant_idx: int) -> bool:
        """Check if a variant has a sparse representation for dosage or hardcalls."""

        if self.dosage:
            return self.reader.has_sparse_dosages(variant_idx)
        else:
            return self.reader.has_sparse(variant_idx)


__all__ = ["MISSING", "BlockReader", "BlockStats"]
