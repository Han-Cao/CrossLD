"""Block-wise genotype/dosage reading through :mod:`pgenlib`.

Reads dataset variants in blocks, using sparse representations when the underlying
``.pgen`` data is compatibly sparse and falling back to dense reads otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .logging_setup import LOGGER_NAME
from .plinkio import Dataset

logger = logging.getLogger(f"{LOGGER_NAME}.blocks")

MISSING = -9.0


@dataclass
class BlockStats:
    """Per-block statistics emitted at DEBUG level."""

    block_index: int = 1
    block_total: int = 1
    variants: int = 0
    sparse_variants: int = 0
    sparse_fallbacks: int = 0
    pairs_evaluated: int = 0
    kept: int = 0
    below_r2: int = 0
    skipped: int = 0
    missing_frac: float = 0.0
    mode: str = "hardcall"


class BlockReader:
    """Reads variant blocks as float64 matrices on the selected sample subset.

    ``flip[i]`` (when given) marks variants whose ALT-coded vector must be recoded
    to the REF allele (used by ``--ref-based``); in major-allele mode the caller
    decides orientation per block.
    """

    def __init__(
        self,
        dataset: Dataset,
        sparse: str = "auto",
        dosage: bool = False,
    ) -> None:
        self.dataset = dataset
        self.reader = dataset._reader  # type: ignore[attr-defined]
        self.sparse = sparse
        self.dosage = dosage

    @property
    def sample_ct(self) -> int:
        """Samples the reader returns per variant (the active subset size)."""

        return self.dataset.active_sample_ct

    def read(self, indices, flip: np.ndarray | None = None) -> tuple[np.ndarray, BlockStats]:
        """Return ``(matrix, stats)`` for ``indices`` (one row per variant)."""

        idx_array = np.asarray(indices, dtype=np.uint32)
        stats = BlockStats(
            variants=int(idx_array.size),
            mode="dosage" if self.dosage else "hardcall",
        )
        if idx_array.size == 0:
            matrix = np.zeros((0, self.sample_ct), dtype=np.float64)
            return matrix, stats

        if self.sparse != "never":
            matrix = self._read_sparse(idx_array, stats, flip)
            if matrix is not None:
                stats.missing_frac = _missing_fraction(matrix)
                return matrix, stats
            if self.sparse == "require":
                raise RuntimeError(
                    "sparse reads were required but the underlying data does not "
                    "support a compatible sparse representation"
                )
        matrix = self._read_dense(idx_array)
        if flip is not None:
            matrix = _apply_flip(matrix, flip)
        stats.missing_frac = _missing_fraction(matrix)
        return matrix, stats

    def _read_dense(self, idx_array: np.ndarray) -> np.ndarray:
        sample_ct = self.sample_ct
        if self.dosage:
            out = np.empty((idx_array.size, sample_ct), dtype=np.float64)
            self.reader.read_dosages_list(idx_array, out)
            return out
        out = np.empty((idx_array.size, sample_ct), dtype=np.int8)
        self.reader.read_list(idx_array, out)
        return out.astype(np.float64)

    def _read_sparse(
        self,
        idx_array: np.ndarray,
        stats: BlockStats,
        flip: np.ndarray | None,
    ) -> np.ndarray | None:
        """Sparse read, or ``None`` when the block must be read densely.

        Sparse data encodes only non-zero entries (verified against
        ``read``/``read_dosages``), so a zero-filled matrix is an exact
        reconstruction. If any variant in the block lacks a compatible sparse
        representation the whole block falls back to a dense read.
        """

        sample_ct = self.sample_ct
        matrix = np.zeros((idx_array.size, sample_ct), dtype=np.float64)
        for row, raw_index in enumerate(idx_array):
            index = int(raw_index)
            try:
                if self.dosage:
                    if not self.reader.has_sparse_dosages(index):
                        stats.sparse_fallbacks += 1
                        return None
                    sample_idxs, values = self.reader.read_sparse_dosages32(index)
                else:
                    if not self.reader.has_sparse(index):
                        stats.sparse_fallbacks += 1
                        return None
                    sample_idxs, values = self.reader.read_sparse8(index)
            except RuntimeError:
                stats.sparse_fallbacks += 1
                logger.debug(
                    "variant %s: sparse read unavailable, using dense read",
                    self.dataset.ids[index],
                )
                return None

            matrix[row, sample_idxs.astype(np.int64)] = values.astype(np.float64)
            stats.sparse_variants += 1

        if flip is not None:
            matrix = _apply_flip(matrix, flip)
        return matrix


def _apply_flip(matrix: np.ndarray, flip: np.ndarray) -> np.ndarray:
    flipped = np.asarray(flip, dtype=bool)[:, None]
    return np.where(flipped & (matrix >= 0), 2.0 - matrix, matrix)


def _missing_fraction(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float((matrix < 0).sum()) / float(matrix.size)


__all__ = ["BlockReader", "BlockStats", "MISSING"]
