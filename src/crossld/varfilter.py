"""Dataset-2 variant filtering: ``--extract``, ``--exclude``, and BED interval files.

Each option accepts exactly one file. The resulting boolean mask is applied to
dataset 2 *before* window selection, LD computation, and output, so excluded
variants behave as if they were absent from the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crossld.logging_setup import LOGGER_NAME
from crossld.plinkio import PlinkDataset

logger = logging.getLogger(f"{LOGGER_NAME}.varfilter")

ID_OPTIONS = ("extract", "exclude")
BED_OPTIONS = ("extract_bed", "exclude_bed")


class VarFilterError(Exception):
    """Fatal, user-facing error (reported as ``error: ...`` and exit code 2)."""


@dataclass
class VariantFilter:
    """The four dataset-2 filter options."""

    extract: str | None = None
    exclude: str | None = None
    extract_bed: str | None = None
    exclude_bed: str | None = None
    has_extract: bool = False
    has_exclude: bool = False

    @property
    def active(self) -> bool:
        return any(
            path is not None
            for path in (self.extract, self.exclude, self.extract_bed, self.exclude_bed)
        )

    def validate(self) -> None:
        self.has_extract = bool(self.extract or self.extract_bed)
        self.has_exclude = bool(self.exclude or self.exclude_bed)
        if self.has_extract and self.has_exclude:
            raise VarFilterError(
                "--extract/--extract-bed cannot be combined with "
                "--exclude/--exclude-bed"
            )

    def build_mask(self, ds: PlinkDataset) -> np.ndarray:
        """Return a boolean mask over dataset-2 variants.

        ``--extract`` and ``--extract-bed`` are unioned;
        ``--exclude`` and ``--exclude-bed`` are unioned.
        Mixing the two polarities is rejected in :meth:`validate`.
        """

        self.validate()

        if self.has_extract:
            mask = np.zeros(ds.variant_ct, dtype=bool)
        elif self.has_exclude:
            mask = np.ones(ds.variant_ct, dtype=bool)
        else:
            return np.ones(ds.variant_ct, dtype=bool)

        # build vid index of ds if missing
        if self.extract or self.exclude:
            ds.build_vid_index()

        # TODO:
        # for large ids, consider using np.isin()
        if self.extract:
            ids = read_vid_file(self.extract)
            exist_ids = set(ids).intersection(ds.unique_ids)
            idx_extract: list[int] = []
            for vid in exist_ids:
                idx_extract.append(ds.index_of(vid))
            mask[idx_extract] = True
        if self.extract_bed:
            extract_bed = read_bed_file(self.extract_bed)
            in_extract_bed = _in_intervals(ds, extract_bed)
            mask |= in_extract_bed

        if self.exclude:
            ids = read_vid_file(self.exclude)
            exclude_ids = set(ids).intersection(ds.unique_ids)
            idx_exclude: list[int] = []
            for vid in exclude_ids:
                idx_exclude.append(ds.index_of(vid))
            mask[idx_exclude] = False
        elif self.exclude_bed:
            exclude_bed = read_bed_file(self.exclude_bed)
            in_exclude_bed = _in_intervals(ds, exclude_bed)
            mask &= ~in_exclude_bed

        kept = int(mask.sum())
        if kept == 0:
            raise VarFilterError(
                "dataset-2 variant filtering left no variants; check the filter file"
            )
        logger.info(
            "variant-filter: %s kept %d variants",
            self.describe(),
            kept
        )
        logger.debug(
            "variant-filter: dataset-2 total=%d kept=%d",
            ds.variant_ct,
            kept,
        )
        return mask

    def describe(self) -> str:
        parts = []
        for name in ID_OPTIONS + BED_OPTIONS:
            path = getattr(self, name)
            if path:
                parts.append(f"--{name.replace('_', '-')}")
        return " and ".join(parts)


def read_vid_file(path: str) -> list[str]:
    """Read a single-column variant ID file. Ignore empty lines or lines starts with ``#``."""

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.append(stripped.split()[0])
    logger.debug("id-file %s: %d ID(s)", path, len(ids))
    return ids


def read_bed_file(path: str) -> list[tuple[str, int, int]]:
    """Read a 0-based, half-open interval BED file (``CHROM START END``)."""

    intervals: list[tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for n_line, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(sep="\t")
            if len(fields) != 3:
                raise VarFilterError(
                    f"--extract-bed/--exclude-bed {path}: line {n_line} has "
                    f"{len(fields)} field(s); expected 'CHROM START END'"
                )
            chrom = _normalize_chrom(fields[0])
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise VarFilterError(
                    f"--extract-bed/--exclude-bed {path}: line {n_line} has a "
                    f"non-integer interval bound: {stripped!r}"
                ) from exc
            if start < 0 or start >= end:
                raise VarFilterError(
                    f"--extract-bed/--exclude-bed {path}: line {n_line} has an "
                    f"invalid 0-based half-open interval [{start}, {end})"
                )
            intervals.append((chrom, start, end))
    logger.debug("bed-file %s: %d interval(s)", path, len(intervals))
    return intervals


def _normalize_chrom(chrom: str) -> str:
    """Remove chr prefix"""
    chrom = chrom.lower()
    return chrom.removeprefix("chr")


def _in_intervals(ds: PlinkDataset, intervals: list[tuple[str, int, int]]) -> npt.NDArray[np.bool_]:
    mask = np.zeros(ds.variant_ct, dtype=bool)
    for chrom, start, end in intervals:
        entry = ds.get_sorted_index(chrom)
        if entry is None:
            continue

        # pos is 1-based postion
        # start, end is 0-based half-open interval
        pos, idx = entry
        lo = np.searchsorted(pos, start + 1, side="left")
        hi = np.searchsorted(pos, end, side="right")
        if lo < hi:
            mask[idx[lo:hi]] = True

    return mask


__all__ = [
    "BED_OPTIONS",
    "ID_OPTIONS",
    "VarFilterError",
    "VariantFilter",
    "read_bed_file",
    "read_vid_file",
]
