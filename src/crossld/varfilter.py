"""Dataset-2 variant filtering: ``--extract``, ``--exclude``, and BED interval files.

Each option accepts exactly one file. The resulting boolean mask is applied to
dataset 2 *before* window selection, LD computation, and output, so excluded
variants behave as if they were absent from the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .logging_setup import LOGGER_NAME
from .plinkio import CrossLDError, Dataset

logger = logging.getLogger(f"{LOGGER_NAME}.varfilter")

ID_OPTIONS = ("extract", "exclude")
BED_OPTIONS = ("extract_bed", "exclude_bed")


@dataclass
class VariantFilter:
    """The four dataset-2 filter options (at most one file each, mutually exclusive)."""

    extract: str | None = None
    exclude: str | None = None
    extract_bed: str | None = None
    exclude_bed: str | None = None

    @property
    def active(self) -> bool:
        return any(
            path is not None
            for path in (self.extract, self.exclude, self.extract_bed, self.exclude_bed)
        )

    def validate(self) -> None:
        if self.extract and self.exclude:
            raise CrossLDError("--extract and --exclude are mutually exclusive")
        if self.extract_bed and self.exclude_bed:
            raise CrossLDError("--extract-bed and --exclude-bed are mutually exclusive")
        has_extract = bool(self.extract or self.extract_bed)
        has_exclude = bool(self.exclude or self.exclude_bed)
        if has_extract and has_exclude:
            raise CrossLDError(
                "--extract/--extract-bed cannot be combined with "
                "--exclude/--exclude-bed"
            )

    def build_mask(self, ds: Dataset) -> np.ndarray:
        """Return a boolean mask over dataset-2 variants.

        ``--extract`` and ``--extract-bed`` are unioned (a variant survives if
        either selects it); ``--exclude`` and ``--exclude-bed`` are likewise
        unioned. Mixing the two polarities is rejected in :meth:`validate`.
        """

        self.validate()
        mask = np.ones(ds.variant_ct, dtype=bool)

        if self.extract:
            ids, _ = read_id_file(self.extract)
            self._warn_unmatched("--extract", self.extract, ids, ds)
            present = np.zeros(ds.variant_ct, dtype=bool)
            known = set(ids)
            for i, vid in enumerate(ds.ids):
                if vid in known:
                    present[i] = True
            mask = present
        if self.extract_bed:
            intervals = read_bed_file(self.extract_bed)
            selected = _in_intervals(ds, intervals)
            mask = selected if not self.extract else (mask | selected)

        if self.exclude:
            ids, _ = read_id_file(self.exclude)
            self._warn_unmatched("--exclude", self.exclude, ids, ds)
            known = set(ids)
            for i, vid in enumerate(ds.ids):
                if vid in known:
                    mask[i] = False
        elif self.exclude_bed:
            intervals = read_bed_file(self.exclude_bed)
            mask &= ~_in_intervals(ds, intervals)

        kept = int(mask.sum())
        if kept == 0:
            raise CrossLDError(
                "dataset-2 variant filtering left no variants; check the filter file"
            )
        if kept < 3:
            logger.warning(
                "dataset-2 variant filtering left only %d variant(s)", kept
            )
        logger.info(
            "variant-filter: %s kept=%d excluded=%d",
            self.describe(),
            kept,
            ds.variant_ct - kept,
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
                parts.append(f"--{name.replace('_', '-')}={path}")
        return " ".join(parts) if parts else "none"

    @staticmethod
    def _warn_unmatched(option: str, path: str, ids: list[str], ds: Dataset) -> None:
        unmatched = [vid for vid in ids if ds.index_of(vid) is None]
        if not unmatched:
            return
        sample = ", ".join(unmatched[:20])
        more = "" if len(unmatched) <= 20 else f" (and {len(unmatched) - 20} more)"
        logger.warning(
            "%s %s: %d of %d listed ID(s) matched no variant in dataset 2: %s%s",
            option,
            path,
            len(unmatched),
            len(ids),
            sample,
            more,
        )


def read_id_file(path: str) -> tuple[list[str], list[str]]:
    """Read a single-column variant ID file.

    Returns ``(ids, unmatched)``; ``unmatched`` is filled in by the caller against
    a specific dataset, so it is always empty here.
    """

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.append(stripped.split()[0])
    logger.debug("id-file %s: %d ID(s)", path, len(ids))
    return ids, []


def read_bed_file(path: str) -> list[tuple[str, int, int]]:
    """Read a 0-based, half-open interval BED file (``CHROM START END``)."""

    intervals: list[tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise CrossLDError(
                    f"--extract-bed/--exclude-bed {path}: line {lineno} has "
                    f"{len(fields)} field(s); expected 'CHROM START END'"
                )
            chrom = _normalize_chrom(fields[0])
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise CrossLDError(
                    f"--extract-bed/--exclude-bed {path}: line {lineno} has a "
                    f"non-integer interval bound: {stripped!r}"
                ) from exc
            if start < 0 or start >= end:
                raise CrossLDError(
                    f"--extract-bed/--exclude-bed {path}: line {lineno} has an "
                    f"invalid 0-based half-open interval [{start}, {end})"
                )
            intervals.append((chrom, start, end))
    logger.debug("bed-file %s: %d interval(s)", path, len(intervals))
    return intervals


def _normalize_chrom(chrom: str) -> str:
    chrom = chrom.lower()
    return chrom[3:] if chrom.startswith("chr") else chrom


def _in_intervals(ds: Dataset, intervals: list[tuple[str, int, int]]) -> np.ndarray:
    mask = np.zeros(ds.variant_ct, dtype=bool)
    for chrom, start, end in intervals:
        norm = _normalize_chrom(chrom)
        for i in range(ds.variant_ct):
            if mask[i]:
                continue
            if _normalize_chrom(ds.chrom[i]) != norm:
                continue
            if start <= int(ds.pos[i]) < end:
                mask[i] = True
    return mask


__all__ = [
    "BED_OPTIONS",
    "ID_OPTIONS",
    "VariantFilter",
    "read_bed_file",
    "read_id_file",
]
