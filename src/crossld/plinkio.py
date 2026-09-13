"""Dataset resolution, sample selection, and variant index handling.

Supports PLINK 1 ``.bed/.bim/.fam`` and PLINK 2 ``.pgen/.pvar/.psam`` prefixes, read
through :mod:`pgenlib` (:class:`~pgenlib.PgenReader` + :class:`~pgenlib.PvarReader`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pgenlib

from .logging_setup import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.plinkio")


class CrossLDError(Exception):
    """Fatal, user-facing error (reported as ``error: ...`` and exit code 2)."""


@dataclass
class Dataset:
    """A resolved PLINK 1 or PLINK 2 dataset."""

    prefix: str
    label: str
    fmt: str  # "bfile" or "pfile"
    genotype_path: str
    variant_path: str
    sample_path: str
    chrom: list[str] = field(default_factory=list)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    ids: list[str] = field(default_factory=list)
    ref: list[str] = field(default_factory=list)
    alt: list[str] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    _id_to_index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.ids and not self._id_to_index:
            self._id_to_index = {vid: i for i, vid in enumerate(self.ids)}

    @property
    def variant_ct(self) -> int:
        return len(self.ids)

    @property
    def sample_ct(self) -> int:
        return len(self.sample_ids)

    @property
    def max_allele_ct(self) -> int:
        return self._max_allele_ct

    def index_of(self, variant_id: str) -> int | None:
        return self._id_to_index.get(variant_id)

    @property
    def reader(self):
        """The active :class:`pgenlib.PgenReader`."""

        return self._reader

    @property
    def active_sample_ct(self) -> int:
        """Number of samples the reader currently returns per variant."""

        return self._active_sample_ct

    def set_sample_subset(self, subset: np.ndarray | None) -> None:
        """Switch the reader to ``subset`` and track the resulting sample count."""

        self._reader.change_sample_subset(subset)
        self._active_sample_ct = self.sample_ct if subset is None else int(len(subset))
        logger.debug(
            "dataset %s: active sample subset size %d",
            self.label,
            self._active_sample_ct,
        )

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            try:
                reader.close()
            finally:
                self._reader = None
        pvar = getattr(self, "_pvar", None)
        if pvar is not None:
            try:
                pvar.close()
            finally:
                self._pvar = None


@dataclass
class SelectionSummary:
    """Outcome of sample selection."""

    count1: int
    count2: int
    shared: int
    requested: int | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def missing_count(self) -> int:
        return len(self.missing)


def _parse_sample_file(path: str) -> list[str]:
    """Read IIDs from a ``.psam`` or ``.fam`` file (column 1 for psam, 1 for fam)."""

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            ids.append(fields[0])
    return ids


def _parse_fam(path: str, fid_iid: bool = False) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise CrossLDError(f"{path}: malformed .fam line: {stripped!r}")
            ids.append(f"{fields[0]}_{fields[1]}" if fid_iid else fields[1])
    return ids


def _first_existing(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_files(prefix: str, fmt: str) -> tuple[str, str, str]:
    if fmt == "bfile":
        bed = _first_existing([f"{prefix}.bed"])
        bim = _first_existing([f"{prefix}.bim"])
        fam = _first_existing([f"{prefix}.fam"])
        missing = [
            name
            for name, path in ((".bed", bed), (".bim", bim), (".fam", fam))
            if path is None
        ]
        if missing:
            raise CrossLDError(
                f"--bfile {prefix}: missing file(s) {', '.join(prefix + m for m in missing)}"
            )
        return bed, bim, fam

    pgen = _first_existing([f"{prefix}.pgen", f"{prefix}.bed"])
    if pgen is None:
        raise CrossLDError(f"--pfile {prefix}: missing {prefix}.pgen (or {prefix}.bed)")
    variant_path = _first_existing([f"{prefix}.pvar", f"{prefix}.bim"])
    if variant_path is None:
        raise CrossLDError(f"--pfile {prefix}: missing {prefix}.pvar (or {prefix}.bim)")
    sample_path = _first_existing([f"{prefix}.psam", f"{prefix}.fam"])
    if sample_path is None:
        raise CrossLDError(f"--pfile {prefix}: missing {prefix}.psam (or {prefix}.fam)")
    return pgen, variant_path, sample_path


def load_dataset(
    prefix: str,
    label: str,
    fmt: str,
    dosage: bool = False,
    fid_iid: bool = False,
    sample_subset: np.ndarray | None = None,
) -> Dataset:
    """Open a dataset and read its variant metadata.

    When ``sample_subset`` is given the reader is created with that subset so that
    downstream reads honor it; otherwise the reader spans all samples (the subset
    can be applied later with :meth:`pgenlib.PgenReader.change_sample_subset`).
    """

    genotype_path, variant_path, sample_path = _resolve_files(prefix, fmt)

    if sample_path.endswith(".psam"):
        sample_ids = _parse_sample_file(sample_path)
    else:
        sample_ids = _parse_fam(sample_path, fid_iid=fid_iid)
    if not sample_ids:
        raise CrossLDError(f"{prefix}: no samples found in {sample_path}")

    pvar = pgenlib.PvarReader(variant_path.encode())
    variant_ct = pvar.get_variant_ct()
    offsets = pvar.get_allele_idx_offsets()
    max_allele_ct = max(2, int(np.diff(offsets).max()) if variant_ct else 2)
    if max_allele_ct > 2:
        raise CrossLDError(
            f"{prefix}: multiallelic variants are not supported "
            f"(max allele count {max_allele_ct})"
        )

    chrom: list[str] = []
    pos = np.empty(variant_ct, dtype=np.int64)
    ids: list[str] = []
    ref: list[str] = []
    alt: list[str] = []
    for i in range(variant_ct):
        chrom.append(pvar.get_variant_chrom(i).decode())
        pos[i] = pvar.get_variant_pos(i)
        ids.append(pvar.get_variant_id(i).decode())
        ref.append(pvar.get_allele_code(i, 0).decode())
        alt.append(pvar.get_allele_code(i, 1).decode())

    is_bed = genotype_path.endswith(".bed")
    reader_kwargs: dict = {"sample_subset": sample_subset}
    if is_bed:
        reader_kwargs["raw_sample_ct"] = len(sample_ids)
        reader_kwargs["variant_ct"] = variant_ct
    else:
        reader_kwargs["pvar"] = pvar
    reader = pgenlib.PgenReader(genotype_path.encode(), **reader_kwargs)
    if reader.get_raw_sample_ct() != len(sample_ids):
        raise CrossLDError(
            f"{prefix}: {sample_path} describes {len(sample_ids)} samples but the "
            f"genotype file has {reader.get_raw_sample_ct()}"
        )

    ds = Dataset(
        prefix=prefix,
        label=label,
        fmt=fmt,
        genotype_path=genotype_path,
        variant_path=variant_path,
        sample_path=sample_path,
        chrom=chrom,
        pos=pos,
        ids=ids,
        ref=ref,
        alt=alt,
        sample_ids=sample_ids,
    )
    ds._reader = reader  # type: ignore[attr-defined]
    ds._pvar = pvar  # type: ignore[attr-defined]
    ds._max_allele_ct = max_allele_ct  # type: ignore[attr-defined]
    ds._dosage = dosage  # type: ignore[attr-defined]
    ds._active_sample_ct = (  # type: ignore[attr-defined]
        len(sample_ids) if sample_subset is None else int(len(sample_subset))
    )
    logger.info(
        "dataset %s: format=%s variants=%d samples=%d genotype=%s",
        label,
        fmt,
        variant_ct,
        len(sample_ids),
        genotype_path,
    )
    logger.debug(
        "dataset %s: variant-meta=%s sample-meta=%s chroms=%s",
        label,
        variant_path,
        sample_path,
        sorted(set(chrom)),
    )
    return ds


def read_keep_file(path: str, fid_iid: bool = False) -> list[str]:
    """Read sample IDs from a ``--keep`` file.

    A first line whose first whitespace-delimited token starts with ``#`` is
    treated as a header (first column used, extra columns ignored). Without a
    header the file must contain exactly one column.
    """

    requested: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if stripped.startswith("#"):
                logger.debug("keep %s: header/comment line %r skipped", path, stripped)
                continue
            if len(fields) != 1:
                raise CrossLDError(
                    f"--keep {path}: line {lineno} has {len(fields)} columns; a "
                    "single-column IID file is required (or a '#IID' header line)"
                )
            requested.append(fields[0])
    logger.debug("keep %s: %d requested sample(s)", path, len(requested))
    return requested


def select_samples(
    ds1: Dataset,
    ds2: Dataset,
    keep: str | None = None,
    fid_iid: bool = False,
) -> tuple[np.ndarray, np.ndarray, SelectionSummary]:
    """Choose the shared sample subset, returning increasing index arrays.

    Default: the IID intersection in dataset-1 order. With ``--keep``: the listed
    samples present in both datasets, in keep-file order.
    """

    idx1 = {sid: i for i, sid in enumerate(ds1.sample_ids)}
    idx2 = {sid: i for i, sid in enumerate(ds2.sample_ids)}

    if keep is None:
        requested = None
        shared = [sid for sid in ds1.sample_ids if sid in idx2]
        missing: list[str] = []
    else:
        requested_ids = read_keep_file(keep, fid_iid=fid_iid)
        requested = len(requested_ids)
        shared = []
        missing = []
        missing1: list[str] = []
        missing2: list[str] = []
        seen: set[str] = set()
        for sid in requested_ids:
            if sid in seen:
                continue
            seen.add(sid)
            if sid in idx1 and sid in idx2:
                shared.append(sid)
                continue
            missing.append(sid)
            if sid not in idx1:
                missing1.append(sid)
            if sid not in idx2:
                missing2.append(sid)
        _warn_missing(keep, "dataset 1", missing1, requested)
        _warn_missing(keep, "dataset 2", missing2, requested)
        if missing:
            logger.warning(
                "%d of %d --keep sample(s) not found in both datasets; using %d "
                "shared sample(s)",
                len(missing),
                requested,
                len(shared),
            )

    if len(shared) < 3:
        raise CrossLDError(
            f"only {len(shared)} shared sample(s) between dataset 1 "
            f"({ds1.sample_ct} samples) and dataset 2 ({ds2.sample_ct} samples); "
            "at least 3 are required"
        )

    subset1 = np.array([idx1[sid] for sid in shared], dtype=np.uint32)
    subset2 = np.array([idx2[sid] for sid in shared], dtype=np.uint32)
    summary = SelectionSummary(
        count1=ds1.sample_ct,
        count2=ds2.sample_ct,
        shared=len(shared),
        requested=requested,
        missing=missing,
    )
    if missing:
        shown = missing[:20]
        for sid in shown:
            logger.warning(
                "keep: sample %s not found in %s",
                sid,
                "dataset 1 and/or dataset 2",
            )
        if len(missing) > len(shown):
            logger.warning("keep: ...and %d more missing sample(s)", len(missing) - len(shown))
    logger.info(
        "samples: dataset1=%d dataset2=%d shared=%d%s",
        ds1.sample_ct,
        ds2.sample_ct,
        len(shared),
        "" if requested is None else f" requested={requested} missing={len(missing)}",
    )
    return subset1, subset2, summary


def _warn_missing(path: str, label: str, ids: list[str], requested: int | None) -> None:
    if not ids:
        return
    shown = ", ".join(ids[:20])
    more = "" if len(ids) <= 20 else f" (and {len(ids) - 20} more)"
    logger.warning(
        "--keep %s: %d of %s requested sample(s) absent from %s: %s%s",
        path,
        len(ids),
        requested,
        label,
        shown,
        more,
    )


def expand_ld_snps(spec: str, ds: Dataset) -> list[int]:
    """Resolve ``--ld-snps`` (comma-separated IDs and ``CHR:START-END`` ranges)."""

    indices: list[int] = []
    seen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        hit = _expand_token(token, ds)
        if hit is None:
            logger.warning("--ld-snps: %r matched no variant in dataset 1", token)
            continue
        for i in hit:
            if i not in seen:
                seen.add(i)
                indices.append(i)
    return indices


def _expand_token(token: str, ds: Dataset) -> list[int] | None:
    """Resolve one ``--ld-snps`` token to dataset-1 indexes.

    Accepts a variant ID or a range in ``CHR:START-END`` (single chromosome) or
    ``CHR:START-CHR:END`` form; bounds are inclusive base-pair coordinates.
    """

    ranges = _parse_range(token)
    if ranges is not None:
        chrom, start, end, chrom2 = ranges
        if chrom != chrom2:
            return []
        return [
            i
            for i in range(ds.variant_ct)
            if ds.chrom[i] == chrom and start <= ds.pos[i] <= end
        ]
    index = ds.index_of(token)
    return None if index is None else [index]


def _parse_range(token: str) -> tuple[str, int, int, str] | None:
    """Parse ``CHR:START-END`` / ``CHR:START-CHR:END``; ``None`` if not a range."""

    if token.count("-") != 1:
        return None
    head, tail = token.split("-", 1)
    if ":" not in head or not head.count(":"):
        return None
    chrom, _, start_s = head.rpartition(":")
    if ":" in tail:
        chrom2, _, end_s = tail.rpartition(":")
    else:
        chrom2, end_s = chrom, tail
    try:
        start, end = int(start_s), int(end_s)
    except ValueError:
        return None
    if end < start:
        start, end = end, start
    return chrom, start, end, chrom2


def read_ld_snp_list(path: str) -> list[str]:
    """Read ``--ld-snp-list``: one variant ID per line, ``#`` comments allowed."""

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.append(stripped.split()[0])
    return ids


def resolve_targets(
    ds: Dataset,
    single: str | None = None,
    snps: str | None = None,
    snp_list: str | None = None,
) -> list[int]:
    """Resolve the dataset-1 target variants, warning about anything unmatched."""

    if single is not None:
        index = ds.index_of(single)
        if index is None:
            logger.warning("--ld-snp: %r matched no variant in dataset 1", single)
            return []
        return [index]
    if snps is not None:
        return expand_ld_snps(snps, ds)
    if snp_list is not None:
        indices: list[int] = []
        seen: set[int] = set()
        for vid in read_ld_snp_list(snp_list):
            index = ds.index_of(vid)
            if index is None:
                logger.warning("--ld-snp-list: %r matched no variant in dataset 1", vid)
                continue
            if index not in seen:
                seen.add(index)
                indices.append(index)
        return indices
    return []


def candidate_indices(
    ds1: Dataset,
    ds2: Dataset,
    target_index: int,
    window_kb: float,
    window_count: int,
    kept_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Dataset-2 variant indexes within the target's window.

    ``window_count`` is the maximum number of dataset-2 variants (by file order
    over the post-filter set) on each side of the target; ``0`` means no limit.
    This deliberately differs from PLINK 2's ``--ld-window`` (which counts the
    target column itself), because dataset 1 and dataset 2 are separate files.
    """

    chrom = ds1.chrom[target_index]
    pos = int(ds1.pos[target_index])
    if window_kb < 0:
        lo, hi = 0, np.iinfo(np.int64).max
    else:
        span = int(window_kb * 1000)
        lo, hi = pos - span, pos + span

    base = np.asarray(
        kept_indices if kept_indices is not None else np.arange(ds2.variant_ct),
        dtype=np.int64,
    )
    target_rank = int(np.searchsorted(base, target_index))
    if target_rank >= base.size or int(base[target_rank]) != int(target_index):
        # The target's positional counterpart was filtered out of dataset 2, so it
        # cannot appear in the window: anchor just left of its insertion point.
        anchor = target_rank - 1
    else:
        anchor = target_rank

    if window_count > 0:
        # At most ``window_count`` kept variants on each side of the target's
        # own column (which itself counts as neither).
        first = max(0, anchor - window_count)
        last = min(base.size, anchor + window_count + 1)
        window = base[first:last]
    else:
        window = base

    same_chrom = np.array([ds2.chrom[i] == chrom for i in window], dtype=bool)
    in_range = same_chrom & (ds2.pos[window] >= lo) & (ds2.pos[window] <= hi)
    return window[in_range]


