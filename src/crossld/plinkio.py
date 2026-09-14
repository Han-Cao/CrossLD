"""Dataset resolution, sample selection, and variant index handling.

Supports PLINK 1 ``.bed/.bim/.fam`` and PLINK 2 ``.pgen/.pvar/.psam`` prefixes, read
through :mod:`pgenlib` (:class:`~pgenlib.PgenReader` + :class:`~pgenlib.PvarReader`).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pgenlib

from crossld.logging_setup import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.plinkio")


class PlinkIOError(Exception):
    """Fatal, user-facing error (reported as ``error: ...`` and exit code 2)."""


@dataclass
class PlinkDataset:
    """A resolved PLINK 1 or PLINK 2 dataset."""

    prefix: str
    fmt: str  # "bfile" or "pfile"
    genotype_path: str
    variant_path: str
    sample_path: str
    sample_ids: npt.NDArray[np.str_]
    sample_subset: npt.NDArray[np.uint32] = None
    dosage: bool = False
    chrom: list[str] = field(default_factory=list)
    pos: npt.NDArray[np.uint32]
    ids: list[str]
    ref: list[str] = field(default_factory=list)
    alt: list[str] = field(default_factory=list)
    idx_bridge: npt.NDArray[np.uint32] = None
    _reader: pgenlib.PgenReader
    _pvar: pgenlib.PvarReader
    _vid_index: dict[str, int] | None = None
    _group_sort: dict[str, tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] | None = None

    def __post_init__(self) -> None:
        if self.sample_subset is None:
            self.active_sample_ct = self.sample_ct
        else:
            self.active_sample_ct = len(self.sample_subset)

    def build_vid_index(self) -> None:
        if self.ids and not self._vid_index:
            self._vid_index = {vid: i for i, vid in enumerate(self.ids)}

    def index_of(self, variant_id: str) -> int | None:
        if self._vid_index is None:
            return None
        return self._vid_index.get(variant_id)

    @property
    def unique_ids(self) -> set[str]:
        if self._vid_index is None:
            logger.debug("Unexpected behavior: query unique variant ID before building _vid_index")
            return set(self.ids)
        else:
            return self._vid_index.keys()

    def sort_pos_by_chrom(self) -> None:
        """Build a mapping: chrom -> (sorted positions, variant indices for sorted positions)."""
        if self._group_sort is not None:
            return

        by_chrom: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.chrom):
            by_chrom[c].append(i)

        group_sort: dict[str, tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = {}
        for c, idx_list in by_chrom.items():
            idx = np.asarray(idx_list, dtype=np.uint32)
            order = np.argsort(self.pos[idx], kind="stable")
            group_sort[c] = (self.pos[idx[order]], idx[order])

        self._group_sort = group_sort

    def get_sorted_index(self, chrom: str) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        if self._group_sort is None:
            return None
        return self._group_sort.get(chrom)

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


def _parse_fam(path: str) -> npt.NDArray[np.str_]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for n_line, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise PlinkIOError(f"{path}: empty sample line in line {n_line}")
            fields = stripped.split()
            if len(fields) < 6:
                raise PlinkIOError(f"{path}: malformed sample line: {stripped!r}")
            ids.append(fields[1])
    return np.array(ids)


def _parse_psam(path: str) -> npt.NDArray[np.str_]:
    """Read IIDs from a ``.psam`` file."""

    ids: list[str] = []
    # check header for the first line
    # if starts with #FID, find idx of IID
    # if starts with #IID, set iid_idx to 0
    # if starts with #other, raise error
    # if no header, do _parse_fam
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header.startswith("#FID"):
            fields = header.split()
            iid_idx = fields.index("IID")
        elif header.startswith("#IID"):
            iid_idx = 0
        elif header.startswith("#"):
            raise PlinkIOError(f"{path}: malformed .psam header: {header!r}")
        else:
            return _parse_fam(path)

    with open(path, "r", encoding="utf-8") as handle:
        for n_line, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise PlinkIOError(f"{path}: empty sample line in line {n_line}")
            fields = stripped.split()
            ids.append(fields[iid_idx])
    return np.array(ids)


def parse_sample(path: str, type: str) -> npt.NDArray[np.str_]:
    """Read sample IDs from a ``.fam`` or ``.psam`` file."""

    if type == "bfile":
        samples = _parse_fam(path)
    elif type == "pfile":
        samples = _parse_psam(path)
    else:
        raise PlinkIOError(f"unknown dataset type {type!r}")

    # check empty file
    if len(samples) == 0:
        raise PlinkIOError(f"empty sample file {path}")

    # check sample name uniqueness
    if len(samples) != len(np.unique(samples)):
        raise PlinkIOError(f"duplicate sample IDs in {path}")
    return samples


def read_keep_file(path: str) -> npt.NDArray[np.str_]:
    """Read sample IDs from a ``--keep`` file. Ignore empty lines or lines starts with ``#``."""

    keep_samples: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for n_line, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 1:
                raise PlinkIOError(
                    f"--keep {path}: line {n_line} has {len(fields)} columns; a "
                    "single-column IID file is required"
                )
            keep_samples.append(fields[0])
    logger.debug("keep %s: %d requested sample(s)", path, len(keep_samples))
    return np.unique(np.array(keep_samples))


def select_samples(
    samples1: npt.NDArray[np.str_],
    samples2: npt.NDArray[np.str_],
    keep_samples: npt.NDArray[np.str_] | None = None,
) -> npt.NDArray[np.str_]:
    """Find overlapping samples used for LD analysis."""

    # check uniqueness of samples1 and samples2
    dataset_overlap = np.intersect1d(samples1, samples2, assume_unique=True)
    
    if keep_samples is not None:
        samples_ld = np.intersect1d(dataset_overlap, keep_samples, assume_unique=True)
        missing = np.setdiff1d(keep_samples, dataset_overlap, assume_unique=True)
    else:
        samples_ld = dataset_overlap
        missing = None

    logger.info(
        "samples: dataset1=%d dataset2=%d shared=%d%s",
        len(samples1),
        len(samples2),
        len(dataset_overlap),
        "" if keep_samples is None else f" keep={len(samples_ld)} missing={len(missing)}",
    )

    if len(samples_ld) < 3:
        raise PlinkIOError(
            "At least 3 samples are required for LD analysis!"
        )

    return samples_ld


def _get_file(candidate: str) -> str | None:
    """Return path to a file, returning ``None`` if not found."""

    if os.path.exists(candidate):
        return candidate
    return None


def _resolve_files(prefix: str, fmt: str) -> tuple[str, str, str]:
    """Resolve PLINK bfile or pfile."""

    if fmt == "bfile":
        bed = _get_file(f"{prefix}.bed")
        bim = _get_file(f"{prefix}.bim")
        fam = _get_file(f"{prefix}.fam")
        missing = [
            name
            for name, path in ((".bed", bed), (".bim", bim), (".fam", fam))
            if path is None
        ]
        if missing:
            raise PlinkIOError(
                f"--bfile {prefix}: missing file(s) {', '.join(prefix + m for m in missing)}"
            )
        return bed, bim, fam
    
    elif fmt == "pfile":
        pgen = _get_file(f"{prefix}.pgen")
        pvar = _get_file(f"{prefix}.pvar")
        psam = _get_file(f"{prefix}.psam")
        missing = [
            name
            for name, path in ((".pgen", pgen), (".pvar", pvar), (".psam", psam))
            if path is None
        ]
        if missing:
            raise PlinkIOError(
                f"--pfile {prefix}: missing file(s) {', '.join(prefix + m for m in missing)}"
            )
        return pgen, pvar, psam

    else:
        raise PlinkIOError(f"unknown format {fmt!r}")


def load_two_datasets(
    prefix1: str, fmt1: str, dosage1: bool,
    prefix2: str, fmt2: str, dosage2: bool,
    keep: str | None = None
) -> tuple[PlinkDataset, PlinkDataset]:
    """Load dataset 1 and dataset 2, map analysis samples in dataset 2 to dataset 1."""

    # select samples for analysis
    samples1 = parse_sample(prefix1, fmt1)
    samples2 = parse_sample(prefix2, fmt2)
    if keep is None:
        samples_ld = select_samples(samples1, samples2)
    else:
        samples_ld = select_samples(samples1, samples2, read_keep_file(keep))

    # find subset idx
    sample_subset1 = np.flatnonzero(np.isin(samples1, samples_ld)).astype(np.uint32)
    sample_subset2 = np.flatnonzero(np.isin(samples2, samples_ld)).astype(np.uint32)

    # map subset 1 to subset 2 for same samples
    # since more query on dataset 2, we use its order to speed up mapping
    # idx_bridge is samples1[sample_subset1][idx_bridge] == samples2[sample_subset2]
    subset1_id2pos = {id: i for i, id in enumerate(samples1[sample_subset1].tolist())}
    idx_bridge = np.array([subset1_id2pos[id] for id in samples2[sample_subset2].tolist()], dtype=np.uint32)

    # load dataset 1
    # dataset 1 alwasy query by id, build index here
    logger.info("loading dataset 1: %s", prefix1)
    ds1 = load_dataset(prefix1, fmt1, samples1, dosage1, sample_subset1, idx_bridge)
    ds1.build_vid_index()

    # load dataset 2
    # dataset 2 always query by position, sort it here
    # varfilter will also query by id, so build vid index later when needed
    logger.info("loading dataset 2: %s", prefix2)
    ds2 = load_dataset(prefix2, fmt2, samples2, dosage2, sample_subset2)
    ds2.sort_pos_by_chrom()

    return ds1, ds2


def load_dataset(
    prefix: str,
    fmt: str,
    sample_ids: npt.NDArray[np.str_],
    dosage: bool = False,
    sample_subset: npt.NDArray[np.uint32] | None = None,
    idx_bridge: npt.NDArray[np.uint32] | None = None
) -> PlinkDataset:
    """Open a dataset and read its variant metadata.

    When ``sample_subset`` is given the reader is created with that subset so that
    downstream reads honor it; otherwise the reader spans all samples (the subset
    can be applied later with :meth:`pgenlib.PgenReader.change_sample_subset`).
    """

    genotype_path, variant_path, sample_path = _resolve_files(prefix, fmt)

    pvar = pgenlib.PvarReader(variant_path.encode())
    variant_ct = pvar.get_variant_ct()
    max_allele_ct = pvar.get_max_allele_ct()
    if max_allele_ct > 2:
        raise PlinkIOError(
            f"{prefix}: multiallelic variants are not supported "
            f"(max allele count {max_allele_ct})"
        )

    chrom: list[str] = []
    pos = list[int] = []
    ids: list[str] = []
    ref: list[str] = []
    alt: list[str] = []
    for i in range(variant_ct):
        # TODO: keep using bytes here? only decode when need to print
        chrom.append(pvar.get_variant_chrom(i).decode())
        pos.append(pvar.get_variant_pos(i))
        ids.append(pvar.get_variant_id(i).decode())
        ref.append(pvar.get_allele_code(i, 0).decode())
        alt.append(pvar.get_allele_code(i, 1).decode())
    pos = np.array(pos, dtype=np.uint32)

    is_bed = genotype_path.endswith(".bed")
    reader_kwargs: dict = {"sample_subset": sample_subset}
    if is_bed:
        reader_kwargs["raw_sample_ct"] = len(sample_ids)
        reader_kwargs["variant_ct"] = variant_ct
    else:
        reader_kwargs["pvar"] = pvar
    reader = pgenlib.PgenReader(genotype_path.encode(), **reader_kwargs)
    if reader.get_raw_sample_ct() != len(sample_ids):
        raise PlinkIOError(
            f"{prefix}: {sample_path} describes {len(sample_ids)} samples but the "
            f"genotype file has {reader.get_raw_sample_ct()}"
        )

    ds = PlinkDataset(
        prefix=prefix,
        fmt=fmt,
        genotype_path=genotype_path,
        variant_path=variant_path,
        sample_path=sample_path,
        sample_ids=sample_ids,
        sample_subset=sample_subset,
        dosage=dosage,
        chrom=chrom,
        pos=pos,
        ids=ids,
        ref=ref,
        alt=alt,
        idx_bridge=idx_bridge,
        _reader=reader,
        _pvar=pvar,
    )

    logger.info(
        "dataset: format=%s variants=%d samples=%d genotype=%s",
        fmt,
        variant_ct,
        len(sample_ids),
        genotype_path,
    )
    logger.debug(
        "dataset: variant-meta=%s sample-meta=%s chroms=%s",
        variant_path,
        sample_path,
        sorted(set(chrom)),
    )
    return ds


def read_ld_snps(ld_snps_str: str) -> list[int]:
    """Resolve ``--ld-snps`` (comma-separated IDs)."""

    ids: list[str] = []
    for id in ld_snps_str.split(","):
        id = id.strip()
        if not id:
            continue
        ids.append(id)
    return ids


def read_ld_snp_list(path: str) -> list[str]:
    """Read ``--ld-snp-list``: one variant ID per line, ``#`` comments allowed."""

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.append(stripped)
    return ids


def resolve_targets(
    ds: PlinkDataset,
    single: str | None = None,
    snps: str | None = None,
    snp_list: str | None = None,
) -> list[int]:
    """Resolve the dataset-1 target variants, warning about anything unmatched."""

    ids = list[str] = []
    indices: list[int] = []
    seen: set[int] = set()
    if single is not None:
        ids.append(single)
    if snps is not None:
        ids += read_ld_snps(snps)
    if snp_list is not None:
        ids += read_ld_snp_list(snp_list)

    n_miss = 0
    for vid in ids:
        index = ds.index_of(vid)
        if index is None:
            n_miss += 1
            continue
        if index not in seen:
            seen.add(index)
            indices.append(index)
    if n_miss > 0:
        logger.warning("%d target LD variants not found in dataset 1", n_miss)
    return indices


def candidate_indices(
    ds1: PlinkDataset,
    ds2: PlinkDataset,
    target_index: int,
    window_kb: float,
    window_count: int,
    kept_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Dataset-2 variant indexes within the target's window.

    ``window_count`` is the maximum number of dataset-2 variants (by file order
    over the post-filter set) on both side of the target; ``0`` means no limit.
    """
    # TODO:
    # make sure returned index is sorted and deduplicated
    # which may be required by pgenlib in the future
    chrom = ds1.chrom[target_index]
    pos = int(ds1.pos[target_index])
    if window_kb < 0:
        lo, hi = 0, np.iinfo(np.uint32).max
    else:
        span = window_kb * 1000
        lo, hi = pos - span, pos + span

    base = np.asarray(
        kept_indices if kept_indices is not None else np.arange(ds2.variant_ct),
        dtype=np.uint32,
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

