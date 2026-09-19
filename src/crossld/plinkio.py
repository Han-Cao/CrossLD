"""Dataset resolution, sample selection, and variant index handling.

Supports PLINK 1 ``.bed/.bim/.fam`` and PLINK 2 ``.pgen/.pvar/.psam`` prefixes, read
through :mod:`pgenlib` (:class:`~pgenlib.PgenReader` + :class:`~pgenlib.PvarReader`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pgenlib

from crossld.logging_setup import LOGGER_NAME

logger = logging.getLogger(f"{LOGGER_NAME}.plinkio")


class PlinkIOError(Exception):
    """Fatal, user-facing error (reported as ``error: ...`` and exit code 2)."""


@dataclass(slots=True)
class ChromIndex:
    """Compressed chromosome labels for a dataset's variants.

    ``names`` holds the distinct chromosome labels in file order. When the file
    is chromosome-contiguous (the common PLINK layout), ``_starts`` holds the
    ``K + 1`` block boundaries and a variant's label is found by binary search;
    per-chromosome variant indices are then a slice. When a chromosome appears
    in more than one run (an interleaved file), a compact unsigned per-variant
    code array (``_codes``) is kept instead, where the codes can be converted
    back to chromosome name by ``names[code]``.

    The type is list-like over variants: ``len(chrom) == variant_ct`` and
    ``chrom[i]`` is the label of variant ``i``.

    Examples:

        >>> chrom = ChromIndex(names=["1", "2"], variant_ct=10, _starts=[0, 5, 10])
        >>> chrom[6]
        '2'
        >>> chrom.indices("1")
        array([0, 1, 2, 3, 4], dtype=uint32)
    """

    names: list[str]                        # unique chr labels in file order
    variant_ct: int
    _starts: npt.NDArray[np.uint32]         # block boundaries, used by chr-sorted files
    _codes: npt.NDArray | None = None       # per-variant chr codes, used by interleaved files
    _name_to_code: dict[str, int] | None = None   # chr name -> code, where code = nth chr
    _index_cache: dict[str, npt.NDArray[np.uint32]] | None = None

    @property
    def n_chroms(self) -> int:
        """Number of distinct chromosome labels."""

        return len(self.names)

    def __len__(self) -> int:
        return self.variant_ct

    def _code_dict(self) -> dict[str, int]:
        """Chromosome name to code lookup table, where code is the nth chr in file order."""
        if self._name_to_code is None:
            self._name_to_code = {name: i for i, name in enumerate(self.names)}
        return self._name_to_code

    def _code_of(self, chrom: str) -> int | None:
        """Get chromosome code by name"""
        return self._code_dict().get(chrom)

    def __getitem__(self, index: int) -> str:
        """Chromosome label of variant ``index``."""

        if self._codes is not None:
            return self.names[int(self._codes[index])]
        code = int(np.searchsorted(self._starts, index, side="right")) - 1
        return self.names[code]

    def indices(self, chrom: str) -> npt.NDArray[np.uint32] | None:
        """Ascending variant indexes for ``chrom``, or ``None`` when absent."""

        # TODO:
        # a hybrid strategy could help when some chr are contiguous and others not
        # for contiguous chr, do slicing; for interleaved chr, do lookup
        # However, this should introduce more cost on finding contiguous/interleaved
        # and such file is very rare in real PLINK data
        # so we leave it for now
        code = self._code_of(chrom)
        if code is None:
            return None
        # chr-sorted file, slice from start to end
        if self._codes is None:
            return np.arange(
                self._starts[code], self._starts[code + 1], dtype=np.uint32
            )
        # interleaved file, filter by chrom and return indices
        # save to cache for future use
        if self._index_cache is None:
            self._index_cache = {}
        cached = self._index_cache.get(chrom)
        if cached is None:
            cached = np.flatnonzero(self._codes == code).astype(np.uint32)
            self._index_cache[chrom] = cached
        return cached

    def codes_for(self, indices: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        """Chromosome codes for the given variant indices, in file order.

        Returned codes are ints where ``names[code]`` is the chromosome label,
        and sorting by code gives the correct chromosome order in the file.
        """
        idx = np.asarray(indices, dtype=np.uint32)
        if self._codes is not None:
            return self._codes[idx].astype(np.uint32)
        return (np.searchsorted(self._starts, idx, side="right") - 1).astype(np.uint32)


def _make_chrom_index(
    names: list[str],
    codes: npt.NDArray[np.uint32],
    runs: list[int],
    variant_ct: int,
) -> ChromIndex:
    """Build a :class:`ChromIndex` from a single-pass scan.

    ``runs`` holds the start index of every maximal run of equal codes. The file
    is contiguous exactly when there is one run per distinct label; otherwise
    the per-variant codes are kept (downcast to the smallest unsigned dtype that
    can hold ``len(names)`` codes).
    """

    starts = np.asarray(runs + [variant_ct], dtype=np.uint32)
    if len(runs) == len(names):
        return ChromIndex(names=names, variant_ct=variant_ct, _starts=starts)
    dtype = np.uint8 if len(names) <= 256 else np.uint16
    return ChromIndex(
        names=names,
        variant_ct=variant_ct,
        _starts=starts,
        _codes=codes.astype(dtype),
    )


@dataclass
class PlinkDataset:
    """A resolved PLINK 1 or PLINK 2 dataset.

    ``sample_ids`` holds every sample in file order; ``sample_subset`` holds the
    increasing indexes of the samples used for analysis (``None`` = all
    samples). Genotype reads therefore always come back in file order over
    ``sample_subset``.

    ``idx_bridge`` (dataset 1 only) converts a dataset-1 active-subset vector into
    dataset-2 active-subset order::

        samples1[sample_subset1][idx_bridge] == samples2[sample_subset2]

    REF/ALT alleles are not cached: :meth:`ref_allele` / :meth:`alt_allele` read
    them from the underlying ``.pvar``/``.bim`` reader on demand, so only the rows
    that are actually written pay for them.
    """

    # required
    prefix: str
    fmt: str  # "bfile" or "pfile"
    genotype_path: str
    variant_path: str
    sample_path: str
    sample_ids: npt.NDArray[np.str_]
    chrom: ChromIndex
    pos: npt.NDArray[np.uint32]
    ids: list[str]
    _reader: pgenlib.PgenReader
    _pvar: pgenlib.PvarReader
    # optional
    sample_subset: npt.NDArray[np.uint32] | None = None
    idx_bridge: npt.NDArray[np.uint32] | None = None
    dosage: bool = False
    # _vid_index: dict[str, int] | None = None
    _group_sort: dict[str, tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] | None = None
    _variant_mask: npt.NDArray[np.bool_] | None = None

    @property
    def variant_ct(self) -> int:
        return len(self.ids)

    @property
    def sample_ct(self) -> int:
        return len(self.sample_ids)

    @property
    def active_sample_ct(self) -> int:
        """Samples the reader returns per variant (the active subset size)."""

        if self.sample_subset is None:
            return self.sample_ct
        return len(self.sample_subset)

    @property
    def reader(self) -> pgenlib.PgenReader:
        """The active :class:`pgenlib.PgenReader`."""

        return self._reader

    # this index cannot handle duplicated ids
    # def build_vid_index(self) -> None:
    #     """Build the variant-ID lookup, idempotently."""

    #     if self._vid_index is None:
    #         self._vid_index = {vid: i for i, vid in enumerate(self.ids)}

    def indices_of(self, variant_ids: list[str]) -> tuple[npt.NDArray[np.uint32], int, int]:
        """Find variant indices for a list of variant IDs in the dataset.
           
           Return (indices, n_miss, n_duphit)
        """

        all_ids = np.asarray(self.ids)
        query_ids = np.unique(np.asarray(variant_ids))

        mask = np.isin(all_ids, query_ids)
        n_uniq_hit = np.unique(all_ids[mask]).size
        # find miss and duplicated ids
        n_miss = query_ids.size - n_uniq_hit
        n_duphit = mask.sum() - n_uniq_hit

        return np.flatnonzero(mask).astype(np.uint32), n_miss, n_duphit
    

    def ref_allele(self, index: int) -> str:
        """Decoded REF allele for a variant, read on demand from the pvar/bim reader."""

        return self._pvar.get_allele_code(index, 0).decode()

    def alt_allele(self, index: int) -> str:
        """Decoded ALT allele for a variant, read on demand from the pvar/bim reader."""

        return self._pvar.get_allele_code(index, 1).decode()

    def apply_variant_filter(self, mask: npt.NDArray[np.bool_]) -> None:
        """Restrict position/interval queries to the kept variants.

        ``mask`` is a boolean array over **all** variants in file order. It is
        applied once and reused by every :meth:`get_sorted_index` query, so
        downstream code (window selection) sees only kept variants.
        """

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.variant_ct,):
            raise PlinkIOError(
                f"variant filter has shape {mask.shape}, expected ({self.variant_ct},)"
            )
        self._variant_mask = mask
        logger.debug(
            "Kept %d of %d variants after variant filter",
            int(mask.sum()),
            self.variant_ct,
        )

    def sort_pos_by_chrom(self) -> None:
        """Build a mapping: chrom -> (positions sorted, variant indices for those positions).

        Each chromosome's variants come from :meth:`ChromIndex.indices` (a slice
        for chromosome-contiguous files), so no per-variant Python container is
        built. The stable ``argsort`` is skipped when the block is already
        position-sorted, which is the usual PLINK layout.
        """

        if self._group_sort is not None:
            return

        group_sort: dict[str, tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = {}
        for name in self.chrom.names:
            idx = self.chrom.indices(name)
            # for name in self.chrom.names, must be non-empty idx
            assert idx is not None
            pos = self.pos[idx]
            if pos.size > 1 and not bool(np.all(pos[:-1] <= pos[1:])):
                order = np.argsort(pos, kind="stable")
                pos = pos[order]
                idx = idx[order]
            group_sort[name] = (pos, idx)

        self._group_sort = group_sort

    def get_sorted_index(
        self, chrom: str, filtered: bool = True
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]] | None:
        """Position-sorted ``(pos, variant_idx)`` for ``chrom``.

        When a variant filter has been applied (see
        :meth:`apply_variant_filter`) and ``filtered`` is true, only kept
        variants are returned. Pass ``filtered=False`` to query the full file
        (used while building the filter itself).
        """

        if self._group_sort is None:
            logger.warning("Unexpected behavior: query sorted index before building _group_sort")
            self.sort_pos_by_chrom()
        entry = self._group_sort.get(chrom)
        if entry is None:
            return None
        pos, idx = entry
        if filtered and self._variant_mask is not None:
            keep = self._variant_mask[idx]
            if not keep.all():
                return pos[keep], idx[keep]
        return pos, idx

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
    """Read IIDs from a ``.psam`` file.

    A ``#FID``/``#IID`` header is honoured (the IID column is located by name);
    a file without a header is parsed as a ``.fam`` file.
    """

    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header.startswith(("#FID", "#IID")):
            fields = header.lstrip("#").split()
            try:
                iid_idx = fields.index("IID")
            except ValueError as exc:
                raise PlinkIOError(
                    f"{path}: .psam header has no IID column: {header!r}"
                ) from exc
        elif header.startswith("#"):
            raise PlinkIOError(f"{path}: malformed .psam header: {header!r}")
        else:
            return _parse_fam(path)

        for n_line, line in enumerate(handle, start=2):
            stripped = line.strip()
            if not stripped:
                raise PlinkIOError(f"{path}: empty sample line in line {n_line}")
            fields = stripped.split()
            if iid_idx >= len(fields):
                raise PlinkIOError(f"{path}: malformed sample line: {stripped!r}")
            ids.append(fields[iid_idx])
    return np.array(ids)


def parse_sample(path: str, fmt: str) -> npt.NDArray[np.str_]:
    """Read sample IDs from a ``.fam`` or ``.psam`` file."""

    if fmt == "bfile":
        samples = _parse_fam(path)
    elif fmt == "pfile":
        samples = _parse_psam(path)
    else:
        raise PlinkIOError(f"unknown dataset type {fmt!r}")

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

    missing = None
    if keep_samples is not None:
        samples_ld = np.intersect1d(dataset_overlap, keep_samples, assume_unique=True)
        missing = np.setdiff1d(keep_samples, dataset_overlap, assume_unique=True)
    else:
        samples_ld = dataset_overlap

    logger.info(
        "Parsed samples: dataset1=%d dataset2=%d shared=%d%s",
        len(samples1),
        len(samples2),
        len(dataset_overlap),
        "" if keep_samples is None else f" keep={len(samples_ld)} missing={len(missing)}",
    )

    if len(samples_ld) < 3:
        raise PlinkIOError("At least 3 samples are required for LD analysis!")

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
    keep_sample_file: str | None = None,
) -> tuple[PlinkDataset, PlinkDataset]:
    """Load dataset 1 and dataset 2, map analysis samples in dataset 1 to dataset 2."""

    # select samples for analysis
    sample_path1 = _resolve_files(prefix1, fmt1)[2]
    sample_path2 = _resolve_files(prefix2, fmt2)[2]
    samples1 = parse_sample(sample_path1, fmt1)
    samples2 = parse_sample(sample_path2, fmt2)
    if keep_sample_file is None:
        samples_ld = select_samples(samples1, samples2)
    else:
        samples_ld = select_samples(samples1, samples2, read_keep_file(keep_sample_file))

    # find subset idx; np.flatnonzero keeps them increasing, which pgenlib requires
    sample_subset1 = np.flatnonzero(np.isin(samples1, samples_ld)).astype(np.uint32)
    sample_subset2 = np.flatnonzero(np.isin(samples2, samples_ld)).astype(np.uint32)

    # map subset 1 to subset 2 for the same samples if they have different order
    # since more query on dataset 2, we use its order to speed up mapping
    # idx_bridge is samples1[sample_subset1][idx_bridge] == samples2[sample_subset2]
    if np.array_equal(samples1[sample_subset1], samples2[sample_subset2]):
        idx_bridge = None
    else:
        subset1_id2pos = {sid: i for i, sid in enumerate(samples1[sample_subset1].tolist())}
        idx_bridge = np.array(
            [subset1_id2pos[sid] for sid in samples2[sample_subset2].tolist()], dtype=np.uint32
        )

    # load dataset 1
    # dataset 1 always query by id, build index here
    logger.info("Loading dataset 1: %s", prefix1)
    ds1 = load_dataset(
        prefix1, fmt1, samples1, dosage1, sample_subset1, idx_bridge,
    )

    # load dataset 2
    # dataset 2 always query by position, sort it here
    # varfilter will also query by id, so build vid index later when needed
    logger.info("Loading dataset 2: %s", prefix2)
    ds2 = load_dataset(
        prefix2, fmt2, samples2, dosage2, sample_subset2,
    )
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
    downstream reads honor it; otherwise the reader spans all samples. REF/ALT
    alleles are not read here; use :meth:`PlinkDataset.ref_allele` /
    :meth:`PlinkDataset.alt_allele` when they are needed.
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

    chrom_names: list[str] = []           # unique chr labels in file order
    byte_to_code: dict[bytes, int] = {}   # unique chr bytes to chr code
    chr_runs: list[int] = []              # list of chr start indecies
    chr_code_arr = np.empty(variant_ct, dtype=np.uint32)  # per-variant chr code, every element will be reset
    pos: list[int] = []
    ids: list[str] = []
    last_code = -1
    for i in range(variant_ct):
        chr_byte = pvar.get_variant_chrom(i)
        chr_code = byte_to_code.get(chr_byte)
        # process a new chromosome
        if chr_code is None:
            chr_code = len(chrom_names)
            byte_to_code[chr_byte] = chr_code
            chrom_names.append(chr_byte.decode())
        chr_code_arr[i] = chr_code
        if chr_code != last_code:
            chr_runs.append(i)
            last_code = chr_code
        pos.append(pvar.get_variant_pos(i))
        ids.append(pvar.get_variant_id(i).decode())
    pos_array = np.array(pos, dtype=np.uint32)
    chrom_index = _make_chrom_index(chrom_names, chr_code_arr, chr_runs, variant_ct)

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
        chrom=chrom_index,
        pos=pos_array,
        ids=ids,
        sample_subset=sample_subset,
        idx_bridge=idx_bridge,
        dosage=dosage,
        _reader=reader,
        _pvar=pvar,
    )

    logger.info(
        "format=%s variants=%d samples=%d",
        fmt,
        variant_ct,
        len(sample_ids),
    )
    logger.debug(
        "chroms=%s",
        chrom_index.names,
    )
    return ds


def read_ld_snps(ld_snps_str: str) -> list[str]:
    """Resolve ``--ld-snps`` (comma-separated IDs)."""

    ids: list[str] = []
    for vid in ld_snps_str.split(","):
        vid = vid.strip()
        if not vid:
            continue
        ids.append(vid)
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
    """Resolve the sorted dataset-1 target variants, warning about anything unmatched."""

    ids: list[str] = []
    indices: list[int] = []
    if single is not None:
        ids.append(single)
    if snps is not None:
        ids += read_ld_snps(snps)
    if snp_list is not None:
        ids += read_ld_snp_list(snp_list)

    indices, n_miss, n_duphit = ds.indices_of(ids)
    if n_miss > 0:
        logger.warning("%d target LD variants not found in dataset 1", n_miss)
    if n_duphit > 0:
        logger.warning("%d target LD variants have duplicated records in dataset 1", n_duphit)

    # sort by chrom code (i.e. file order) and position
    indices = indices[np.lexsort((ds.pos[indices], ds.chrom.codes_for(indices)))]

    return indices.tolist()


def candidate_indices(
    ds1: PlinkDataset,
    ds2: PlinkDataset,
    target_index: int,
    window_kb: float,
    window_count: int,
) -> npt.NDArray[np.uint32]:
    """Dataset-2 variant indexes within the target's window.

    ``window_count`` is the maximum number of dataset-2 variants kept, chosen as
    the nearest by absolute position distance (``0`` means no limit). Only
    variants surviving the dataset's variant filter are eligible (see
    :meth:`PlinkDataset.apply_variant_filter`). The returned indexes are unique
    and in position order (the fixed order from :meth:`PlinkDataset.get_sorted_index`),
    which is what the block cache relies on; pgenlib currently accepts such
    (not necessarily index-ascending) input.
    """

    # Important:
    # the returned index is sorted only if dataset 2 is postion-sorted
    # this is OK with current pgenlib but may break in the future
    # the current implementation is important for cache input
    # if pgenlib no longer support unsorted index to query
    # we should disable cache and output sorted index

    chrom = ds1.chrom[target_index]
    pos_a = int(ds1.pos[target_index])

    entry = ds2.get_sorted_index(chrom)
    if entry is None:
        return np.empty(0, dtype=np.uint32)
    pos_k, idx_k = entry

    if window_kb >= 0 and pos_k.size:
        span = window_kb * 1000
        lo = int(np.searchsorted(pos_k, pos_a - span, side="left"))
        hi = int(np.searchsorted(pos_k, pos_a + span, side="right"))
        pos_k = pos_k[lo:hi]
        idx_k = idx_k[lo:hi]

    if window_count > 0 and idx_k.size > window_count:
        # Cast to int64 before subtracting: pos arrays are uint32, and an
        # unsigned underflow would report variants left of the target as
        # ~2**32 bp away instead of their true distance.
        distance = np.abs(pos_k.astype(np.int64) - pos_a)
        keep_idx = np.argsort(distance, kind="stable")[:window_count]
        # sort to make sure the returned index is position-sorted
        keep_idx.sort()
        return idx_k[keep_idx]

    return idx_k


def _windows_may_overlap(
    ds1: PlinkDataset,
    index_a: int,
    index_b: int,
    window_kb: float,
) -> bool:
    """Whether two dataset-1 targets can share dataset-2 variants.

    Two targets can only share a candidate when they are on the same
    chromosome and their ``+/- window_kb`` windows overlap, i.e. their
    positional distance is at most twice the window radius. The comparison is
    closed so a variant sitting exactly on the shared window edge is not
    missed; ``window_kb == 0`` therefore only allows equal positions.
    """

    if ds1.chrom[index_a] != ds1.chrom[index_b]:
        return False
    span = window_kb * 1000.0
    return abs(int(ds1.pos[index_a]) - int(ds1.pos[index_b])) <= 2.0 * span


def iter_target_candidates(
    ds1: PlinkDataset,
    ds2: PlinkDataset,
    target_indices: list[int],
    window_kb: float,
    window_count: int,
) -> Iterator[tuple[int, npt.NDArray[np.uint32], npt.NDArray[np.uint32]]]:
    """Yield ``(target_index, candidates, keep_for_next)`` for a target run.

    ``candidates`` is :func:`candidate_indices` for the current target. Windows
    are computed one step ahead, so on the next iteration ``keep_for_next``
    becomes the next target's ``candidates`` and every window is computed
    exactly once.

    ``keep_for_next`` equals the next target's candidate array when the two
    windows may overlap (same chromosome and positional distance at most twice
    the window radius, see :func:`_windows_may_overlap`); otherwise it is an
    empty array. It exists only to drive the dataset-2 read cache: candidates
    listed there are the only ones worth keeping for the next target.
    """

    empty = np.empty(0, dtype=np.uint32)
    if not target_indices:
        return

    next_candidates = candidate_indices(
        ds1, ds2, target_indices[0], window_kb, window_count
    )
    for position, index in enumerate(target_indices):
        candidates = next_candidates
        if position + 1 < len(target_indices):
            next_index = target_indices[position + 1]
            next_candidates = candidate_indices(
                ds1, ds2, next_index, window_kb, window_count
            )
            keep = (
                next_candidates
                if _windows_may_overlap(ds1, index, next_index, window_kb)
                else empty
            )
        else:
            next_candidates = empty
            keep = empty
        yield index, candidates, keep


__all__ = [
    "ChromIndex",
    "PlinkDataset",
    "PlinkIOError",
    "candidate_indices",
    "iter_target_candidates",
    "load_dataset",
    "load_two_datasets",
    "parse_sample",
    "read_keep_file",
    "read_ld_snp_list",
    "read_ld_snps",
    "resolve_targets",
    "select_samples",
]
