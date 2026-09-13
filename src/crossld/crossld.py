"""Command-line interface and run orchestration for crossld."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .blocks import BlockStats, BlockReader
from .ld import SKIP_MISSING, SKIP_SAME_ID, SKIP_ZERO_VARIANCE, major_allele_flipped
from .ld import pearson_pairwise
from .logging_setup import LOGGER_NAME, configure_logging
from .plinkio import (
    CrossLDError,
    Dataset,
    candidate_indices,
    load_dataset,
    resolve_targets,
    select_samples,
)
from .results import write_header, write_row
from .varfilter import VariantFilter

logger = logging.getLogger(f"{LOGGER_NAME}.crossld")

DEFAULT_LD_WINDOW = 1000
DEFAULT_LD_WINDOW_KB = 1000.0
DEFAULT_LD_WINDOW_R2 = 0.2
DEFAULT_BLOCK_SIZE = 16384


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that reports errors through the module logger."""

    def error(self, message: str):  # pragma: no cover - exercised via CLI tests
        raise CrossLDError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="crossLD",
        description=(
            "Cross-dataset linkage disequilibrium: correlate dataset-1 target "
            "variants against nearby variants in dataset 2."
        ),
    )

    ds1 = parser.add_mutually_exclusive_group(required=True)
    ds1.add_argument("--bfile1", metavar="PREFIX", help="PLINK 1 .bed/.bim/.fam prefix (dataset 1)")
    ds1.add_argument("--pfile1", metavar="PREFIX", help="PLINK 2 .pgen/.pvar/.psam prefix (dataset 1)")

    ds2 = parser.add_mutually_exclusive_group(required=True)
    ds2.add_argument("--bfile2", metavar="PREFIX", help="PLINK 1 .bed/.bim/.fam prefix (dataset 2)")
    ds2.add_argument("--pfile2", metavar="PREFIX", help="PLINK 2 .pgen/.pvar/.psam prefix (dataset 2)")

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--ld-snp", metavar="ID", help="single dataset-1 target variant ID")
    target.add_argument(
        "--ld-snps",
        metavar="LIST",
        help="comma-separated dataset-1 variant IDs and/or CHR:START-END ranges",
    )
    target.add_argument(
        "--ld-snp-list",
        metavar="FILE",
        help="file of dataset-1 target variant IDs (one per line)",
    )

    parser.add_argument("--keep", metavar="FILE", help="single-column (or '#IID' headed) sample IID file")

    parser.add_argument("--extract", metavar="FILE", help="keep only these dataset-2 variant IDs")
    parser.add_argument("--exclude", metavar="FILE", help="drop these dataset-2 variant IDs")
    parser.add_argument(
        "--extract-bed",
        metavar="FILE",
        help="keep only dataset-2 variants in these 0-based half-open CHROM/START/END intervals",
    )
    parser.add_argument(
        "--exclude-bed",
        metavar="FILE",
        help="drop dataset-2 variants in these 0-based half-open CHROM/START/END intervals",
    )

    parser.add_argument(
        "--ld-window",
        type=int,
        default=DEFAULT_LD_WINDOW,
        metavar="N",
        help=(
            "max number of dataset-2 variants on each side of the target "
            "(0 disables the count limit; default %(default)s)"
        ),
    )
    parser.add_argument(
        "--ld-window-kb",
        type=float,
        default=DEFAULT_LD_WINDOW_KB,
        metavar="KB",
        help="max distance in kb between the target and a dataset-2 variant (default %(default)s)",
    )
    parser.add_argument(
        "--ld-window-r2",
        type=float,
        default=DEFAULT_LD_WINDOW_R2,
        metavar="R2",
        help=(
            "only report pairs with r^2 at least this value "
            "(0 disables the filter; default %(default)s)"
        ),
    )

    dosage = parser.add_mutually_exclusive_group()
    dosage.add_argument("--dosage", action="store_true", help="use dosages for both datasets")
    dosage.add_argument("--dosage1", action="store_true", help="use dosages for dataset 1")
    dosage.add_argument("--dosage2", action="store_true", help="use dosages for dataset 2")

    parser.add_argument(
        "--ref-based",
        action="store_true",
        help="orient r to REF alleles instead of each variant's major allele",
    )
    parser.add_argument(
        "--r",
        dest="with_r",
        action="store_true",
        help="add a signed UNPHASED_R column before UNPHASED_R2",
    )
    parser.add_argument("--out", metavar="FILE", default=None, help="output file (default: stdout)")
    parser.add_argument(
        "--sparse",
        choices=("auto", "never", "require"),
        default="auto",
        help="sparse genotype/dosage reads (default %(default)s)",
    )
    parser.add_argument(
        "--sample-id",
        choices=("iid", "fid-iid"),
        default="iid",
        help="sample IDs used for matching, for .fam-based inputs (default %(default)s)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        metavar="N",
        help="variants read per block (default %(default)s)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging, including per-block statistics",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="WARNING logging only",
    )
    parser.add_argument("--version", action="version", version=f"crossLD {__version__}")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.ld_window < 0:
        errors.append("--ld-window must be 0 (no limit) or a positive integer")
    if args.ld_window_kb < 0:
        errors.append("--ld-window-kb must be >= 0")
    if not 0.0 <= args.ld_window_r2 <= 1.0:
        errors.append("--ld-window-r2 must be between 0 and 1 (0 disables the filter)")
    if args.block_size < 1:
        errors.append("--block-size must be a positive integer")
    if errors:
        raise CrossLDError("; ".join(errors))


@dataclass
class _TargetProgress:
    index: int
    variant_id: str
    chrom: str
    pos: int
    candidates: int = 0
    kept: int = 0
    below_r2: int = 0
    skipped: int = 0
    blocks: int = 0
    elapsed: float = 0.0
    skip_reasons: dict = field(default_factory=dict)


def run(args: argparse.Namespace) -> int:
    validate_args(args)

    variant_filter = VariantFilter(
        extract=args.extract,
        exclude=args.exclude,
        extract_bed=args.extract_bed,
        exclude_bed=args.exclude_bed,
    )
    variant_filter.validate()

    fid_iid = args.sample_id == "fid-iid"
    dosage1 = args.dosage or args.dosage1
    dosage2 = args.dosage or args.dosage2
    # With --r the coding allele is reported for each side, so the column name
    # depends on the orientation mode.
    allele_label = "REF" if args.ref_based else "MAJ"

    ds1 = load_dataset(
        args.bfile1 or args.pfile1,
        "dataset1",
        "bfile" if args.bfile1 else "pfile",
        dosage=dosage1,
        fid_iid=fid_iid,
    )
    ds2 = load_dataset(
        args.bfile2 or args.pfile2,
        "dataset2",
        "bfile" if args.bfile2 else "pfile",
        dosage=dosage2,
        fid_iid=fid_iid,
    )

    try:
        mask = (
            variant_filter.build_mask(ds2)
            if variant_filter.active
            else np.ones(ds2.variant_ct, dtype=bool)
        )
        kept_indices = np.flatnonzero(mask).astype(np.int64)

        subset1, subset2, _ = select_samples(ds1, ds2, keep=args.keep, fid_iid=fid_iid)
        ds1.set_sample_subset(subset1)
        ds2.set_sample_subset(subset2)

        targets = resolve_targets(
            ds1,
            single=args.ld_snp,
            snps=args.ld_snps,
            snp_list=args.ld_snp_list,
        )
        if not targets:
            logger.warning("no dataset-1 target variants resolved; nothing to do")
            logger.info("done: 0 targets, 0 pairs written")

        with _open_output(args.out) as out:
            write_header(out, args.with_r, allele_label)
            written = _process_targets(
                args, ds1, ds2, targets, kept_indices, out, allele_label
            )
            out.flush()
    finally:
        ds1.close()
        ds2.close()

    logger.info("done: %d target(s), %d pair(s) written", len(targets), written)
    return 0


def _open_output(path: str | None):
    if path is None:
        return _Stdout()
    return open(path, "w", encoding="utf-8")


class _Stdout:
    def __enter__(self):
        return sys.stdout

    def __exit__(self, *exc):
        return False


def _process_targets(
    args: argparse.Namespace,
    ds1: Dataset,
    ds2: Dataset,
    targets: list[int],
    kept_indices: np.ndarray,
    out,
    allele_label: str = "MAJ",
) -> int:
    block_size = args.block_size
    total = len(targets)
    written_total = 0
    r2_threshold = args.ld_window_r2
    reader1 = BlockReader(ds1, sparse="never", dosage=bool(ds1._dosage))
    reader2 = BlockReader(ds2, sparse=args.sparse, dosage=bool(ds2._dosage))

    # Orient each dataset-1 target once; the values do not depend on the pairing.
    # The coding allele is recorded alongside so --r can report it.
    target_vectors: dict[int, np.ndarray] = {}
    target_alleles: dict[int, str] = {}
    for index in targets:
        matrix, _ = reader1.read([index])
        vector = matrix[0]
        if args.ref_based:
            target_alleles[index] = ds1.ref[index]
        elif major_allele_flipped(vector):
            target_alleles[index] = ds1.ref[index]
            vector = _to_ref(vector)
        else:
            target_alleles[index] = ds1.alt[index]
        target_vectors[index] = vector

    for order, index in enumerate(targets, start=1):
        start_time = time.perf_counter()
        progress = _TargetProgress(
            index=index,
            variant_id=ds1.ids[index],
            chrom=ds1.chrom[index],
            pos=int(ds1.pos[index]),
        )
        candidates = candidate_indices(
            ds1,
            ds2,
            index,
            args.ld_window_kb,
            args.ld_window,
            kept_indices=kept_indices,
        )
        progress.candidates = int(len(candidates))
        rows: list[tuple] = []
        target = target_vectors[index]

        if len(candidates) == 0:
            logger.debug(
                "target %s: window holds no dataset-2 variants", ds1.ids[index]
            )

        block_total = max(1, -(-len(candidates) // block_size)) if len(candidates) else 0
        for block_no, (start, end) in enumerate(_block_slices(len(candidates), block_size), start=1):
            chunk = candidates[start:end]
            matrix, block_stats = reader2.read(chunk)
            # The coding allele per dataset-2 variant, for the --r allele column.
            block_alleles = _coding_alleles(ds2, chunk, matrix, allele_label)
            if not args.ref_based:
                flips = np.array([major_allele_flipped(matrix[i]) for i in range(len(chunk))])
                matrix = np.where(
                    flips[:, None] & (matrix >= 0), 2.0 - matrix, matrix
                )
            r_values, _ = pearson_pairwise(target, matrix)
            block_stats.block_index = block_no
            block_stats.block_total = block_total
            for row_offset, var_index in enumerate(chunk):
                r = float(r_values[row_offset])
                if not np.isfinite(r):
                    reason = _skip_reason(target, matrix[row_offset])
                elif ds1.ids[index] == ds2.ids[var_index]:
                    reason = SKIP_SAME_ID
                else:
                    reason = None
                if reason is not None:
                    progress.skipped += 1
                    block_stats.skipped += 1
                    progress.skip_reasons[reason] = progress.skip_reasons.get(reason, 0) + 1
                    continue
                r2 = r * r
                # 0 doubles as "no filter"; any non-zero threshold keeps pairs
                # whose r^2 is at least that large.
                if r2_threshold == 0.0 or r2 >= r2_threshold:
                    rows.append((var_index, r, r2, block_alleles[row_offset]))
                    progress.kept += 1
                    block_stats.kept += 1
                else:
                    progress.below_r2 += 1
                    block_stats.below_r2 += 1
                block_stats.pairs_evaluated += 1
            progress.blocks += 1
            logger.debug(
                "block %d/%d variants=%d rows=%d-%d samples=%d mode=%s "
                "sparse_variants=%d sparse_fallbacks=%d pairs_evaluated=%d "
                "pairs_kept=%d pairs_below_r2=%d pairs_skipped=%d missing_frac=%.4f "
                "elapsed=%.3fs",
                block_stats.block_index,
                max(block_stats.block_total, 1),
                len(chunk),
                int(chunk[0]) if len(chunk) else -1,
                int(chunk[-1]) if len(chunk) else -1,
                len(target),
                block_stats.mode,
                block_stats.sparse_variants,
                block_stats.sparse_fallbacks,
                block_stats.pairs_evaluated,
                block_stats.kept,
                block_stats.below_r2,
                block_stats.skipped,
                block_stats.missing_frac,
                time.perf_counter() - start_time,
            )

        if rows:
            order_idx = np.argsort(np.array([ds2.pos[r[0]] for r in rows], dtype=np.int64), kind="stable")
            for position in order_idx:
                var_index, r, r2, allele_b = rows[position]
                write_row(
                    out,
                    with_r=args.with_r,
                    chrom_a=ds1.chrom[index],
                    pos_a=int(ds1.pos[index]),
                    id_a=ds1.ids[index],
                    chrom_b=ds2.chrom[var_index],
                    pos_b=int(ds2.pos[var_index]),
                    id_b=ds2.ids[var_index],
                    r=r,
                    r2=r2,
                    allele_label=allele_label,
                    allele_a=target_alleles[index],
                    allele_b=allele_b,
                )
            written_total += len(rows)

        progress.elapsed = time.perf_counter() - start_time
        logger.info(
            "target %d/%d %s %s:%d candidates=%d pairs_kept=%d pairs_below_r2=%d "
            "pairs_skipped=%d elapsed=%.3fs",
            order,
            total,
            ds1.ids[index],
            ds1.chrom[index],
            int(ds1.pos[index]),
            progress.candidates,
            progress.kept,
            progress.below_r2,
            progress.skipped,
            progress.elapsed,
        )
        if progress.skip_reasons:
            logger.debug(
                "target %s: skip reasons %s",
                ds1.ids[index],
                ", ".join(f"{k}={v}" for k, v in sorted(progress.skip_reasons.items())),
            )
    return written_total


def _to_ref(values: np.ndarray) -> np.ndarray:
    return np.where(values < 0, values, 2.0 - values)


def _block_slices(total: int, block_size: int):
    start = 0
    while start < total:
        yield start, min(start + block_size, total)
        start += block_size


def _coding_alleles(
    dataset: Dataset,
    chunk: np.ndarray,
    raw_matrix: np.ndarray,
    allele_label: str,
) -> list[str]:
    """The allele each dataset-2 variant is coded by, for the ``--r`` output.

    ``allele_label`` is ``"REF"`` for ``--ref-based`` and ``"MAJ"`` for the
    default major-allele orientation (which mirrors the ALT counts when ALT is
    not the major allele). ``raw_matrix`` must hold the un-mirrored ALT counts.
    """

    if allele_label == "REF":
        return [dataset.ref[int(i)] for i in chunk]
    return [
        dataset.ref[int(i)] if major_allele_flipped(raw_matrix[row]) else dataset.alt[int(i)]
        for row, i in enumerate(chunk)
    ]


def _skip_reason(target: np.ndarray, other: np.ndarray) -> str:
    valid = (target >= 0) & (other >= 0)
    if int(valid.sum()) < 3:
        return SKIP_MISSING
    return SKIP_ZERO_VARIANCE


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        level = logging.INFO
        if args.verbose:
            level = logging.DEBUG
        elif args.quiet:
            level = logging.WARNING
        configure_logging(level)
        return run(args)
    except CrossLDError as exc:
        logging.getLogger(LOGGER_NAME).error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
