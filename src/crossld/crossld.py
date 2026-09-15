"""Command-line interface and run orchestration for crossld."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .blocks import BlockReader
from .ld import (
    SKIP_MISSING,
    SKIP_SAME_ID,
    SKIP_ZERO_VARIANCE,
    major_allele_flipped,
    major_allele_flipped_sparse,
    pearson_pairwise_block,
)
from .logging_setup import LOGGER_NAME, configure_logging
from .plinkio import (
    PlinkDataset,
    PlinkIOError,
    candidate_indices,
    load_two_datasets,
    resolve_targets,
)
from .results import write_header, write_row
from .varfilter import VariantFilter

logger = logging.getLogger(f"{LOGGER_NAME}.crossld")

DEFAULT_LD_WINDOW = 1000
DEFAULT_LD_WINDOW_KB = 1000.0
DEFAULT_LD_WINDOW_R2 = 0.2
DEFAULT_BLOCK_SIZE = 200


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that reports errors through the module logger."""

    def error(self, message: str):  # pragma: no cover - exercised via CLI tests
        raise PlinkIOError(message)


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
        help="comma-separated dataset-1 target variant IDs",
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
            "max number of dataset-2 variants nearest to the target "
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
        raise PlinkIOError("; ".join(errors))


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

    dosage1 = args.dosage or args.dosage1
    dosage2 = args.dosage or args.dosage2
    # With --r the coding allele is reported for each side, so the column name
    # depends on the orientation mode.
    allele_label = "REF" if args.ref_based else "MAJ"

    ds1, ds2 = load_two_datasets(
        args.bfile1 or args.pfile1,
        "bfile" if args.bfile1 else "pfile",
        dosage1,
        args.bfile2 or args.pfile2,
        "bfile" if args.bfile2 else "pfile",
        dosage2,
        keep=args.keep,
    )

    try:
        mask = (
            variant_filter.build_mask(ds2)
            if variant_filter.active
            else np.ones(ds2.variant_ct, dtype=bool)
        )
        kept_indices = np.flatnonzero(mask).astype(np.uint32)

        targets = resolve_targets(
            ds1,
            single=args.ld_snp,
            snps=args.ld_snps,
            snp_list=args.ld_snp_list,
        )
        if not targets:
            logger.warning("no dataset-1 target variants resolved; nothing to do")

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
    ds1: PlinkDataset,
    ds2: PlinkDataset,
    targets: list[int],
    kept_indices: np.ndarray,
    out,
    allele_label: str = "MAJ",
) -> int:
    block_size = args.block_size
    total = len(targets)
    written_total = 0
    r2_threshold = args.ld_window_r2
    ref_based = args.ref_based
    reader1 = BlockReader(ds1, dosage=bool(ds1.dosage), force_dense=True)
    reader2 = BlockReader(ds2, dosage=bool(ds2.dosage))
    bridge = ds1.idx_bridge

    # Orient each dataset-1 target once. The coding allele is recorded alongside
    # so --r can report it; no matrix/vector recoding is done for r itself.
    target_vectors: dict[int, np.ndarray] = {}
    target_alleles: dict[int, str] = {}
    target_signs: dict[int, float] = {}
    for index in targets:
        raw = reader1.read([index]).dense[0].astype(np.float64)
        if ref_based:
            target_alleles[index] = ds1.ref[index]
            target_signs[index] = 1.0
        elif major_allele_flipped(raw):
            target_alleles[index] = ds1.ref[index]
            target_signs[index] = -1.0
        else:
            target_alleles[index] = ds1.alt[index]
            target_signs[index] = 1.0
        target_vectors[index] = raw[bridge] if bridge is not None else raw

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
        progress.candidates = len(candidates)
        rows: list[tuple] = []
        target = target_vectors[index]
        sign_a = target_signs[index]

        if len(candidates) == 0:
            logger.debug("target %s: window holds no dataset-2 variants", ds1.ids[index])

        block_total = max(1, -(-len(candidates) // block_size)) if len(candidates) else 0
        for block_no, (start, end) in enumerate(_block_slices(len(candidates), block_size), start=1):
            chunk = candidates[start:end]
            block = reader2.read(chunk)
            block_stats = block.stats
            block_stats.block_index = block_no
            block_stats.block_total = block_total

            r_values, n_values = pearson_pairwise_block(target, block)
            if args.with_r:
                signs_b, alleles_b = _block_allele_info(ds2, chunk, block, allele_label)
            else:
                signs_b, alleles_b = None, [""] * len(chunk)

            for row_offset, var_index in enumerate(chunk):
                r = float(r_values[row_offset])
                if not np.isfinite(r):
                    reason = SKIP_MISSING if int(n_values[row_offset]) < 3 else SKIP_ZERO_VARIANCE
                elif ds1.ids[index] == ds2.ids[var_index]:
                    reason = SKIP_SAME_ID
                else:
                    reason = None
                if reason is not None:
                    progress.skipped += 1
                    block_stats.skipped += 1
                    progress.skip_reasons[reason] = progress.skip_reasons.get(reason, 0) + 1
                    continue

                r_report = r
                if args.with_r and not ref_based:
                    r_report = r * sign_a * signs_b[row_offset]
                r2 = r * r
                # 0 doubles as "no filter"; any non-zero threshold keeps pairs
                # whose r^2 is at least that large.
                if r2_threshold == 0.0 or r2 >= r2_threshold:
                    rows.append((var_index, r_report, r2, alleles_b[row_offset]))
                    progress.kept += 1
                    block_stats.kept += 1
                else:
                    progress.below_r2 += 1
                    block_stats.below_r2 += 1
                block_stats.pairs_evaluated += 1
            progress.blocks += 1
            logger.debug(
                "block %d/%d variants=%d rows=%d-%d samples=%d mode=%s "
                "dense_variants=%d sparse_variants=%d pairs_evaluated=%d "
                "pairs_kept=%d pairs_below_r2=%d pairs_skipped=%d elapsed=%.3fs",
                block_stats.block_index,
                max(block_stats.block_total, 1),
                block_stats.variants,
                int(chunk[0]) if len(chunk) else -1,
                int(chunk[-1]) if len(chunk) else -1,
                len(target),
                block_stats.mode,
                block_stats.dense_variants,
                block_stats.sparse_variants,
                block_stats.pairs_evaluated,
                block_stats.kept,
                block_stats.below_r2,
                block_stats.skipped,
                time.perf_counter() - start_time,
            )

        if rows:
            order_idx = np.argsort(
                np.array([ds2.pos[r[0]] for r in rows], dtype=np.int64), kind="stable"
            )
            for position in order_idx:
                var_index, r_report, r2, allele_b = rows[position]
                write_row(
                    out,
                    with_r=args.with_r,
                    chrom_a=ds1.chrom[index],
                    pos_a=int(ds1.pos[index]),
                    id_a=ds1.ids[index],
                    chrom_b=ds2.chrom[var_index],
                    pos_b=int(ds2.pos[var_index]),
                    id_b=ds2.ids[var_index],
                    r=r_report,
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


def _block_slices(total: int, block_size: int):
    start = 0
    while start < total:
        yield start, min(start + block_size, total)
        start += block_size


def _block_allele_info(
    dataset: PlinkDataset,
    chunk: np.ndarray,
    block,
    allele_label: str,
) -> tuple[np.ndarray | None, list[str]]:
    """Per-row major-allele sign and coding allele for a block.

    In ``REF`` mode the reported r needs no sign adjustment
    (``signs`` is ``None``); in ``MAJ`` mode ``signs`` is ``-1`` for rows whose
    ALT allele is the minor allele (i.e. the row must be mirrored to be
    major-allele coded).
    """

    if allele_label == "REF":
        return None, [dataset.ref[int(i)] for i in chunk]

    sample_ct = dataset.active_sample_ct
    signs = np.ones(len(chunk), dtype=np.float64)
    alleles = [""] * len(chunk)

    for row, position in enumerate(block.dense_rows.tolist()):
        flipped = major_allele_flipped(block.dense[row])
        signs[position] = -1.0 if flipped else 1.0
        index = int(chunk[position])
        alleles[position] = dataset.ref[index] if flipped else dataset.alt[index]

    for row, position in enumerate(block.sparse_rows.tolist()):
        flipped = major_allele_flipped_sparse(block.sparse_values[row], sample_ct)
        signs[position] = -1.0 if flipped else 1.0
        index = int(chunk[position])
        alleles[position] = dataset.ref[index] if flipped else dataset.alt[index]

    return signs, alleles


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
    except (PlinkIOError, OSError) as exc:
        logging.getLogger(LOGGER_NAME).error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
