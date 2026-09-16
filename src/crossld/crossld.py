"""Run crossld with command-line arguments."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from crossld import __version__
from crossld.blocks import BlockReader
from crossld.ld import (
    SKIP_MISSING,
    SKIP_ZERO_VARIANCE,
    major_allele_flipped,
    major_allele_flipped_sparse,
    pearson_pairwise_block,
)
from crossld.logging_setup import LOGGER_NAME, configure_logging
from crossld.plinkio import (
    PlinkDataset,
    PlinkIOError,
    candidate_indices,
    load_two_datasets,
    resolve_targets,
)
from crossld.results import write_header, write_row
from crossld.varfilter import VarFilterError, VariantFilter

logger = logging.getLogger(f"{LOGGER_NAME}.crossld")

DEFAULT_LD_WINDOW = 0
DEFAULT_LD_WINDOW_KB = 1000.0
DEFAULT_LD_WINDOW_R2 = 0.2
DEFAULT_BLOCK_SIZE = 500

class ArgumentError(Exception):
    """Command-line argument error."""

class _Parser(argparse.ArgumentParser):
    """ArgumentParser that reports errors through the module logger."""

    def error(self, message: str):
        raise ArgumentError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="CrossLD",
                     description="Cross-dataset linkage disequilibrium",
                     add_help=False)

    require_arg = parser.add_argument_group("Required")
    ds1 = require_arg.add_mutually_exclusive_group(required=True)
    ds1.add_argument("--bfile1", metavar="PREFIX", help="PLINK 1 .bed/.bim/.fam prefix (dataset 1)")
    ds1.add_argument("--pfile1", metavar="PREFIX", help="PLINK 2 .pgen/.pvar/.psam prefix (dataset 1)")

    ds2 = require_arg.add_mutually_exclusive_group(required=True)
    ds2.add_argument("--bfile2", metavar="PREFIX", help="PLINK 1 .bed/.bim/.fam prefix (dataset 2)")
    ds2.add_argument("--pfile2", metavar="PREFIX", help="PLINK 2 .pgen/.pvar/.psam prefix (dataset 2)")

    require_arg.add_argument("-o", "--out", metavar="FILE", default=None,
                             help="output file (default: stdout)")

    require_arg.add_argument("--ld-snp", metavar="ID",
                             help="single target variant ID in dataset 1")
    require_arg.add_argument("--ld-snps", metavar="LIST",
                             help="comma-separated target variant IDs in dataset 1")
    require_arg.add_argument("--ld-snp-list", metavar="FILE",
                             help="file of dataset-1 target variant IDs (one per line)")

    filter_arg = parser.add_argument_group("Variant filtering")
    filter_arg.add_argument("--keep", metavar="FILE",
                            help="single-column list of sample IID to keep for LD analysis (default: all overlap samples)")
    filter_arg.add_argument("--extract", metavar="FILE",
                            help="single-column list of variant IDs to keep in dataset 2")
    filter_arg.add_argument("--extract-bed", metavar="BED",
                            help="BED file of genomic regions to keep in dataset 2 (0-based half-open)")
    filter_arg.add_argument("--exclude", metavar="FILE",
                            help="single-column list of variant IDs to drop in dataset 2")
    filter_arg.add_argument("--exclude-bed", metavar="BED",
                            help="BED file of genomic regions to drop in dataset 2 (0-based half-open)")

    ld_arg = parser.add_argument_group("LD analysis")
    ld_arg.add_argument("--r", dest="with_r", action="store_true",
                        help="report unphased r instead of r2 (default: r2)")
    ld_arg.add_argument("--ref-based", action="store_true",
                        help="report r to REF alleles instead of each variant's major allele")
    ld_arg.add_argument("--ld-window", type=int, default=DEFAULT_LD_WINDOW, metavar="N",
                        help="max number of variants in dataset 2 to include for LD analysis (default: %(default)s for no limit)")
    ld_arg.add_argument("--ld-window-kb", type=float, default=DEFAULT_LD_WINDOW_KB, metavar="KB",
                        help="max distance in kb to include variants for LD analysis (default: %(default)s)")
    ld_arg.add_argument("--ld-window-r2", type=float, default=DEFAULT_LD_WINDOW_R2, metavar="R2",
                        help="min r2 to report LD pairs (default: %(default)s)")
    ld_arg.add_argument("--dosage", action="store_true", help="use genotype dosages for both datasets")
    ld_arg.add_argument("--dosage1", action="store_true", help="use genotype dosages for dataset 1")
    ld_arg.add_argument("--dosage2", action="store_true", help="use genotype dosages for dataset 2")
    ld_arg.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, metavar="N",
                        help="number of variants read per block (default: %(default)s)")

    other_arg = parser.add_argument_group("Others")
    verbosity = other_arg.add_mutually_exclusive_group()
    verbosity.add_argument("--debug", action="store_true", help="DEBUG logging")
    verbosity.add_argument("--quiet", action="store_true", help="WARNING logging only")
    other_arg.add_argument("-v", "--version", action="version", version=f"crossLD {__version__}")
    other_arg.add_argument("-h", "--help", action="help", help="show this help message and exit")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if args.ld_snp is None and args.ld_snps is None and args.ld_snp_list is None:
        errors.append("one of the arguments --ld-snp, --ld-snps, or --ld-snp-list is required")
    if args.ld_window < 0:
        errors.append("--ld-window must be 0 (no limit) or a positive integer")
    if args.ld_window_kb < 0:
        errors.append("--ld-window-kb must be >= 0")
    if not 0.0 <= args.ld_window_r2 <= 1.0:
        errors.append("--ld-window-r2 must be between 0 and 1")
    if args.block_size < 1:
        errors.append("--block-size must be a positive integer")
    if errors:
        raise ArgumentError("; ".join(errors))


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


def run_crossld(args: argparse.Namespace) -> int:
    """Parse command line arguments and run crossld."""

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

    allele_label = "REF" if args.ref_based else "MAJ"

    ds1, ds2 = load_two_datasets(
        args.bfile1 or args.pfile1,
        "bfile" if args.bfile1 else "pfile",
        dosage1,
        args.bfile2 or args.pfile2,
        "bfile" if args.bfile2 else "pfile",
        dosage2,
        keep_sample_file=args.keep,
    )

    try:
        if variant_filter.active:
            ds2.apply_variant_filter(variant_filter.build_mask(ds2))

        targets = resolve_targets(
            ds1,
            single=args.ld_snp,
            snps=args.ld_snps,
            snp_list=args.ld_snp_list,
        )
        if len(targets) == 0:
            logger.error("no target variants found in dataset 1")

        with _open_output(args.out) as out:
            write_header(out, args.with_r, allele_label)
            written = crossld(args, ds1, ds2, targets, out, allele_label)
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


def crossld(
    args: argparse.Namespace,
    ds1: PlinkDataset,
    ds2: PlinkDataset,
    targets: list[int],
    out,
    allele_label: str = "MAJ",
) -> int:
    """Main crossld loop."""

    block_size = args.block_size
    total = len(targets)
    written_total = 0
    r2_threshold = args.ld_window_r2
    ref_based = args.ref_based
    reader1 = BlockReader(ds1, dosage=bool(ds1.dosage), force_dense=True)
    reader2 = BlockReader(ds2, dosage=bool(ds2.dosage))
    bridge = ds1.idx_bridge

    # Orient each dataset-1 target once. The coding allele/sign are only needed
    # for --r, so they are skipped entirely when REF/ALT were not loaded.
    target_vectors: dict[int, npt.NDArray] = {}
    target_alleles: dict[int, str] = {}
    target_signs: dict[int, int] = {} # only store -1
    for index in targets:
        raw = reader1.read([index]).dense[0]
        if args.with_r:
            if ref_based:
                target_alleles[index] = ds1.ref_allele(index)
            elif major_allele_flipped(raw):
                target_alleles[index] = ds1.alt_allele(index)
                target_signs[index] = -1
            else:
                target_alleles[index] = ds1.ref_allele(index)
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
        )
        progress.candidates = len(candidates)
        rows: list[tuple] = []
        target_arr = target_vectors[index]
        sign_a = target_signs.get(index, 1)

        if len(candidates) == 0:
            logger.debug("Variant %s has no dataset-2 variants within --ld-window-kb %s", 
                         ds1.ids[index], args.ld_window_kb)
            block_total = 0
        else:
            block_total = max(1, -(-len(candidates) // block_size))
            
        for block_no, (start, end) in enumerate(_block_slices(len(candidates), block_size), start=1):
            block_var_idx = candidates[start:end]
            block = reader2.read(block_var_idx)
            block_stats = block.stats
            block_stats.block_index = block_no
            block_stats.block_total = block_total

            r_values, n_values = pearson_pairwise_block(target_arr, block)
            r2_values = r_values * r_values # r2 is always required for thresholding
            if args.with_r:
                signs_b, alleles_b = _block_allele_info(ds2, block_var_idx, block, allele_label)
                if signs_b is None:
                    r_values_report = r_values
                else:
                    r_values_report = r_values * sign_a * signs_b
            else:
                signs_b, alleles_b = None, [""] * len(block_var_idx)

            for row_offset, var_index in enumerate(block_var_idx):
                r = float(r_values[row_offset])
                r_report = float(r_values_report[row_offset]) if args.with_r else None
                r2 = float(r2_values[row_offset])
                if not np.isfinite(r):
                    reason = SKIP_MISSING if int(n_values[row_offset]) < 3 else SKIP_ZERO_VARIANCE
                    progress.skipped += 1
                    block_stats.skipped += 1
                    progress.skip_reasons[reason] = progress.skip_reasons.get(reason, 0) + 1
                    continue

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
                "block %d/%d variants=%d rows=%d-%d mode=%s "
                "dense_variants=%d sparse_variants=%d rows_fast=%d rows_masked=%d "
                "pairs_evaluated=%d "
                "pairs_kept=%d pairs_below_r2=%d pairs_skipped=%d elapsed=%.3fs",
                block_stats.block_index,
                max(block_stats.block_total, 1),
                block_stats.variants,
                int(block_var_idx[0]),
                int(block_var_idx[-1]),
                block_stats.mode,
                block_stats.dense_variants,
                block_stats.sparse_variants,
                block_stats.fast_rows,
                block_stats.masked_rows,
                block_stats.pairs_evaluated,
                block_stats.kept,
                block_stats.below_r2,
                block_stats.skipped,
                time.perf_counter() - start_time,
            )

        if rows:
            order_idx = np.argsort(
                np.array([ds2.pos[x[0]] for x in rows], dtype=np.uint32), kind="stable"
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
                    allele_a=target_alleles.get(index, ""),
                    allele_b=allele_b,
                )
            written_total += len(rows)

        progress.elapsed = time.perf_counter() - start_time
        logger.info(
            "target %d/%d %s at %s:%d candidates=%d pairs_kept=%d pairs_below_r2=%d "
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
    """Block slices generator."""
    start = 0
    while start < total:
        yield start, min(start + block_size, total)
        start += block_size


def _block_allele_info(
    ds: PlinkDataset,
    indices: npt.NDArray[np.uint32],
    block,
    allele_label: str,
) -> tuple[npt.NDArray[np.int8] | None, list[str]]:
    """Per-row major-allele sign and coding allele for a block.

    In ``REF`` mode the reported r needs no sign adjustment
    (``signs`` is ``None``); in ``MAJ`` mode ``signs`` is ``-1`` for rows whose
    ALT allele is the major allele.
    """

    if allele_label == "REF":
        return None, [ds.ref_allele(int(i)) for i in indices]

    sample_ct = ds.active_sample_ct
    signs = np.ones(len(indices), dtype=np.int8)
    alleles = [""] * len(indices)

    for row, position in enumerate(block.dense_rows.tolist()):
        flipped = major_allele_flipped(block.dense[row])
        signs[position] = -1 if flipped else 1
        index = int(indices[position])
        alleles[position] = (
            ds.alt_allele(index) if flipped else ds.ref_allele(index)
        )

    for row, position in enumerate(block.sparse_rows.tolist()):
        flipped = major_allele_flipped_sparse(block.sparse_values[row], sample_ct)
        signs[position] = -1 if flipped else 1
        index = int(indices[position])
        alleles[position] = (
            ds.alt_allele(index) if flipped else ds.ref_allele(index)
        )

    return signs, alleles


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        level = logging.INFO
        if args.debug:
            level = logging.DEBUG
        elif args.quiet:
            level = logging.WARNING
        configure_logging(level)
        return run_crossld(args)
    except (PlinkIOError, VarFilterError, ArgumentError, OSError) as exc:
        logging.getLogger(LOGGER_NAME).error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
