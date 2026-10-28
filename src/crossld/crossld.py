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
from crossld.blocks import BlockCache, BlockReader
from crossld.ld import (
    SKIP_LOW_VARIANCE,
    SKIP_MISSING,
    major_allele_flipped,
    major_allele_flipped_sparse,
    pearson_pairwise_block,
    target_qc,
)
from crossld.logging_setup import LOGGER_NAME, configure_logging
from crossld.plinkio import (
    PlinkDataset,
    PlinkIOError,
    iter_target_candidates,
    load_two_datasets,
    resolve_targets,
)
from crossld.results import write_header, write_row
from crossld.varfilter import VarFilterError, VariantFilter

logger = logging.getLogger(f"{LOGGER_NAME}.crossld")

DEFAULT_LD_WINDOW = 0
DEFAULT_LD_WINDOW_KB = 1000.0
DEFAULT_LD_WINDOW_R2 = 0.2
DEFAULT_BLOCK_SIZE = 200
DEFAULT_CACHE_LIMIT = 5
DEFAULT_CACHE_MIN_OVERLAP = 0.5

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
    ld_arg.add_argument("--cache-limit", type=int, default=DEFAULT_CACHE_LIMIT, metavar="N",
                        help="max number of dataset-2 blocks cached across overlapping "
                             "target windows, keeping the rightmost (default: %(default)s; 0 disables)")
    ld_arg.add_argument("--cache-min-overlap", type=float, default=DEFAULT_CACHE_MIN_OVERLAP,
                        metavar="F",
                        help="cache a block only when its overlap with the next target's "
                             "candidates exceeds this fraction of the block (default: %(default)s)")

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
    if args.cache_limit < 0:
        errors.append("--cache-limit must be 0 (disabled) or a positive integer")
    if not 0.0 <= args.cache_min_overlap <= 1.0:
        errors.append("--cache-min-overlap must be between 0 and 1")
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
    target_indices: list[int],
    f_out,
    allele_label: str = "MAJ",
) -> int:
    """Main crossld loop."""

    block_size = args.block_size
    total = len(target_indices)
    written_total = 0
    r2_threshold = args.ld_window_r2
    ref_based = args.ref_based
    reader1 = BlockReader(ds1, dosage=bool(ds1.dosage), force_dense=True)
    reader2 = BlockReader(ds2, dosage=bool(ds2.dosage))
    cache = BlockCache(limit=args.cache_limit, min_overlap=args.cache_min_overlap)
    bridge = ds1.idx_bridge

    target_alleles: dict[int, str] = {} # index -> major/ref allele
    target_neg_signs: dict[int, int] = {} # only store -1 to reverse r
    for order, (index, candidates, keep_for_next) in enumerate(
        iter_target_candidates(
            ds1, ds2, target_indices, args.ld_window_kb, args.ld_window
        ),
        start=1,
    ):
        # target always read as dense vector
        raw = reader1.read([index]).dense[0]

        # The coding allele/sign are only needed for --r, 
        # so they are skipped entirely when REF/ALT were not loaded.
        if args.with_r:
            if ref_based:
                target_alleles[index] = ds1.ref_allele(index)
            elif major_allele_flipped(raw):
                target_alleles[index] = ds1.alt_allele(index)
                target_neg_signs[index] = -1
            else:
                target_alleles[index] = ds1.ref_allele(index)

        # use idx_bridge to align target vector to the dataset-2 sample order
        target_arr = raw[bridge] if bridge is not None else raw

        start_time = time.perf_counter()
        progress = _TargetProgress(
            index=index,
            variant_id=ds1.ids[index],
            chrom=ds1.chrom[index],
            pos=int(ds1.pos[index]),
            skip_reasons={SKIP_MISSING: 0, SKIP_LOW_VARIANCE: 0},
        )
        # candidates were prefetched together with the target; ``keep_for_next``
        # is the next target's window when it may overlap this one.
        progress.candidates = len(candidates)
        rows_result: list[tuple] = []
        sign_a = target_neg_signs.get(index, 1) # if not in target_neg_signs, it is 1
        reused_before = cache.reused_variants

        # Target-level QC depends only on the target vector, so it is checked
        # once per target. A degenerate target skips every candidate pair, so the
        # fresh blocks are never read.
        skip_reason = target_qc(target_arr)
        if skip_reason is not None:
            progress.skipped += len(candidates)
            progress.skip_reasons[skip_reason] += len(candidates)
        elif len(candidates) == 0:
            logger.debug("Variant %s has no dataset-2 variants within --ld-window-kb %s", 
                         ds1.ids[index], args.ld_window_kb)

        planned = cache.plan(candidates, block_size)
        block_total = len(planned)
        cache.begin()
        for block_no, (block_var_idx, cached_block) in enumerate(planned, start=1):
            if skip_reason is not None:
                # no fresh reads; keep the cached views for the next target
                if cached_block is not None:
                    cache.consider(block_var_idx, cached_block, keep_for_next)
                continue
            block = cached_block if cached_block is not None else reader2.read(block_var_idx)
            cache.consider(block_var_idx, block, keep_for_next)
            block_stats = block.stats
            block_stats.block_index = block_no
            block_stats.block_total = block_total

            r_values, _n_values = pearson_pairwise_block(target_arr, block)
            r2_values = r_values * r_values # r2 is always required for thresholding
            if args.with_r:
                signs_b, alleles_b = _block_allele_info(ds2, block_var_idx, block, allele_label)
                if signs_b is None:
                    r_values_report = r_values
                else:
                    r_values_report = r_values * sign_a * signs_b
            else:
                # these will never be used, just need the shape
                alleles_b = np.empty(len(block_var_idx), dtype=bool)
                r_values_report = np.empty(len(block_var_idx), dtype=bool)

            # select LD pairs > --ld-window-r2 for output
            keep_rows = r2_values >= r2_threshold if r2_threshold > 0 else ~np.isnan(r2_values)
            if keep_rows.all():
                n_kept = len(block_var_idx)
                n_below_r2 = 0
                result_block_var_idx = block_var_idx
                result_r_report = r_values_report
                result_r2 = r2_values
                result_alleles_b = alleles_b
            else:
                n_kept = int(keep_rows.sum())
                n_below_r2 = int(np.sum(r2_values < r2_threshold)) # np.nan -> 0
                result_block_var_idx = block_var_idx[keep_rows]
                result_r_report = r_values_report[keep_rows]
                result_r2 = r2_values[keep_rows]
                result_alleles_b = alleles_b[keep_rows]

            # prepare output
            # tuple of (block_var_index, r, r2, block_var allele)
            # for r2, r and allele is None
            rows_result.extend(zip(
                result_block_var_idx.tolist(),
                result_r_report.tolist(),
                result_r2.tolist(),
                result_alleles_b.tolist()
            ))

            # count summary
            progress.kept += n_kept
            progress.below_r2 += n_below_r2
            progress.skipped += block.stats.skip_missing + block.stats.skip_low_variance
            progress.skip_reasons[SKIP_MISSING] += block.stats.skip_missing
            progress.skip_reasons[SKIP_LOW_VARIANCE] += block.stats.skip_low_variance
            progress.blocks += 1

            block.stats.kept += n_kept
            block.stats.below_r2 += n_below_r2

            logger.debug(
                "block %d/%d variants=%d rows=%d-%d mode=%s "
                "dense_variants=%d sparse_variants=%d "
                "complete_pair=%d incomplete_pair=%d "
                "raw_moment_pair=%d centered_moment_pair=%d "
                "pairs_kept=%d pairs_below_r2=%d "
                "skip_missing=%d skip_low_variance=%d elapsed=%.3fs",
                block_stats.block_index,
                max(block_stats.block_total, 1),
                block_stats.variants,
                int(block_var_idx[0]),
                int(block_var_idx[-1]),
                block_stats.mode,
                block_stats.dense_variants,
                block_stats.sparse_variants,
                block_stats.complete_pair,
                block_stats.incomplete_pair,
                block_stats.raw_moment_pair,
                block_stats.centered_moment_pair,
                block_stats.kept,
                block_stats.below_r2,
                block_stats.skip_missing,
                block_stats.skip_low_variance,
                time.perf_counter() - start_time,
            )

        # make this target's rightmost eligible blocks the cache for the next
        cache.commit()

        if rows_result:
            pos_b = np.array([ds2.pos[x[0]] for x in rows_result], dtype=np.uint32)
            idx_b = np.array([x[0] for x in rows_result], dtype=np.uint32)
            # primary key is position; the dataset-2 index breaks ties so the
            # output cannot depend on block composition (i.e. on --cache-limit)
            order_idx = np.lexsort((idx_b, pos_b))
            for position in order_idx:
                var_index, r_report, r2, allele_b = rows_result[position]
                write_row(
                    f_out,
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
            written_total += len(rows_result)

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
        logger.debug(
            "target %s reused_variants=%d cached_blocks=%d",
            ds1.ids[index],
            cache.reused_variants - reused_before,
            cache.size,
        )
    return written_total


def _block_allele_info(
    ds: PlinkDataset,
    indices: npt.NDArray[np.uint32],
    block,
    allele_label: str,
) -> tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.str_]]:
    """Per-row major-allele sign and coding allele for a block.

    In ``REF`` mode the reported r needs no sign adjustment
    (``signs`` is ``None``); in ``MAJ`` mode ``signs`` is ``-1`` for rows whose
    ALT allele is the major allele.
    """

    if allele_label == "REF":
        return None, np.asarray([ds.ref_allele(int(i)) for i in indices])

    sample_ct = ds.active_sample_ct
    signs = [1] * len(indices)
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

    return np.asarray(signs, dtype=np.int8), np.asarray(alleles)


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
