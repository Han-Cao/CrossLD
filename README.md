# crossLD

Cross-dataset linkage disequilibrium between PLINK 1 (`--bfile`) and PLINK 2
(`--pgen/--pvar/--psam`) datasets.

`crossLD` picks target variants from **dataset 1** and correlates them (pairwise
unphased Pearson r / r²) against every nearby variant in **dataset 2**, reading
both files block-wise through [`pgenlib`](https://pypi.org/project/Pgenlib/).
Because the two datasets are read with the same per-sample index arrays, LD is
always computed on the samples the two datasets share.

## Install

```bash
pip install -e .            # provides the `crossLD` console script
```

`pip install -e .` needs `setuptools` available in the environment. Tests can
also be run straight from a checkout without installing: the repository
`conftest.py` puts `src/` on `sys.path`.

## Quick start

```bash
crossLD \
  --pfile1 tests/data/1kg.sample1 \
  --pfile2 tests/data/1kg.sample2 \
  --ld-snp-list tests/data/ld_snp_list.txt \
  --extract tests/data/ld_extract_snp_list.txt \
  --ld-window-r2 0.05 \
  --out out.vcor
```

`tests/data/` carries three curated reference files for that command over four
dataset-1 targets: `ld_results_r2.vcor` (`UNPHASED_R2`),
`ld_results_r_major.vcor` (signed `UNPHASED_R` toward each variant's major
allele, with `MAJ_A`/`MAJ_B` columns) and `ld_results_r_ref.vcor` (signed
`UNPHASED_R` toward REF, with `REF_A`/`REF_B`). All three are reproduced by
`tests/test_reference.py`.

PLINK 1 inputs work the same way:

```bash
crossLD --bfile1 cohort --bfile2 reference --ld-snp 20:114311:G:T --ld-window-r2 0.2
```

## Options

| Option | Description |
| --- | --- |
| `--bfile1 P` / `--pfile1 P` | dataset-1 prefix (PLINK 1 `.bed/.bim/.fam` or PLINK 2 `.pgen/.pvar/.psam`) |
| `--bfile2 P` / `--pfile2 P` | dataset-2 prefix |
| `--ld-snp ID` | single dataset-1 target variant |
| `--ld-snps LIST` | comma-separated dataset-1 target variant IDs |
| `--ld-snp-list FILE` | file of target IDs, one per line (`#` comments ignored) |
| `--keep FILE` | restrict samples to those listed (single-column IID file; a leading `#` header line is ignored) |
| `--extract FILE` / `--exclude FILE` | keep/drop dataset-2 variants by ID (one ID per line) |
| `--extract-bed FILE` / `--exclude-bed FILE` | keep/drop dataset-2 variants in 0-based, half-open interval BED regions (`CHROM START END`) |
| `--ld-window N` | max number of dataset-2 variants nearest to the target (by absolute position distance); `0` disables the count limit (default `1000`) |
| `--ld-window-kb KB` | max distance in kb from the target (default `1000`) |
| `--ld-window-r2 R2` | only report pairs with `r²` at least this large; `0` disables the filter (default `0.2`) |
| `--dosage` / `--dosage1` / `--dosage2` | use dosages instead of hardcalls for both / one dataset |
| `--ref-based` | orient r to REF alleles instead of each variant's major allele |
| `--r` | report signed `UNPHASED_R` plus the coding allele per side (`MAJ_A`/`MAJ_B`, or `REF_A`/`REF_B` with `--ref-based`) instead of `UNPHASED_R2` |
| `--block-size N` | variants read per block (default `500`) |
| `--out FILE` | output file (default: stdout) |
| `--debug` / `--quiet` | DEBUG / WARNING logging (default INFO) |

## Output

The default output is `r²` only, tab-separated with six significant digits:

```
#CHROM_A	POS_A	ID_A	CHROM_B	POS_B	ID_B	UNPHASED_R2
20	114311	20:114311:G:T	20	101498	20:101498:A:G	0.0910081
```

With `--r` the signed correlation is reported instead, together with the allele
each side is coded by, so the sign is interpretable:

```
#CHROM_A	POS_A	ID_A	MAJ_A	CHROM_B	POS_B	ID_B	MAJ_B	UNPHASED_R
20	114311	20:114311:G:T	G	20	101498	20:101498:A:G	A	-0.301675
```

The allele columns are named `MAJ_A`/`MAJ_B` for the default major-allele
orientation and `REF_A`/`REF_B` with `--ref-based`. `--ld-window-r2` still
filters on `r²` in either shape.

Rows are emitted in dataset-1 variant order and then by ascending dataset-2
position. A dataset-1 target that is also present in dataset 2 is correlated
against itself and reported with `r`/`r²` of `1`.

## Conventions

* **Samples**: the IID intersection of the two datasets is used by default;
  `--keep` narrows it further and warns about requested samples that are absent
  from either dataset. Each dataset keeps its own file order (pgenlib requires
  increasing sample indexes); `idx_bridge` permutes the dataset-1 vectors into
  dataset-2 order before correlation, so the two files may list their samples in
  any order.
* **Sign**: `r` is the Pearson correlation of unphased ALT-coded dosages. No
  vector is recoded: `r^2` is orientation independent, `--ref-based` reports the
  ALT-coded correlation (identical to the REF-coded one, since both sides flip),
  and the default major-allele `r` is the ALT-coded `r` negated for each side
  whose ALT allele is the minor allele. A positive sign then means the major
  alleles are positively correlated. The `MAJ_A`/`MAJ_B` (or `REF_A`/`REF_B`)
  columns record the allele each side is oriented to.
* **Orientation**: variants whose REF/ALT are swapped between datasets are
  handled automatically, because each side is oriented within its own dataset.
* **Allele metadata**: REF/ALT strings are never cached; they are decoded on
  demand from the `.pvar`/`.bim` reader only for the rows that are actually
  written. An `r²`-only run reports `r²` without reading any allele strings, and
  a `--r` run only pays for its output rows.
* **Filters**: `--extract*`/`--exclude*` apply to dataset 2 only, and the variant
  count used by `--ld-window` is measured over the surviving variants.
* **Windows**: `--ld-window N` keeps the `N` dataset-2 variants nearest to the
  target by absolute position distance, and `--ld-window-kb` bounds the physical
  distance. `--ld-window 0` disables the count limit. Because dataset 1 and
  dataset 2 are separate files, this is deliberately not PLINK 2's
  "variant-column distance + 1" interpretation.
* **r² threshold**: `--ld-window-r2` keeps pairs with `r²` greater than **or
  equal to** the value (so `--ld-window-r2 1` keeps the perfectly correlated
  pairs); `--ld-window-r2 0` disables the filter entirely.
* **Sparse reads**: each block is split per variant. Variants whose `.pgen`
  representation is sparse are read with `pgenlib`'s sparse readers and their
  correlations are computed directly from the sparse rows (samples absent from a
  sparse row are homozygous REF), while the remaining variants are read densely.
  Target vectors are always read densely. Sparse reads are used automatically
  whenever the data supports them; there is no CLI switch.
* Only biallelic variants are supported; multiallelic input is rejected.

## Logging

Diagnostics go to **stderr**, so stdout stays a clean result stream.

* Default (`INFO`): per-dataset summary, sample/filter summaries, one line per
  resolved target with its candidate/pair counts and elapsed time, and a final
  run summary.
* `--debug` (`DEBUG`): adds detailed per-block statistics (variant counts, index
  range, read mode, dense/sparse row counts, pairs evaluated/kept/skipped,
  elapsed time), per-target plan details and skip reasons.
* `--quiet` (`WARNING`): suppresses the INFO lines.

```
[2026-09-14 12:00:00] - [INFO]: target 1/4 20:114311:G:T 20:114311 candidates=537 pairs_kept=231 pairs_below_r2=306 pairs_skipped=0 elapsed=0.006s
[2026-09-14 12:00:00] - [DEBUG]: block 1/3 variants=200 rows=0-205 mode=hardcall dense_variants=114 sparse_variants=86 pairs_evaluated=200 pairs_kept=58 pairs_below_r2=142 pairs_skipped=0 elapsed=0.002s
[2026-09-14 12:00:00] - [INFO]: done: 4 target(s), 788 pair(s) written
```

## Development

```bash
pytest                       # runs the whole test suite
pytest --cov=crossld --cov-report=term-missing   # optional coverage report
```

The layout is `src/crossld/`: `plinkio` (dataset/sample/variant resolution),
`varfilter` (dataset-2 extraction), `blocks` (sparse-aware block reads), `ld`
(correlation math), `results` (output formatting), `logging_setup`, and
`crossld` (the CLI / orchestration module).