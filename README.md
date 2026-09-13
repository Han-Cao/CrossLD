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
| `--ld-snps LIST` | comma-separated IDs and ranges (`CHR:START-END`, `CHR:START-CHR:END`) |
| `--ld-snp-list FILE` | file of target IDs, one per line (`#` comments ignored) |
| `--keep FILE` | restrict samples to those listed; a leading `#IID` header makes the first column the IID column, otherwise the file must be single-column |
| `--extract FILE` / `--exclude FILE` | keep/drop dataset-2 variants by ID (one ID per line) |
| `--extract-bed FILE` / `--exclude-bed FILE` | keep/drop dataset-2 variants in 0-based, half-open interval BED regions (`CHROM START END`) |
| `--ld-window N` | max number of dataset-2 variants on each side of the target; `0` disables the count limit (default `1000`) |
| `--ld-window-kb KB` | max distance in kb from the target (default `1000`) |
| `--ld-window-r2 R2` | only report pairs with `r²` at least this large; `0` disables the filter (default `0.2`) |
| `--dosage` / `--dosage1` / `--dosage2` | use dosages instead of hardcalls for both / one dataset |
| `--ref-based` | orient r to REF alleles instead of each variant's major allele |
| `--r` | report signed `UNPHASED_R` plus the coding allele per side (`MAJ_A`/`MAJ_B`, or `REF_A`/`REF_B` with `--ref-based`) instead of `UNPHASED_R2` |
| `--sparse {auto,never,require}` | sparse `pgen` reads (default `auto`, with dense fallback) |
| `--block-size N` | variants read per block (default `16384`) |
| `--sample-id {iid,fid-iid}` | sample matching key for `.fam`-based inputs |
| `--out FILE` | output file (default: stdout) |
| `-v` / `-q` | DEBUG / WARNING logging (default INFO) |

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

Rows are emitted in `--ld-snp*` order and then by ascending dataset-2 position.
Pairs whose two variant IDs are identical are skipped.

## Conventions

* **Samples**: the IID intersection of the two datasets, in dataset-1 order, is
  used by default; `--keep` narrows it further and warns about requested samples
  that are absent from either dataset.
* **Sign**: `r` is the Pearson correlation of unphased allele dosages. By
  default each variant is oriented to its major allele, so a positive sign means
  the major alleles are positively correlated; `--ref-based` orients to REF
  instead.
* **Orientation**: variants whose REF/ALT are swapped between datasets are
  handled automatically, because each side is oriented within its own dataset.
* **Filters**: `--extract*`/`--exclude*` apply to dataset 2 only, and the variant
  count used by `--ld-window` is measured over the surviving variants.
* **Windows**: `--ld-window N` permits at most `N` dataset-2 variants on *each*
  side of the target's own column (so the target itself does not consume budget),
  and `--ld-window-kb` bounds the physical distance. `--ld-window 0` disables the
  count limit. Because dataset 1 and dataset 2 are separate files, this is
  deliberately not PLINK 2's "variant-column distance + 1" interpretation.
* **r² threshold**: `--ld-window-r2` keeps pairs with `r²` greater than **or
  equal to** the value (so `--ld-window-r2 1` keeps the perfectly correlated
  pairs); `--ld-window-r2 0` disables the filter entirely.
* **Sparse reads**: when the underlying `.pgen` stores a variant sparsely,
  `crossLD` uses `pgenlib`'s sparse readers and reconstructs the dense vector;
  otherwise it falls back to a dense block read (`--sparse never` disables
  sparse use entirely, `require` turns the fallback into an error).
* Only biallelic variants are supported; multiallelic input is rejected.

## Logging

Diagnostics go to **stderr**, so stdout stays a clean result stream.

* Default (`INFO`): per-dataset summary, sample/filter summaries, one line per
  resolved target with its candidate/pair counts and elapsed time, and a final
  run summary.
* `-v` (`DEBUG`): adds detailed per-block statistics (variant counts, index
  range, read mode, sparse hits/fallbacks, pairs evaluated/kept/skipped,
  missing fraction, elapsed time), per-target plan details and skip reasons.

```
INFO [crossld.crossld] target 1/4 20:114311:G:T 20:114311 candidates=537 pairs_kept=230 pairs_below_r2=306 pairs_skipped=1 elapsed=0.003s
DEBUG [crossld.crossld] block 1/3 variants=200 rows=0-205 samples=100 mode=hardcall sparse_variants=0 sparse_fallbacks=1 pairs_evaluated=199 pairs_kept=57 pairs_below_r2=142 pairs_skipped=1 missing_frac=0.0000 elapsed=0.001s
INFO [crossld.crossld] done: 4 target(s), 784 pair(s) written
```

## Development

```bash
pytest                       # runs the suite with coverage (fail_under = 90)
pytest --cov=crossld --cov-report=term-missing
```

The layout is `src/crossld/`: `plinkio` (dataset/sample/variant resolution),
`varfilter` (dataset-2 extraction), `blocks` (sparse-aware block reads), `ld`
(correlation math), `results` (output formatting), `logging_setup`, and
`crossld` (the CLI / orchestration module).

Naming: the importable package is lowercase (`crossld`, so
`import crossld.crossld`), while the user-facing program and console script keep
the `crossLD` spelling. Test fixtures live under `tests/data/`, and the synthetic
input builders plus fixtures are in `tests/conftest.py`.
