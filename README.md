# CrossLD

[![pytest](https://github.com/Han-Cao/crossLD/actions/workflows/pytest.yml/badge.svg)](https://github.com/Han-Cao/crossLD/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/han-cao/5baaf521e0e161712d3493546f6a8876/raw/CrossLD-cobertura-coverage.json)](https://github.com/Han-Cao/crossLD)

Cross-dataset Linkage Disequilibrium analysis.

- [Overview](#overview)
- [Installation](#installation)

## Overview
`CrossLD` calculates linkage disequilibrium (LD) between variants from different [PLINK1/2 binary](https://www.cog-genomics.org/plink/2.0/input) datasets. It selects target variant(s) from one dataset and calculates unphased LD with their nearby variants from another dataset. This is useful to analyze LD between different variant callsets (e.g., SNV vs CNV) of the same (or overlapping) samples without merging datasets, particularly for large biobank-scale datasets.

Key features:
- Fast PLINK1/2 binary reading with [pgenlib](https://pypi.org/project/pgenlib/), supporting hardcall and dosage genotypes.
- Optimized data structure and algorithm for sparse genotype matrices to speed up analysis for biobank-scale datasets with low memory usage.
- Identical LD results to PLINK2's `--r[2]-unphased` function.


## Installation

With Python 3.10+, clone this repo and install by pip:
```bash
pip install -e .
```

For development and testing, install the dev dependencies:
```bash
pip install -e .[dev]
```


## Quick start

```bash
crossld \
  --pfile1 tests/data/1kg.sample1 \
  --pfile2 tests/data/1kg.sample2 \
  --ld-snp-list tests/data/ld_snp_list.txt \
  --extract tests/data/ld_extract_snp_list.txt \
  --ld-window-r2 0.05 \
  --ld-window-kb 500 \
  --out out.vcor
```

This will calculate LD between two example datasets from the 1000 Genomes Project. Each dataset contains 200 samples with 100 samples overlapping. For each target variant in `ld_snp_list.txt`, the program will calculate LD with nearby variants in `ld_extract_snp_list.txt` within 500kb and output results with r2 >= 0.05 to `out.vcor`. The output format is the same as PLINK2's [.vcor](https://www.cog-genomics.org/plink/2.0/formats#vcor) format.

## Options

```text
Required:
  --bfile1 PREFIX       PLINK 1 .bed/.bim/.fam prefix (dataset 1)
  --pfile1 PREFIX       PLINK 2 .pgen/.pvar/.psam prefix (dataset 1)
  --bfile2 PREFIX       PLINK 1 .bed/.bim/.fam prefix (dataset 2)
  --pfile2 PREFIX       PLINK 2 .pgen/.pvar/.psam prefix (dataset 2)
  -o, --out FILE        output file (default: stdout)
  --ld-snp ID           single target variant ID in dataset 1
  --ld-snps LIST        comma-separated target variant IDs in dataset 1
  --ld-snp-list FILE    file of dataset-1 target variant IDs (one per line)

Variant filtering:
  --keep FILE           single-column list of sample IID to keep for LD analysis (default: all overlap samples)
  --extract FILE        single-column list of variant IDs to keep in dataset 2
  --extract-bed BED     BED file of genomic regions to keep in dataset 2 (0-based half-open)
  --exclude FILE        single-column list of variant IDs to drop in dataset 2
  --exclude-bed BED     BED file of genomic regions to drop in dataset 2 (0-based half-open)

LD analysis:
  --r                   report unphased r instead of r2 (default: r2)
  --ref-based           report r to REF alleles instead of each variant's major allele
  --ld-window N         max number of variants in dataset 2 to include for LD analysis (default: 0 for no limit)
  --ld-window-kb KB     max distance in kb to include variants for LD analysis (default: 1000.0)
  --ld-window-r2 R2     min r2 to report LD pairs (default: 0.2)
  --dosage              use genotype dosages for both datasets
  --dosage1             use genotype dosages for dataset 1
  --dosage2             use genotype dosages for dataset 2
  --block-size N        number of variants read per block (default: 200)
  --cache-limit N       max number of dataset-2 blocks cached across overlapping target windows, keeping the rightmost (default: 5; 0 disables)
  --cache-min-overlap F
                        cache a block only when its overlap with the next target's candidates exceeds this fraction of the block (default: 0.5)

Others:
  --debug               DEBUG logging
  --quiet               WARNING logging only
  -v, --version         show program's version number and exit
  -h, --help            show this help message and exit
```

## Memory usage

For biobank-scale datasets, memory usage can be a concern when a large LD window is used. By default, `crossld` reads variants within each LD window by blocks. So reducing `--block-size` can significantly reduce memory usage. However, a very small block size (e.g., < 100) will also slow down the analysis. Using PLINK2's binary format can also reduce memory usage because it supports sparse representation of rare variants.

If there are multiple consecutive target variants, `crossld` can cache the dataset-2 blocks to be reused for the next target. The cache size is controlled by `--cache-limit` and `--cache-min-overlap`. A larger cache limit will reduce I/O and speed up the analysis, but it will also increase memory usage.

The default `--block-size 200 --cache-limit 5` is optimized for balanced memory usage and runtime on the 500k UK Biobank dataset with multiple consecutive target variants. For smaller datasets or less consecutive targets, increasing `--block-size` should speed up the analysis.


## Important notes

- **PLINK1 binary vs PLINK2 binary**: `crossld` supports both PLINK1 and PLINK2 binary formats. The two input datasets can be in different formats, but PLINK2 is always recommended for better performance. Genotype dosage is only supported in PLINK2 binary format.

- **Position sorting**: both sorted and unsorted datasets are supported. Because `crossld` will sort dataset-2 to index variants if it is unsorted, using a sorted dataset 2 will save time. However, since [pgenlib](https://github.com/chrchang/plink-ng/blob/master/2.0/Python/python_api.txt) may not always support unsorted multi-variant query, sorted dataset 2 may be required in the future.

- **Liftover**: it is possible that different variant callset were generated based on different genome builds. Since `crossld` does not require dataset 1 to be position sorted, it would be easier to only liftover dataset 1 by directly replacing the `.pvar` or `.bim` file with the lifted-over version rather than liftover the whole dataset.

## LICENSE
[MIT License](LICENSE)