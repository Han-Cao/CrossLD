"""Shared fixtures and synthetic PLINK builders for the crossld test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
FIXTURE_DIR = DATA_DIR


# --------------------------------------------------------------------------- #
# session fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return TESTS_DIR


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def sample1_prefix() -> str:
    return str(DATA_DIR / "1kg.sample1")


@pytest.fixture(scope="session")
def sample2_prefix() -> str:
    return str(DATA_DIR / "1kg.sample2")


@pytest.fixture(scope="session")
def ld_snp_list() -> str:
    return str(DATA_DIR / "ld_snp_list.txt")


@pytest.fixture(scope="session")
def extract_list() -> str:
    return str(DATA_DIR / "ld_extract_snp_list.txt")


@pytest.fixture(scope="session")
def truth_r2_path() -> Path:
    return DATA_DIR / "ld_results_r2.vcor"


@pytest.fixture(scope="session")
def truth_r_major_path() -> Path:
    return DATA_DIR / "ld_results_r_major.vcor"


@pytest.fixture(scope="session")
def truth_r_ref_path() -> Path:
    return DATA_DIR / "ld_results_r_ref.vcor"


@pytest.fixture(scope="session")
def loaded_pair(sample1_prefix, sample2_prefix):
    """The curated real datasets, loaded with their shared sample subset.

    Session-scoped because opening a ``.pgen`` is the expensive part of the
    suite; tests must not mutate the datasets (apply their own filters on
    copies or synthetic data).
    """

    from crossld.plinkio import load_two_datasets

    ds1, ds2 = load_two_datasets(
        sample1_prefix, "pfile", False, sample2_prefix, "pfile", False
    )
    yield ds1, ds2
    ds1.close()
    ds2.close()


# --------------------------------------------------------------------------- #
# metadata / list writers
# --------------------------------------------------------------------------- #

def write_fam(path: Path, sample_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(f"0\t{iid}\t0\t0\t0\t-9\n" for iid in sample_ids)


def write_bim(path: Path, variants: list[tuple[str, str, int, str, str]]) -> None:
    """``variants`` entries are ``(chrom, variant_id, pos, ref, alt)``.

    In ``.bim`` the second-to-last column is ALT and the last column is REF.
    """

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(
            f"{chrom}\t{vid}\t0\t{pos}\t{alt}\t{ref}\n" 
            for chrom, vid, pos, ref, alt in variants
        )


def write_psam(path: Path, sample_ids: list[str]) -> None:
    """Write a minimal ``.psam`` with an ``#IID`` header."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#IID\tSEX\n")
        handle.writelines(f"{iid}\tNA\n" for iid in sample_ids)


def write_pvar(path: Path, variants: list[tuple[str, str, int, str, str]]) -> None:
    """Write a minimal ``.pvar`` (``variants`` are ``(chrom, id, pos, ref, alt)``)."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        handle.writelines(
            f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\n" 
            for chrom, vid, pos, ref, alt in variants
        )


def write_bed(path: Path, genotypes: np.ndarray, sample_ids: list[str]) -> None:
    """Write a PLINK 1 ``.bed`` from a variant-major 0/1/2/-9 matrix.

    ``pgenlib`` reads each sample's 2-bit code little-endian from bit ``2*i`` of
    the 1-byte-per-4-samples block, with the code mapping
    ``{2: 0b00, -9: 0b01, 1: 0b10, 0: 0b11}`` (verified empirically).
    """

    sample_ct = len(sample_ids)
    bytes_per_variant = (sample_ct + 3) // 4
    codes = {2: 0b00, -9: 0b01, 1: 0b10, 0: 0b11}
    with open(path, "wb") as handle:
        handle.write(bytes([0x6C, 0x1B, 0x01]))
        for row in genotypes:
            payload = bytearray(bytes_per_variant)
            for i, value in enumerate(row):
                payload[i // 4] |= codes[int(value)] << (2 * (i % 4))
            handle.write(bytes(payload))


def write_extract(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{vid}\n" for vid in ids), encoding="utf-8")


def write_bed_intervals(path: Path, intervals: list[tuple[str, int, int]]) -> None:
    path.write_text(
        "".join(f"{chrom}\t{start}\t{end}\n" for chrom, start, end in intervals),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# synthetic dataset builders
# --------------------------------------------------------------------------- #

def make_bfile(
    tmp_path: Path,
    name: str,
    genotypes: np.ndarray,
    sample_ids: list[str],
    variants: list[tuple[str, str, int, str, str]],
):
    """Create a synthetic ``.bed/.bim/.fam`` dataset and return its prefix."""

    prefix = tmp_path / name
    write_fam(Path(f"{prefix}.fam"), sample_ids)
    write_bim(Path(f"{prefix}.bim"), variants)
    write_bed(Path(f"{prefix}.bed"), genotypes, sample_ids)
    return str(prefix)


def make_pfile(
    tmp_path: Path,
    name: str,
    genotypes: np.ndarray,
    sample_ids: list[str],
    variants: list[tuple[str, str, int, str, str]],
):
    """Create a synthetic ``.pgen/.pvar/.psam`` dataset and return its prefix."""

    import pgenlib

    prefix = tmp_path / name
    write_psam(Path(f"{prefix}.psam"), sample_ids)
    write_pvar(Path(f"{prefix}.pvar"), variants)
    writer = pgenlib.PgenWriter(
        f"{prefix}.pgen".encode(),
        len(sample_ids),
        variant_ct=len(variants),
        nonref_flags=None,
    )
    writer.append_biallelic_batch(np.ascontiguousarray(genotypes, dtype=np.int8))
    writer.close()
    return str(prefix)


def make_dosage_pfile(
    tmp_path: Path,
    name: str,
    dosages: np.ndarray,
    sample_ids: list[str],
    variants: list[tuple[str, str, int, str, str]],
):
    """Create a synthetic dosage ``.pgen/.pvar/.psam`` dataset."""

    import pgenlib

    prefix = tmp_path / name
    write_psam(Path(f"{prefix}.psam"), sample_ids)
    write_pvar(Path(f"{prefix}.pvar"), variants)
    writer = pgenlib.PgenWriter(
        f"{prefix}.pgen".encode(),
        len(sample_ids),
        variant_ct=len(variants),
        nonref_flags=None,
        dosage_present=True,
    )
    writer.append_dosages_batch(np.ascontiguousarray(dosages, dtype=np.float32))
    writer.close()
    return str(prefix)


# --------------------------------------------------------------------------- #
# independent math oracle
# --------------------------------------------------------------------------- #

def pearson_reference(target: np.ndarray, row: np.ndarray) -> float:
    """Pairwise-valid Pearson r via :func:`numpy.corrcoef` (independent oracle).

    Missing calls are negative values on either side. Returns ``NaN`` when
    fewer than three samples are usable or either side has zero variance, the
    same contract as :func:`crossld.ld.pearson_pairwise`.
    """

    target = np.asarray(target, dtype=float)
    row = np.asarray(row, dtype=float)
    valid = (target >= 0) & (row >= 0)
    if int(valid.sum()) < 3:
        return float("nan")
    x = target[valid]
    y = row[valid]
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


__all__ = [
    "DATA_DIR",
    "FIXTURE_DIR",
    "TESTS_DIR",
    "make_bfile",
    "make_dosage_pfile",
    "make_pfile",
    "pearson_reference",
    "write_bed",
    "write_bed_intervals",
    "write_bim",
    "write_extract",
    "write_fam",
    "write_psam",
    "write_pvar",
]
