"""Shared fixtures and synthetic PLINK builders for the crossld test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
FIXTURE_DIR = DATA_DIR


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


def write_fam(path: Path, sample_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for iid in sample_ids:
            handle.write(f"0\t{iid}\t0\t0\t0\t-9\n")


def write_bim(path: Path, variants: list[tuple[str, str, int, str, str]]) -> None:
    """``variants`` entries are ``(chrom, variant_id, pos, ref, alt)``.

    In ``.bim`` the second-to-last column is ALT and the last column is REF.
    """

    with open(path, "w", encoding="utf-8") as handle:
        for chrom, vid, pos, ref, alt in variants:
            handle.write(f"{chrom}\t{vid}\t0\t{pos}\t{alt}\t{ref}\n")


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
    """Create a synthetic ``.pgen/.bim/.fam`` dataset (pgen + bim metadata)."""

    import pgenlib

    prefix = tmp_path / name
    write_fam(Path(f"{prefix}.fam"), sample_ids)
    write_bim(Path(f"{prefix}.bim"), variants)
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
    """Create a synthetic dosage ``.pgen/.bim/.fam`` dataset."""

    import pgenlib

    prefix = tmp_path / name
    write_fam(Path(f"{prefix}.fam"), sample_ids)
    write_bim(Path(f"{prefix}.bim"), variants)
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


__all__ = [
    "DATA_DIR",
    "FIXTURE_DIR",
    "TESTS_DIR",
    "make_bfile",
    "make_dosage_pfile",
    "make_pfile",
    "write_bed",
    "write_bed_intervals",
    "write_bim",
    "write_extract",
    "write_fam",
]
