"""Real-data LD acceptance tests: byte-for-byte reproduction of the references.

Three curated files in ``tests/data/`` pin the statistics over four dataset-1
targets (``tests/data/ld_snp_list.txt``) with ``--extract`` and
``--ld-window-r2 0.05``:

* ``ld_results_r2.vcor``      -- ``UNPHASED_R2`` (the default output shape)
* ``ld_results_r_major.vcor`` -- ``--r``: signed ``UNPHASED_R`` with the coding
                                 allele per side (``MAJ_A``/``MAJ_B``)
* ``ld_results_r_ref.vcor``   -- ``--r --ref-based``: same, with
                                 ``REF_A``/``REF_B``
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crossld.blocks import BlockReader
from crossld.crossld import main as cli_main
from crossld.ld import major_allele_flipped, pearson_pairwise
from crossld.plinkio import load_dataset, resolve_targets, select_samples
from crossld.varfilter import VariantFilter

SRC = Path(__file__).resolve().parent.parent / "src"

ACCEPTANCE_ARGS = ["--ld-window-r2", "0.05"]

# Values are compared at the 6-significant-digit precision the truth files use.
VALUE_TOLERANCE = 6e-7

# Every reference file has this many rows, all of them reproducible.
TRUTH_ROWS = 784


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _parse(path: Path) -> tuple[str, list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0], [line.split("\t") for line in lines[1:] if line]


def _run_cli(argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "crossld", *argv],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _acceptance_args(sample1_prefix, sample2_prefix, out_path, extra=()):
    data = _data_dir()
    return [
        "--pfile1",
        sample1_prefix,
        "--pfile2",
        sample2_prefix,
        "--ld-snp-list",
        str(data / "ld_snp_list.txt"),
        "--extract",
        str(data / "ld_extract_snp_list.txt"),
        *ACCEPTANCE_ARGS,
        *extra,
        "--out",
        str(out_path),
    ]


class _RawVectors:
    """Raw ALT-allele dosage vectors, so tests can orient them explicitly."""

    def __init__(self, ds1, ds2):
        subset1, subset2, _ = select_samples(ds1, ds2)
        ds1.set_sample_subset(subset1)
        ds2.set_sample_subset(subset2)
        self.ds1, self.ds2 = ds1, ds2
        data = _data_dir()
        self.targets = resolve_targets(ds1, snp_list=str(data / "ld_snp_list.txt"))
        self.kept = np.flatnonzero(
            VariantFilter(extract=str(data / "ld_extract_snp_list.txt")).build_mask(ds2)
        )
        self._reader1 = BlockReader(ds1, sparse="never")
        self._reader2 = BlockReader(ds2, sparse="never")
        self.a = {ti: self._reader1.read([ti])[0][0].copy() for ti in self.targets}
        self.b = {int(bi): self._reader2.read([bi])[0][0].copy() for bi in self.kept}

    def code(self, vec: np.ndarray, mode: str) -> np.ndarray:
        if mode == "alt":
            return vec
        if mode == "ref":
            return np.where(vec < 0, vec, 2.0 - vec)
        if mode == "major":
            return np.where(vec < 0, vec, 2.0 - vec) if major_allele_flipped(vec) else vec
        raise ValueError(mode)

    def r(self, ti: int, bi: int, mode: str) -> float:
        r, _ = pearson_pairwise(
            self.code(self.a[ti], mode), self.code(self.b[bi], mode).reshape(1, -1)
        )
        return float(r[0])

    def ref(self, ti: int, bi: int) -> tuple[str, str]:
        return self.ds1.ref[ti], self.ds2.ref[bi]

    def major(self, ti: int, bi: int) -> tuple[str, str]:
        maj_a = self.ds1.ref[ti] if major_allele_flipped(self.a[ti]) else self.ds1.alt[ti]
        maj_b = self.ds2.ref[bi] if major_allele_flipped(self.b[bi]) else self.ds2.alt[bi]
        return maj_a, maj_b


@pytest.fixture(scope="module")
def raw_vectors(sample1_prefix, sample2_prefix):
    ds1 = load_dataset(sample1_prefix, "dataset1", "pfile")
    ds2 = load_dataset(sample2_prefix, "dataset2", "pfile")
    yield _RawVectors(ds1, ds2)
    ds1.close()
    ds2.close()


@pytest.fixture(scope="module")
def current_b_ids(sample2_prefix):
    ds2 = load_dataset(sample2_prefix, "dataset2", "pfile")
    try:
        return set(ds2.ids)
    finally:
        ds2.close()


def _comparable(truth_rows, current_b, b_col: int = 5):
    """Truth rows whose dataset-2 variant still exists in the fixture."""

    return [row for row in truth_rows if row[b_col] in current_b]


# --------------------------------------------------------------------------- #
# CLI output shape 1: r^2
# --------------------------------------------------------------------------- #

def test_r2_cli_output_is_byte_identical(
    tmp_path, sample1_prefix, sample2_prefix, truth_r2_path, current_b_ids
):
    out_path = tmp_path / "out_r2.vcor"
    assert cli_main(_acceptance_args(sample1_prefix, sample2_prefix, out_path)) == 0

    header, actual = _parse(out_path)
    truth_header, truth = _parse(truth_r2_path)
    assert header == truth_header == (
        "#CHROM_A\tPOS_A\tID_A\tCHROM_B\tPOS_B\tID_B\tUNPHASED_R2"
    )
    assert len(truth) == TRUTH_ROWS
    assert len(_comparable(truth, current_b_ids)) == TRUTH_ROWS, "truth has stale B ids"
    assert len(actual) == TRUTH_ROWS
    assert out_path.read_text(encoding="utf-8") == truth_r2_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI output shape 2: signed r with coding-allele columns
# --------------------------------------------------------------------------- #

def test_r_major_cli_output_is_byte_identical(
    tmp_path, sample1_prefix, sample2_prefix, truth_r_major_path
):
    out_path = tmp_path / "out_r_major.vcor"
    assert (
        cli_main(
            _acceptance_args(sample1_prefix, sample2_prefix, out_path, extra=["--r"])
        )
        == 0
    )

    header, actual = _parse(out_path)
    truth_header, truth = _parse(truth_r_major_path)
    assert header == truth_header == (
        "#CHROM_A\tPOS_A\tID_A\tMAJ_A\tCHROM_B\tPOS_B\tID_B\tMAJ_B\tUNPHASED_R"
    )
    assert len(actual) == len(truth) == TRUTH_ROWS
    assert out_path.read_text(encoding="utf-8") == truth_r_major_path.read_text(
        encoding="utf-8"
    )


def test_r_ref_cli_output_is_byte_identical(
    tmp_path, sample1_prefix, sample2_prefix, truth_r_ref_path
):
    out_path = tmp_path / "out_r_ref.vcor"
    assert (
        cli_main(
            _acceptance_args(
                sample1_prefix,
                sample2_prefix,
                out_path,
                extra=["--r", "--ref-based"],
            )
        )
        == 0
    )

    header, actual = _parse(out_path)
    truth_header, truth = _parse(truth_r_ref_path)
    assert header == truth_header == (
        "#CHROM_A\tPOS_A\tID_A\tREF_A\tCHROM_B\tPOS_B\tID_B\tREF_B\tUNPHASED_R"
    )
    assert len(actual) == len(truth) == TRUTH_ROWS
    assert out_path.read_text(encoding="utf-8") == truth_r_ref_path.read_text(
        encoding="utf-8"
    )


def test_r_outputs_omit_the_r2_column(tmp_path, sample1_prefix, sample2_prefix):
    """The --r layout reports the coding allele instead of UNPHASED_R2."""

    for extra, label in ((["--r"], "MAJ"), (["--r", "--ref-based"], "REF")):
        out_path = tmp_path / f"out_{label}.vcor"
        assert (
            cli_main(
                _acceptance_args(sample1_prefix, sample2_prefix, out_path, extra=extra)
            )
            == 0
        )
        header, rows = _parse(out_path)
        assert "UNPHASED_R2" not in header
        assert header.split("\t")[3] == f"{label}_A"
        assert header.split("\t")[7] == f"{label}_B"
        assert all(len(row) == 9 for row in rows)


# --------------------------------------------------------------------------- #
# independent value checks (no file formatting involved)
# --------------------------------------------------------------------------- #

def test_r2_values_match_alt_coded_correlation(
    raw_vectors, truth_r2_path, current_b_ids
):
    """r^2 is coding independent, so recompute it from the raw ALT vectors."""

    ds1, ds2 = raw_vectors.ds1, raw_vectors.ds2
    _, truth = _parse(truth_r2_path)
    checked = 0
    for row in _comparable(truth, current_b_ids):
        r = raw_vectors.r(ds1.index_of(row[2]), ds2.index_of(row[5]), "alt")
        assert r * r == pytest.approx(float(row[6]), abs=1e-5), f"{(row[2], row[5])}"
        checked += 1
    assert checked == TRUTH_ROWS


def test_r_ref_values_and_allele_columns(raw_vectors, truth_r_ref_path, current_b_ids):
    """Reference orientation reproduces UNPHASED_R and the REF columns."""

    ds1, ds2 = raw_vectors.ds1, raw_vectors.ds2
    _, truth = _parse(truth_r_ref_path)
    checked = 0
    for row in _comparable(truth, current_b_ids, b_col=6):
        ti, bi = ds1.index_of(row[2]), ds2.index_of(row[6])
        assert (row[3], row[7]) == raw_vectors.ref(ti, bi), f"REF columns {(row[2], row[6])}"
        assert raw_vectors.r(ti, bi, "ref") == pytest.approx(
            float(row[8]), abs=1e-5
        ), f"{(row[2], row[6])}"
        checked += 1
    assert checked == TRUTH_ROWS


def test_r_major_values_and_allele_columns(
    raw_vectors, truth_r_major_path, current_b_ids
):
    """Major-allele detection reproduces UNPHASED_R and the MAJ columns."""

    ds1, ds2 = raw_vectors.ds1, raw_vectors.ds2
    _, truth = _parse(truth_r_major_path)
    checked = 0
    for row in _comparable(truth, current_b_ids, b_col=6):
        ti, bi = ds1.index_of(row[2]), ds2.index_of(row[6])
        assert (row[3], row[7]) == raw_vectors.major(ti, bi), f"MAJ columns {(row[2], row[6])}"
        assert raw_vectors.r(ti, bi, "major") == pytest.approx(
            float(row[8]), abs=1e-5
        ), f"{(row[2], row[6])}"
        checked += 1
    assert checked == TRUTH_ROWS


def test_r_major_and_ref_agree_up_to_sign(raw_vectors, current_b_ids, truth_r_major_path):
    """Switching orientation can only flip the sign, never the magnitude."""

    ds1, ds2 = raw_vectors.ds1, raw_vectors.ds2
    _, truth = _parse(truth_r_major_path)
    for row in _comparable(truth, current_b_ids, b_col=6):
        ti, bi = ds1.index_of(row[2]), ds2.index_of(row[6])
        major = raw_vectors.r(ti, bi, "major")
        ref = raw_vectors.r(ti, bi, "ref")
        assert abs(major) == pytest.approx(abs(ref), abs=1e-9)


# --------------------------------------------------------------------------- #
# stream hygiene
# --------------------------------------------------------------------------- #

def test_cli_subprocess_stdout_has_only_results(sample1_prefix, sample2_prefix):
    """Logs go to stderr, so stdout carries the header plus the results only."""

    data = _data_dir()
    proc = _run_cli(
        [
            "--pfile1",
            sample1_prefix,
            "--pfile2",
            sample2_prefix,
            "--ld-snp-list",
            str(data / "ld_snp_list.txt"),
            "--extract",
            str(data / "ld_extract_snp_list.txt"),
            *ACCEPTANCE_ARGS,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    out_lines = proc.stdout.splitlines()
    assert out_lines[0] == "#CHROM_A\tPOS_A\tID_A\tCHROM_B\tPOS_B\tID_B\tUNPHASED_R2"
    assert len(out_lines) == TRUTH_ROWS + 1  # header + rows
    assert "INFO" in proc.stderr and "DEBUG" not in proc.stderr
