"""Real-data LD acceptance tests: byte-for-byte reproduction of the references.

Three curated files in ``tests/data/`` pin the statistics over four dataset-1
targets (``tests/data/ld_snp_list.txt``) with ``--extract`` and
``--ld-window-r2 0.05``:

* ``ld_results_r2.vcor``      -- ``UNPHASED_R2`` (the default output shape)
* ``ld_results_r_major.vcor`` -- ``--r``: signed ``UNPHASED_R`` with the coding
                                 allele per side (``MAJ_A``/``MAJ_B``)
* ``ld_results_r_ref.vcor``   -- ``--r --ref-based``: same, with
                                 ``REF_A``/``REF_B``

Each file has 788 rows: the 784 target x other-variant pairs plus the four
self-pairs (a target that is also present in dataset 2, correlated against
itself with ``r``/``r²`` of ``1``).
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
from crossld.ld import centered_moment_ld_dense, major_allele_flipped
from crossld.plinkio import (
    load_dataset,
    load_two_datasets,
    parse_sample,
    read_ld_snp_list,
    resolve_targets,
)
from crossld.varfilter import VariantFilter

SRC = Path(__file__).resolve().parent.parent / "src"

ACCEPTANCE_ARGS = ["--ld-window-r2", "0.05"]

# Values are compared at the 6-significant-digit precision the truth files use.
VALUE_TOLERANCE = 6e-7

# Every reference file has this many rows, all of them reproducible.
TRUTH_ROWS = 788


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
        self.ds1, self.ds2 = ds1, ds2
        data = _data_dir()
        self.targets = resolve_targets(ds1, snp_list=str(data / "ld_snp_list.txt"))
        self.kept = np.flatnonzero(
            VariantFilter(extract=str(data / "ld_extract_snp_list.txt")).build_mask(ds2)
        )
        self._reader1 = BlockReader(ds1, force_dense=True)
        self._reader2 = BlockReader(ds2, force_dense=True)
        # Dataset-1 vectors are permuted into dataset-2 sample order so both sides
        # are aligned for pearson_pairwise.
        bridge = ds1.idx_bridge
        self.a = {
            ti: self._reader1.read([ti]).dense[0][bridge].copy() for ti in self.targets
        }
        self.b = {int(bi): self._reader2.read([bi]).dense[0].copy() for bi in self.kept}

    def code(self, vec: np.ndarray, mode: str) -> np.ndarray:
        if mode == "alt":
            return vec
        if mode == "ref":
            return np.where(vec < 0, vec, 2.0 - vec)
        if mode == "major":
            return np.where(vec < 0, vec, 2.0 - vec) if major_allele_flipped(vec) else vec
        raise ValueError(mode)

    def r(self, ti: int, bi: int, mode: str) -> float:
        r, _ = centered_moment_ld_dense(
            self.code(self.a[ti], mode), self.code(self.b[bi], mode).reshape(1, -1)
        )
        return float(r[0])

    def ref(self, ti: int, bi: int) -> tuple[str, str]:
        return self.ds1.ref_allele(ti), self.ds2.ref_allele(bi)

    def major(self, ti: int, bi: int) -> tuple[str, str]:
        maj_a = (
            self.ds1.alt_allele(ti)
            if major_allele_flipped(self.a[ti])
            else self.ds1.ref_allele(ti)
        )
        maj_b = (
            self.ds2.alt_allele(bi)
            if major_allele_flipped(self.b[bi])
            else self.ds2.ref_allele(bi)
        )
        return maj_a, maj_b


@pytest.fixture(scope="module")
def raw_vectors(sample1_prefix, sample2_prefix):
    ds1, ds2 = load_two_datasets(
        sample1_prefix, "pfile", False, sample2_prefix, "pfile", False
    )
    yield _RawVectors(ds1, ds2)
    ds1.close()
    ds2.close()


@pytest.fixture(scope="module")
def current_b_ids(sample2_prefix):
    sample_ids = parse_sample(f"{sample2_prefix}.psam", "pfile")
    ds2 = load_dataset(sample2_prefix, "pfile", sample_ids)
    try:
        return set(ds2.ids)
    finally:
        ds2.close()


def _comparable(truth_rows, current_b, b_col: int = 5):
    """Truth rows whose dataset-2 variant still exists in the fixture."""

    return [row for row in truth_rows if row[b_col] in current_b]


def _pvar_variants(path: Path) -> dict[str, tuple[str, int]]:
    """``id -> (chrom, pos)`` for a ``.pvar``, in file order (test-local oracle)."""

    variants: dict[str, tuple[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        variants[fields[2]] = (fields[0], int(fields[1]))
    return variants


def _window_expected(
    truth_rows,
    variants: dict[str, tuple[str, int]],
    extract_ids: list[str],
    *,
    kb: float | None = None,
    count: int = 0,
) -> list[str]:
    """Truth lines that survive the same window the CLI is asked to apply.

    The nearest-``count`` pool is rebuilt from the ``.pvar`` positions (not from
    the truth rows): a nearer variant whose ``r^2`` fell below the reporting
    threshold still occupies a count slot and pushes a truth row out. The pool
    keeps ``.pvar`` file order and is stably sorted by distance, mirroring the
    implementation's tie-break; ``kb`` is applied before ``count``, as in
    :func:`crossld.plinkio.candidate_indices`.
    """

    extracted = set(extract_ids)
    file_order = [vid for vid in variants if vid in extracted]
    eligible: dict[tuple[str, int], set[str]] = {}
    for row in truth_rows:
        key = (row[0], int(row[1]))
        if key in eligible:
            continue
        chrom_a, pos_a = key
        pool = [
            vid
            for vid in file_order
            if variants[vid][0] == chrom_a
            and (kb is None or abs(variants[vid][1] - pos_a) <= kb * 1000)
        ]
        if count > 0:
            pool = sorted(pool, key=lambda vid: abs(variants[vid][1] - pos_a))[:count]
        eligible[key] = set(pool)

    return [
        "\t".join(row)
        for row in truth_rows
        if row[5] in eligible[(row[0], int(row[1]))]
    ]


def _run_windowed(sample1_prefix, sample2_prefix, out_path, extra) -> str:
    """Run the acceptance CLI with extra window flags and return the output text."""

    assert cli_main(
        _acceptance_args(sample1_prefix, sample2_prefix, out_path, extra=extra)
    ) == 0
    return out_path.read_text(encoding="utf-8")


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


@pytest.mark.parametrize(
    "truth_fixture, extra",
    [
        ("truth_r2_path", []),
        ("truth_r_major_path", ["--r"]),
        ("truth_r_ref_path", ["--r", "--ref-based"]),
    ],
    ids=["r2", "r_major", "r_ref"],
)
def test_cache_limit_preserves_acceptance_output(
    tmp_path, sample1_prefix, sample2_prefix, request, truth_fixture, extra
):
    """A large read cache must not perturb any reference output."""

    truth_path = request.getfixturevalue(truth_fixture)
    out_path = tmp_path / "out_cache.vcor"
    assert cli_main(
        _acceptance_args(
            sample1_prefix,
            sample2_prefix,
            out_path,
            extra=[*extra, "--cache-limit", "100000"],
        )
    ) == 0
    assert out_path.read_text(encoding="utf-8") == truth_path.read_text(encoding="utf-8")


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
# windows: the truth file is a superset of any smaller window
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kb", [0.0, 0.5, 2.0, 10.0])
def test_ld_window_kb_matches_sliced_truth(
    tmp_path, sample1_prefix, sample2_prefix, truth_r2_path, extract_list, kb
):
    """A physical window keeps exactly the truth rows within +/- kb."""

    got = _run_windowed(
        sample1_prefix,
        sample2_prefix,
        tmp_path / f"out_kb_{kb}.vcor",
        ["--ld-window-kb", str(kb)],
    )
    header, truth = _parse(truth_r2_path)
    variants = _pvar_variants(Path(f"{sample2_prefix}.pvar"))
    expected = [header] + _window_expected(
        truth, variants, read_ld_snp_list(str(extract_list)), kb=kb
    )
    assert got == "\n".join(expected) + "\n"


@pytest.mark.parametrize("count", [1, 2, 5, 50, 300])
def test_ld_window_count_matches_sliced_truth(
    tmp_path, sample1_prefix, sample2_prefix, truth_r2_path, extract_list, count
):
    """A count window keeps exactly the truth rows whose B variant is nearest."""

    got = _run_windowed(
        sample1_prefix,
        sample2_prefix,
        tmp_path / f"out_n_{count}.vcor",
        ["--ld-window", str(count)],
    )
    header, truth = _parse(truth_r2_path)
    variants = _pvar_variants(Path(f"{sample2_prefix}.pvar"))
    expected = [header] + _window_expected(
        truth, variants, read_ld_snp_list(str(extract_list)), count=count
    )
    assert got == "\n".join(expected) + "\n"


@pytest.mark.parametrize("kb,count", [(20.0, 10), (0.5, 2), (1.0, 5)])
def test_ld_window_kb_and_count_combine(
    tmp_path, sample1_prefix, sample2_prefix, truth_r2_path, extract_list, kb, count
):
    """The physical bound and the count bound compose (kb first, then count)."""

    got = _run_windowed(
        sample1_prefix,
        sample2_prefix,
        tmp_path / f"out_kb_{kb}_n_{count}.vcor",
        ["--ld-window-kb", str(kb), "--ld-window", str(count)],
    )
    header, truth = _parse(truth_r2_path)
    variants = _pvar_variants(Path(f"{sample2_prefix}.pvar"))
    expected = [header] + _window_expected(
        truth, variants, read_ld_snp_list(str(extract_list)), kb=kb, count=count
    )
    assert got == "\n".join(expected) + "\n"


def test_ld_window_one_keeps_only_self_pairs(
    tmp_path, sample1_prefix, sample2_prefix
):
    """``--ld-window 1`` keeps the target itself: the truly nearest variant."""

    got = _run_windowed(
        sample1_prefix,
        sample2_prefix,
        tmp_path / "out_self.vcor",
        ["--ld-window", "1"],
    )
    lines = got.splitlines()
    rows = [line.split("\t") for line in lines[1:]]
    assert len(rows) == 4
    assert {row[2] for row in rows} == {
        "20:114311:G:T",
        "20:144480:C:T",
        "20:166845:T:C",
        "20:199547:C:T",
    }
    assert all(row[2] == row[5] for row in rows)
    assert all(float(row[6]) == 1.0 for row in rows)


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
        # ds.indices_of return a tuple, where the first element is a list
        # to get index of one variant, do [0][0]
        r = raw_vectors.r(ds1.indices_of([row[2]])[0][0], ds2.indices_of([row[5]])[0][0], "alt")
        assert r * r == pytest.approx(float(row[6]), abs=1e-5), f"{(row[2], row[5])}"
        checked += 1
    assert checked == TRUTH_ROWS


def test_r_ref_values_and_allele_columns(raw_vectors, truth_r_ref_path, current_b_ids):
    """Reference orientation reproduces UNPHASED_R and the REF columns."""

    ds1, ds2 = raw_vectors.ds1, raw_vectors.ds2
    _, truth = _parse(truth_r_ref_path)
    checked = 0
    for row in _comparable(truth, current_b_ids, b_col=6):
        ti, bi = ds1.indices_of([row[2]])[0][0], ds2.indices_of([row[6]])[0][0]
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
        ti, bi = ds1.indices_of([row[2]])[0][0], ds2.indices_of([row[6]])[0][0]
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
        ti, bi = ds1.indices_of([row[2]])[0][0], ds2.indices_of([row[6]])[0][0]
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
