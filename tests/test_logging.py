"""Tests for logging configuration and the INFO/DEBUG log contract."""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest

from crossld.crossld import main as cli_main
from crossld.logging_setup import LOGGER_NAME, configure_logging
from tests.conftest import make_bfile

_IDS = [f"S{i}" for i in range(6)]


@pytest.fixture(autouse=True)
def _reset_package_logger():
    """Drop the handler ``configure_logging`` installs after every test."""

    logger = logging.getLogger(LOGGER_NAME)
    yield
    for handler in list(logger.handlers):
        if getattr(handler, "_crossld_handler", False):
            logger.removeHandler(handler)
            handler.close()


@pytest.fixture()
def captured(caplog):
    """Attach caplog's handler to the package logger.

    ``configure_logging`` sets ``propagate = False`` so records never reach the
    root logger that :mod:`pytest`'s ``caplog`` watches by default.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def _package_handlers(logger):
    return [h for h in logger.handlers if getattr(h, "_crossld_handler", False)]


def _acceptance_argv(sample1_prefix, sample2_prefix, ld_snp_list, extract_list, out):
    return [
        "--pfile1", sample1_prefix,
        "--pfile2", sample2_prefix,
        "--ld-snp-list", ld_snp_list,
        "--extract", extract_list,
        "--ld-window-r2", "0.05",
        "--out", str(out),
    ]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def test_default_level_is_info():
    logger = configure_logging()
    assert logger.level == logging.INFO
    assert logger.name == LOGGER_NAME
    assert logger.propagate is False


def test_levels_accept_ints_and_upper_case_names():
    assert configure_logging(logging.DEBUG).level == logging.DEBUG
    assert configure_logging(logging.WARNING).level == logging.WARNING
    assert configure_logging("DEBUG").level == logging.DEBUG


def test_configure_logging_is_idempotent():
    logger = configure_logging()
    handlers = _package_handlers(logger)
    assert len(handlers) == 1
    configure_logging(logging.DEBUG)
    configure_logging(logging.INFO)
    assert _package_handlers(logger) == handlers


def test_handler_writes_to_stderr_by_default(capsys):
    configure_logging(logging.INFO)
    logging.getLogger(LOGGER_NAME).info("hello")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[INFO]: hello" in captured.err


def test_custom_stream_is_used():
    buffer = io.StringIO()
    configure_logging(logging.INFO, stream=buffer)
    logging.getLogger(LOGGER_NAME).info("to-buffer")
    assert "to-buffer" in buffer.getvalue()


# --------------------------------------------------------------------------- #
# INFO / DEBUG contract
# --------------------------------------------------------------------------- #

def test_info_logs_one_line_per_target_and_no_block_lines(
    captured, sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path
):
    result = cli_main(
        _acceptance_argv(
            sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path / "out.vcor"
        )
    )
    assert result == 0
    target_lines = [r for r in captured.records if r.getMessage().startswith("target ")]
    assert len(target_lines) == 4
    assert "candidates=" in captured.text
    assert "pairs_kept=" in captured.text
    assert "elapsed=" in captured.text
    assert not any(r.getMessage().startswith("block ") for r in captured.records)
    assert any(
        "done: 4 target(s), 788 pair(s) written" in r.getMessage()
        for r in captured.records
    )


def test_debug_logs_per_block_statistics(
    captured, sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path
):
    argv = _acceptance_argv(
        sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path / "out.vcor"
    )
    argv[0:0] = ["--debug", "--block-size", "150"]
    assert cli_main(argv) == 0

    block_lines = [r.getMessage() for r in captured.records if r.getMessage().startswith("block ")]
    assert block_lines
    first = block_lines[0]
    for field in (
        "block 1/",
        "variants=",
        "rows=",
        "mode=hardcall",
        "dense_variants=",
        "sparse_variants=",
        "pairs_kept=",
        "pairs_below_r2=",
        "skip_missing=",
        "skip_low_variance=",
        "elapsed=",
    ):
        assert field in first, f"{field!r} missing from {first!r}"


def test_skipped_pairs_are_logged(captured, tmp_path):
    """A zero-variance candidate yields no r and is reported as skipped."""

    variants = [("1", "1:100:A:G", 100, "A", "G")]
    prefix_a = make_bfile(
        tmp_path, "a", np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int8), _IDS, variants
    )
    prefix_b = make_bfile(
        tmp_path, "b", np.zeros((1, 6), dtype=np.int8), _IDS, variants
    )
    out_path = tmp_path / "out.vcor"
    result = cli_main(
        [
            "--debug",
            "--bfile1", prefix_a,
            "--bfile2", prefix_b,
            "--ld-snp", "1:100:A:G",
            "--ld-window-r2", "0",
            "--out", str(out_path),
        ]
    )
    assert result == 0
    assert any("skip reasons" in r.getMessage() for r in captured.records)
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "#CHROM_A\tPOS_A\tID_A\tCHROM_B\tPOS_B\tID_B\tUNPHASED_R2"
    ]


def test_quiet_suppresses_info(captured, sample1_prefix, sample2_prefix, tmp_path):
    result = cli_main(
        [
            "--quiet",
            "--pfile1", sample1_prefix,
            "--pfile2", sample2_prefix,
            "--ld-snp", "20:114311:G:T",
            "--out", str(tmp_path / "out.vcor"),
        ]
    )
    assert result == 0
    assert not [r for r in captured.records if r.levelno == logging.INFO]


def test_unmatched_target_warning_is_logged(
    captured, sample1_prefix, sample2_prefix, tmp_path
):
    result = cli_main(
        [
            "--pfile1", sample1_prefix,
            "--pfile2", sample2_prefix,
            "--ld-snp", "no-such-variant",
            "--out", str(tmp_path / "out.vcor"),
        ]
    )
    assert result == 0
    assert any("not found in dataset 1" in r.getMessage() for r in captured.records)
