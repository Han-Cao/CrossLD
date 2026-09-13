"""Tests for logging configuration and the INFO/DEBUG log contract."""

from __future__ import annotations

import logging

import pytest

from crossld.crossld import main as cli_main
from crossld.logging_setup import LOGGER_NAME, configure_logging


@pytest.fixture(autouse=True)
def _reset_package_logger():
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


def test_default_level_is_info():
    logger = configure_logging()
    assert logger.level == logging.INFO
    assert logger.name == LOGGER_NAME
    assert logger.propagate is False


def test_verbose_selects_debug_and_quiet_selects_warning():
    assert configure_logging(logging.DEBUG).level == logging.DEBUG
    assert configure_logging(logging.WARNING).level == logging.WARNING
    assert configure_logging("debug").level == logging.DEBUG


def _package_handlers(logger):
    return [h for h in logger.handlers if getattr(h, "_crossld_handler", False)]


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
    assert "hello" in captured.err
    assert "INFO [crossld]" in captured.err


def test_info_logs_one_line_per_target_and_no_block_lines(
    captured, sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path
):
    caplog = captured
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = cli_main(
            [
                "--pfile1",
                sample1_prefix,
                "--pfile2",
                sample2_prefix,
                "--ld-snp-list",
                ld_snp_list,
                "--extract",
                extract_list,
                "--ld-window-r2",
                "0.05",
                "--out",
                str(tmp_path / "out.vcor"),
            ]
        )
    assert result == 0
    target_lines = [r for r in caplog.records if r.message.startswith("target ")]
    assert len(target_lines) == 4
    assert "candidates=" in caplog.text
    assert "pairs_kept=" in caplog.text
    assert "elapsed=" in caplog.text
    assert not any(r.message.startswith("block ") for r in caplog.records)
    assert any("done: 4 target(s), 784 pair(s) written" in r.message for r in caplog.records)


def test_debug_logs_per_block_statistics(
    captured, sample1_prefix, sample2_prefix, ld_snp_list, extract_list, tmp_path
):
    caplog = captured
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = cli_main(
            [
                "-v",
                "--pfile1",
                sample1_prefix,
                "--pfile2",
                sample2_prefix,
                "--ld-snp-list",
                ld_snp_list,
                "--extract",
                extract_list,
                "--ld-window-r2",
                "0.05",
                "--block-size",
                "150",
                "--out",
                str(tmp_path / "out.vcor"),
            ]
        )
    assert result == 0
    block_lines = [r.message for r in caplog.records if r.message.startswith("block ")]
    assert block_lines
    first = block_lines[0]
    for field in (
        "block 1/",
        "variants=",
        "rows=",
        "samples=100",
        "mode=hardcall",
        "sparse_variants=",
        "sparse_fallbacks=",
        "pairs_evaluated=",
        "pairs_kept=",
        "pairs_below_r2=",
        "pairs_skipped=",
        "missing_frac=",
        "elapsed=",
    ):
        assert field in first, f"{field!r} missing from {first!r}"
    assert any("skip reasons" in r.message for r in caplog.records)


def test_quiet_suppresses_info(captured, sample1_prefix, sample2_prefix, tmp_path):
    caplog = captured
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = cli_main(
            [
                "--pfile1",
                sample1_prefix,
                "--pfile2",
                sample2_prefix,
                "--ld-snp",
                "20:114311:G:T",
                "--out",
                str(tmp_path / "out.vcor"),
            ]
        )
    assert result == 0
    assert not [r for r in caplog.records if r.levelno == logging.INFO]


def test_unmatched_target_warning_is_logged(captured, sample1_prefix, sample2_prefix, tmp_path):
    caplog = captured
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = cli_main(
            [
                "--pfile1",
                sample1_prefix,
                "--pfile2",
                sample2_prefix,
                "--ld-snp",
                "no-such-variant",
                "--out",
                str(tmp_path / "out.vcor"),
            ]
        )
    assert result == 0
    assert any("matched no variant" in r.message for r in caplog.records)
