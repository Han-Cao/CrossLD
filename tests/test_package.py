"""Public API tests for the :mod:`crossld` package after the rename."""

from __future__ import annotations

import crossld
from crossld import CrossLDError, Dataset, PairResult, VariantFilter, select_samples
from crossld.crossld import main
from crossld.logging_setup import LOGGER_NAME, configure_logging


def test_package_exposes_public_api():
    assert crossld.__version__ == "0.1.0"
    assert crossld.__name__ == "crossld"
    assert CrossLDError.__module__ == "crossld.plinkio"
    assert Dataset.__module__ == "crossld.plinkio"
    assert VariantFilter.__module__ == "crossld.varfilter"
    assert PairResult.__module__ == "crossld.ld"
    assert callable(select_samples) and callable(main)


def test_logger_hierarchy_uses_the_lowercase_package_name():
    assert LOGGER_NAME == "crossld"
    logger = configure_logging()
    assert logger.name == "crossld"
    for child in ("crossld.crossld", "crossld.plinkio", "crossld.blocks", "crossld.varfilter"):
        assert child.startswith(LOGGER_NAME + ".")
