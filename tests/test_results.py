"""Tests for the output formatter in :mod:`crossld.results`."""

from __future__ import annotations

import pytest

from crossld.results import (
    header_for,
)


def test_header_for_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown allele label"):
        header_for(with_r=True, allele_label="NOPE")

