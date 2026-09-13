"""crossld: cross-dataset linkage disequilibrium for PLINK 1/2 genotype data."""

from .ld import PairResult, compute_pair
from .plinkio import CrossLDError, Dataset, SelectionSummary, select_samples
from .varfilter import VariantFilter

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CrossLDError",
    "Dataset",
    "SelectionSummary",
    "select_samples",
    "VariantFilter",
    "PairResult",
    "compute_pair",
]
