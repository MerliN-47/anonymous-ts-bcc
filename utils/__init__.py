"""
[Anonymous] TS-RAG-2 / CAVIR Utilities Package
Double-Blind Compliant Modular Architecture
"""

from .run_config import add_all_flags
from .metrics import (
    MSE,
    MAE,
    RMSE,
    MAPE,
    MSPE,
    SMAPE,
    ND,
    RSE,
    CORR,
    pinball_loss,
    CRPS,
    WQL,
    MASE,
    Coverage,
    DieboldMarianoTest,
    metric,
    compute_distributional_metrics
)
from .tost_equivalence import (
    compute_tost,
    test_representation_parity
)
from .faiss_index import (
    TimeSeriesFAISSIndex,
    partition_memory
)

__all__ = [
    "add_all_flags",
    "MSE",
    "MAE",
    "RMSE",
    "MAPE",
    "MSPE",
    "SMAPE",
    "ND",
    "RSE",
    "CORR",
    "pinball_loss",
    "CRPS",
    "WQL",
    "MASE",
    "Coverage",
    "DieboldMarianoTest",
    "metric",
    "compute_distributional_metrics",
    "compute_tost",
    "test_representation_parity",
    "TimeSeriesFAISSIndex",
    "partition_memory"
]
