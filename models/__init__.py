"""
[Anonymous] TS-RAG-2 / CAVIR Models Package
Double-Blind Compliant Modular Architecture
"""

from .backbone_interface import (
    BaseTSFMAdapter,
    ChronosBoltAdapter,
    Chronos2Adapter,
    MoiraiAdapter,
    TimesFMAdapter,
    get_backbone_adapter
)
from .injection_heads import (
    LatentSpaceARM,
    OutputSpaceBarycenter,
    TokenSpaceFormatter,
    UnifiedRetrievalInjector
)
from .covariate_and_text import (
    CovariateQueryEmbedder,
    ChannelBlockEncoder,
    FrozenTextEncoderAdapter,
    MultimodalTextAdapter
)
from .scaling_manifold import (
    ScalingLawFitter,
    ManifoldRankEstimator,
    fit_scaling_power_law,
    estimate_two_nn_dimension,
    estimate_pca_effective_rank
)

__all__ = [
    "BaseTSFMAdapter",
    "ChronosBoltAdapter",
    "Chronos2Adapter",
    "MoiraiAdapter",
    "TimesFMAdapter",
    "get_backbone_adapter",
    "LatentSpaceARM",
    "OutputSpaceBarycenter",
    "TokenSpaceFormatter",
    "UnifiedRetrievalInjector",
    "CovariateQueryEmbedder",
    "ChannelBlockEncoder",
    "FrozenTextEncoderAdapter",
    "MultimodalTextAdapter",
    "ScalingLawFitter",
    "ManifoldRankEstimator",
    "fit_scaling_power_law",
    "estimate_two_nn_dimension",
    "estimate_pca_effective_rank"
]
