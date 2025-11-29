"""ApexLab - Offline Training Environment for QuantraCore Apex."""

from .windows import WindowBuilder
from .features import FeatureExtractor
from .labels import LabelGenerator
from .dataset_builder import DatasetBuilder

from .apexlab_v2 import (
    ApexLabV2Row,
    ApexLabV2Builder,
    ApexLabV2DatasetBuilder,
    QualityTier,
    assign_quality_tier,
    compute_runner_flags,
    compute_regime_label,
    compute_future_returns,
    encode_protocol_vector,
)

from .training import (
    TrainingConfig,
    ApexCoreV2Trainer,
    create_walk_forward_splits,
    extract_features,
    extract_targets,
    run_training,
)

from .evaluation import (
    CalibrationResult,
    RankingResult,
    RegimeResult,
    EvaluationReport,
    ApexCoreV2Evaluator,
    evaluate_runner_calibration,
    evaluate_ranking_quality,
    run_evaluation,
)

__all__ = [
    "WindowBuilder",
    "FeatureExtractor",
    "LabelGenerator",
    "DatasetBuilder",
    "ApexLabV2Row",
    "ApexLabV2Builder",
    "ApexLabV2DatasetBuilder",
    "QualityTier",
    "assign_quality_tier",
    "compute_runner_flags",
    "compute_regime_label",
    "compute_future_returns",
    "encode_protocol_vector",
    "TrainingConfig",
    "ApexCoreV2Trainer",
    "create_walk_forward_splits",
    "extract_features",
    "extract_targets",
    "run_training",
    "CalibrationResult",
    "RankingResult",
    "RegimeResult",
    "EvaluationReport",
    "ApexCoreV2Evaluator",
    "evaluate_runner_calibration",
    "evaluate_ranking_quality",
    "run_evaluation",
]
