"""
QuantraCore Apex Accuracy Optimization System.

Comprehensive accuracy enhancement for the trading intelligence engine.
Implements all major accuracy optimization strategies:

1. Protocol Telemetry - Track which protocols contribute to winning trades
2. Feature Store - Centralized feature management with data quality audits
3. Calibration Layer - Probabilistic calibration for confidence accuracy
4. Regime-Gated Ensemble - Different models for different market conditions
5. Uncertainty Head - Conformal prediction for valid confidence bounds
6. Auto-Retraining - Drift detection and automatic model retraining
7. Multi-Horizon Prediction - 1-day to 10-day forecast optimization
8. Cross-Asset Features - VIX, sector rotation, market breadth

Version: 1.0.0
"""

from .protocol_telemetry import (
    ProtocolTelemetry,
    ProtocolMetrics,
    TelemetrySnapshot,
    get_protocol_telemetry,
)

from .feature_store import (
    FeatureStore,
    FeatureRegistry,
    FeatureDefinition,
    FeatureSnapshot,
    FeatureQualityReport,
    get_feature_store,
)

from .calibration import (
    CalibrationLayer,
    CalibrationMetrics,
    CalibratedPrediction,
    PlattScaler,
    IsotonicCalibrator,
    TemperatureScaler,
    get_calibration_layer,
)

from .regime_ensemble import (
    RegimeGatedEnsemble,
    RegimeDetector,
    RegimeSpecialist,
    MarketRegime,
    RegimeDetection,
    EnsemblePrediction,
    get_regime_ensemble,
)

from .uncertainty import (
    UncertaintyHead,
    UncertaintyEstimate,
    ConformalPredictor,
    ConformalPredictionSet,
    get_uncertainty_head,
)

from .auto_retrain import (
    AutoRetrainer,
    DriftDetector,
    SampleWeighter,
    DriftMetrics,
    RetrainingDecision,
    SampleWeight,
    get_auto_retrainer,
)

from .multi_horizon import (
    MultiHorizonPredictor,
    HorizonHead,
    PredictionHorizon,
    HorizonPrediction,
    MultiHorizonPrediction,
    get_multi_horizon_predictor,
)

from .cross_asset import (
    CrossAssetAnalyzer,
    CrossAssetFeatures,
    VIXAnalyzer,
    VIXAnalysis,
    VIXRegime,
    SectorAnalyzer,
    SectorAnalysis,
    SectorName,
    BreadthAnalyzer,
    MarketBreadthAnalysis,
    MarketRegimeType,
    get_cross_asset_analyzer,
)


__version__ = "1.0.0"

__all__ = [
    "ProtocolTelemetry",
    "ProtocolMetrics",
    "TelemetrySnapshot",
    "get_protocol_telemetry",
    "FeatureStore",
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureSnapshot",
    "FeatureQualityReport",
    "get_feature_store",
    "CalibrationLayer",
    "CalibrationMetrics",
    "CalibratedPrediction",
    "PlattScaler",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "get_calibration_layer",
    "RegimeGatedEnsemble",
    "RegimeDetector",
    "RegimeSpecialist",
    "MarketRegime",
    "RegimeDetection",
    "EnsemblePrediction",
    "get_regime_ensemble",
    "UncertaintyHead",
    "UncertaintyEstimate",
    "ConformalPredictor",
    "ConformalPredictionSet",
    "get_uncertainty_head",
    "AutoRetrainer",
    "DriftDetector",
    "SampleWeighter",
    "DriftMetrics",
    "RetrainingDecision",
    "SampleWeight",
    "get_auto_retrainer",
    "MultiHorizonPredictor",
    "HorizonHead",
    "PredictionHorizon",
    "HorizonPrediction",
    "MultiHorizonPrediction",
    "get_multi_horizon_predictor",
    "CrossAssetAnalyzer",
    "CrossAssetFeatures",
    "VIXAnalyzer",
    "VIXAnalysis",
    "VIXRegime",
    "SectorAnalyzer",
    "SectorAnalysis",
    "SectorName",
    "BreadthAnalyzer",
    "MarketBreadthAnalysis",
    "MarketRegimeType",
    "get_cross_asset_analyzer",
]
