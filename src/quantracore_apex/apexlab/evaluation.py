"""
ApexCore V2 Evaluation Harness.

Provides comprehensive evaluation of trained models:
- Calibration curves and ECE
- Ranking quality analysis
- Regime-segmented performance
- Runner detection accuracy
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.quantracore_apex.apexcore.apexcore_v2 import ApexCoreV2Model, ApexCoreV2Ensemble
from src.quantracore_apex.apexcore.manifest import ApexCoreV2Manifest


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    bin_edges: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    expected_calibration_error: float
    max_calibration_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingResult:
    """Results from ranking quality analysis."""
    decile_returns: List[float]
    decile_runner_rates: List[float]
    top_decile_lift: float
    auc_runner: float
    precision_at_k: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeResult:
    """Results from regime-segmented analysis."""
    regime: str
    n_samples: int
    runner_auc: float
    runner_precision: float
    quality_accuracy: float
    avg_quantra_score_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    model_variant: str
    n_samples: int
    calibration: CalibrationResult
    ranking: RankingResult
    regime_results: List[RegimeResult]
    overall_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_variant": self.model_variant,
            "n_samples": self.n_samples,
            "calibration": self.calibration.to_dict(),
            "ranking": self.ranking.to_dict(),
            "regime_results": [r.to_dict() for r in self.regime_results],
            "overall_metrics": self.overall_metrics,
        }
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_runner_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Evaluate runner probability calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        CalibrationResult with ECE and bin-wise metrics
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        bin_counts.append(int(count))
        
        if count > 0:
            accuracy = float(np.mean(y_true[mask]))
            confidence = float(np.mean(y_prob[mask]))
        else:
            accuracy = 0.0
            confidence = (bin_edges[i] + bin_edges[i + 1]) / 2
        
        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)
    
    ece = 0.0
    mce = 0.0
    n_total = len(y_true)
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_error = abs(bin_accuracies[i] - bin_confidences[i])
            ece += bin_error * bin_counts[i] / n_total
            mce = max(mce, bin_error)
    
    return CalibrationResult(
        bin_edges=list(bin_edges),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
    )


def evaluate_ranking_quality(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    n_deciles: int = 10,
) -> RankingResult:
    """
    Evaluate ranking quality by predicted probability.
    
    Args:
        y_true: True binary runner labels
        y_prob: Predicted runner probabilities
        returns: Actual returns for each sample
        n_deciles: Number of deciles for analysis
        
    Returns:
        RankingResult with decile-wise metrics
    """
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    returns_sorted = returns[sorted_indices]
    
    n_samples = len(y_true)
    decile_size = n_samples // n_deciles
    
    decile_returns = []
    decile_runner_rates = []
    
    for i in range(n_deciles):
        start = i * decile_size
        end = (i + 1) * decile_size if i < n_deciles - 1 else n_samples
        
        decile_returns.append(float(np.mean(returns_sorted[start:end])))
        decile_runner_rates.append(float(np.mean(y_true_sorted[start:end])))
    
    top_decile_lift = decile_runner_rates[0] / max(np.mean(y_true), 1e-10)
    
    auc = 0.5
    if len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_prob))
        except (ImportError, ValueError):
            pass
    
    precision_at_k = {}
    for k in [10, 50, 100]:
        if k <= n_samples:
            precision_at_k[k] = float(np.mean(y_true_sorted[:k]))
    
    return RankingResult(
        decile_returns=decile_returns,
        decile_runner_rates=decile_runner_rates,
        top_decile_lift=float(top_decile_lift),
        auc_runner=auc,
        precision_at_k=precision_at_k,
    )


def evaluate_regime_performance(
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
) -> List[RegimeResult]:
    """
    Evaluate model performance segmented by regime.
    
    Args:
        df: DataFrame with regime_label and targets
        outputs: Model outputs dictionary
        
    Returns:
        List of RegimeResult for each regime
    """
    results = []
    
    if "regime_label" not in df.columns:
        return results
    
    regimes = df["regime_label"].unique()
    
    for regime in regimes:
        mask = df["regime_label"] == regime
        n_samples = int(np.sum(mask))
        
        if n_samples < 10:
            continue
        
        runner_auc = 0.5
        runner_precision = 0.0
        quality_accuracy = 0.0
        quantra_error = 0.0
        
        if "hit_runner_threshold" in df.columns and "runner_prob" in outputs:
            y_true = df.loc[mask, "hit_runner_threshold"].values
            y_prob = outputs["runner_prob"][mask]
            
            if len(np.unique(y_true)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score
                    runner_auc = float(roc_auc_score(y_true, y_prob))
                except (ImportError, ValueError):
                    pass
            
            runner_precision = float(np.mean((y_prob > 0.5) == y_true))
        
        if "future_quality_tier" in df.columns and "quality_logits" in outputs:
            pred_tiers = np.argmax(outputs["quality_logits"][mask], axis=1)
            tier_mapping = {"A_PLUS": 0, "A": 1, "B": 2, "C": 3, "D": 4}
            true_tiers = np.array([tier_mapping.get(t, 3) for t in df.loc[mask, "future_quality_tier"]])
            quality_accuracy = float(np.mean(pred_tiers == true_tiers))
        
        if "quantra_score" in df.columns and "quantra_score" in outputs:
            pred_scores = outputs["quantra_score"][mask]
            true_scores = df.loc[mask, "quantra_score"].values
            quantra_error = float(np.mean(np.abs(pred_scores - true_scores)))
        
        results.append(RegimeResult(
            regime=str(regime),
            n_samples=n_samples,
            runner_auc=runner_auc,
            runner_precision=runner_precision,
            quality_accuracy=quality_accuracy,
            avg_quantra_score_error=quantra_error,
        ))
    
    return results


class ApexCoreV2Evaluator:
    """
    Complete evaluation harness for ApexCore V2 models.
    """
    
    def __init__(
        self,
        model: Optional[ApexCoreV2Model] = None,
        ensemble: Optional[ApexCoreV2Ensemble] = None,
        manifest: Optional[ApexCoreV2Manifest] = None,
    ):
        self.model = model
        self.ensemble = ensemble
        self.manifest = manifest
    
    def evaluate(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
    ) -> EvaluationReport:
        """
        Run complete evaluation on dataset.
        
        Args:
            df: DataFrame with targets and metadata
            features: Feature matrix for model input
            
        Returns:
            Complete EvaluationReport
        """
        if self.ensemble is not None:
            outputs = self.ensemble.forward(features)
        elif self.model is not None:
            outputs = self.model.forward(features)
        else:
            raise ValueError("No model or ensemble provided")
        
        calibration = CalibrationResult(
            bin_edges=[0.0, 1.0],
            bin_accuracies=[0.0],
            bin_confidences=[0.0],
            bin_counts=[0],
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
        )
        
        if "hit_runner_threshold" in df.columns and "runner_prob" in outputs:
            y_true = df["hit_runner_threshold"].values
            y_prob = outputs["runner_prob"]
            calibration = evaluate_runner_calibration(y_true, y_prob)
        
        ranking = RankingResult(
            decile_returns=[0.0] * 10,
            decile_runner_rates=[0.0] * 10,
            top_decile_lift=1.0,
            auc_runner=0.5,
            precision_at_k={},
        )
        
        if "hit_runner_threshold" in df.columns and "runner_prob" in outputs:
            y_true = df["hit_runner_threshold"].values
            y_prob = outputs["runner_prob"]
            returns = df["ret_5d"].values if "ret_5d" in df.columns else np.zeros(len(df))
            ranking = evaluate_ranking_quality(y_true, y_prob, returns)
        
        regime_results = evaluate_regime_performance(df, outputs)
        
        overall_metrics = {
            "runner_auc": ranking.auc_runner,
            "calibration_ece": calibration.expected_calibration_error,
            "top_decile_lift": ranking.top_decile_lift,
        }
        
        variant = self.manifest.variant if self.manifest else "unknown"
        
        return EvaluationReport(
            timestamp=datetime.utcnow().isoformat(),
            model_variant=variant,
            n_samples=len(df),
            calibration=calibration,
            ranking=ranking,
            regime_results=regime_results,
            overall_metrics=overall_metrics,
        )
    
    def generate_text_summary(self, report: EvaluationReport) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "ApexCore V2 Evaluation Report",
            "=" * 60,
            f"Timestamp: {report.timestamp}",
            f"Model Variant: {report.model_variant}",
            f"Samples Evaluated: {report.n_samples}",
            "",
            "--- Overall Metrics ---",
            f"Runner AUC: {report.overall_metrics.get('runner_auc', 0):.4f}",
            f"Calibration ECE: {report.overall_metrics.get('calibration_ece', 0):.4f}",
            f"Top Decile Lift: {report.overall_metrics.get('top_decile_lift', 1):.2f}x",
            "",
            "--- Calibration ---",
            f"Expected Calibration Error: {report.calibration.expected_calibration_error:.4f}",
            f"Max Calibration Error: {report.calibration.max_calibration_error:.4f}",
            "",
            "--- Ranking Quality ---",
            f"Top Decile Runner Rate: {report.ranking.decile_runner_rates[0]:.2%}",
            f"Bottom Decile Runner Rate: {report.ranking.decile_runner_rates[-1]:.2%}",
        ]
        
        if report.regime_results:
            lines.append("")
            lines.append("--- Regime Performance ---")
            for r in report.regime_results:
                lines.append(
                    f"  {r.regime}: AUC={r.runner_auc:.3f}, "
                    f"n={r.n_samples}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


def run_evaluation(
    dataset_path: str,
    model_dir: str,
    manifest_path: str,
    output_dir: str = "reports/apexlab_v2_eval",
) -> EvaluationReport:
    """
    Run evaluation from command line.
    
    Args:
        dataset_path: Path to Parquet dataset
        model_dir: Directory containing trained model
        manifest_path: Path to manifest JSON
        output_dir: Directory for output reports
        
    Returns:
        EvaluationReport
    """
    df = pd.read_parquet(dataset_path)
    
    manifest = ApexCoreV2Manifest.load(manifest_path)
    
    ensemble_path = Path(model_dir) / "ensemble"
    if ensemble_path.exists():
        ensemble = ApexCoreV2Ensemble.load(str(ensemble_path))
        evaluator = ApexCoreV2Evaluator(ensemble=ensemble, manifest=manifest)
    else:
        model = ApexCoreV2Model.load(str(Path(model_dir) / "model.joblib"))
        evaluator = ApexCoreV2Evaluator(model=model, manifest=manifest)
    
    from src.quantracore_apex.apexlab.training import extract_features
    features = extract_features(df)
    
    report = evaluator.evaluate(df, features)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir) / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report.save(str(report_dir / "evaluation_summary.json"))
    
    summary = evaluator.generate_text_summary(report)
    with open(report_dir / "evaluation_summary.txt", "w") as f:
        f.write(summary)
    
    return report


__all__ = [
    "CalibrationResult",
    "RankingResult",
    "RegimeResult",
    "EvaluationReport",
    "ApexCoreV2Evaluator",
    "evaluate_runner_calibration",
    "evaluate_ranking_quality",
    "evaluate_regime_performance",
    "run_evaluation",
]
