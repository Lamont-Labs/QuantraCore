"""
Generate the core INVESTOR-REQUIRED image assets for QuantraCore Apex.

This script is designed for Replit / CI use.

Dependencies (add to requirements.txt):
  graphviz
  matplotlib

Also ensure system-level Graphviz is installed in the build environment.
On many systems:
  - Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y graphviz
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from graphviz import Digraph


# ================================================================================================
# CONFIG
# ================================================================================================

OUTPUT_DIR = Path("docs/assets/investor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_HISTORY_FILE = METRICS_DIR / "apexcore_v2_training_history.json"
EVAL_SUMMARY_FILE = METRICS_DIR / "apexlab_v2_eval_summary.json"


# ================================================================================================
# GRAPHVIZ HELPERS
# ================================================================================================

def _new_digraph(name: str, comment: str = "") -> Digraph:
    """
    Create a left-to-right Digraph with sane defaults for architecture diagrams.
    """
    dot = Digraph(name=name, comment=comment, format="png")
    dot.attr(rankdir="LR", splines="ortho", concentrate="true")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#0B1220",
             fontcolor="#E5E7EB", color="#4B5563", penwidth="1.2")
    dot.attr("edge", color="#6B7280", penwidth="1.0", arrowsize="0.8")
    return dot


def _render(dot: Digraph, filename_stem: str) -> None:
    """
    Render both PNG and SVG versions to OUTPUT_DIR with the given stem.
    """
    base_path = OUTPUT_DIR / filename_stem
    dot.render(str(base_path), format="png", cleanup=True)
    dot.render(str(base_path), format="svg", cleanup=True)


# ================================================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM (TOP-LEVEL ECOSYSTEM)
# ================================================================================================

def generate_system_architecture_diagram() -> None:
    """
    Investor-required: full ecosystem map for QuantraCore Apex v9.x.

    Shows how all major components relate:
      - Market Data / ApexFeed
      - Scanner
      - Deterministic Engine (Apex)
      - ApexLab v2
      - ApexCore v2 (Big/Mini)
      - PredictiveAdvisor
      - ApexDesk UI
      - QuantraVision Apex
      - Broker Layer (paper/live, optional)
    """
    dot = _new_digraph("QuantraCoreApex_Ecosystem", "QuantraCore Apex — Ecosystem Overview")

    # External
    dot.node("market_data", "Market Data APIs\n(Polygon, etc.)", shape="component")
    dot.node("user_research", "Research / Analyst\n(Desktop)", shape="ellipse")
    dot.node("retail_user", "Retail User\n(Android)", shape="ellipse")

    # Core backend
    dot.node("apexfeed", "ApexFeed\n(Data Ingest Layer)")
    dot.node("scanner", "Scanner\n(Full Market / Modes)")
    dot.node("engine", "Deterministic Engine\n(QuantraCore Apex)")
    dot.node("monster", "MonsterRunner\n& Runner Logic")

    # Learning / models
    dot.node("apexlab", "ApexLab v2\n(Labeling + Datasets)")
    dot.node("apexcore_big", "ApexCore v2 Big\n(Desktop / Server Model)")
    dot.node("apexcore_mini", "ApexCore v2 Mini\n(Mobile Model)")

    # Predictive integration
    dot.node("pred_advisor", "PredictiveAdvisor\n(Ranker / Triager)")

    # UI
    dot.node("apexdesk", "ApexDesk UI\n(React/Vite Dashboard)")
    dot.node("qvision", "QuantraVision Apex\n(Android Structural Copilot)")

    # Broker
    dot.node("broker", "Broker Layer\n(Paper / Live, Optional)")

    # Edges: data flow
    dot.edge("market_data", "apexfeed", label="raw OHLCV")
    dot.edge("apexfeed", "scanner", label="normalized bars")
    dot.edge("scanner", "engine", label="signals")
    dot.edge("engine", "monster", label="structural context")
    dot.edge("engine", "apexlab", label="teacher labels")
    dot.edge("apexlab", "apexcore_big", label="train\nApexCore v2 Big")
    dot.edge("apexcore_big", "apexcore_mini", label="distill\nMini")

    # Predictive advisory
    dot.edge("engine", "pred_advisor", label="top-N candidates")
    dot.edge("apexcore_big", "pred_advisor", label="ranker features")

    # UI flows
    dot.edge("engine", "apexdesk", label="scores + traces")
    dot.edge("monster", "apexdesk", label="runner metrics")
    dot.edge("pred_advisor", "apexdesk", label="advisory overlays")
    dot.edge("apexdesk", "user_research", label="research view")

    # Vision
    dot.edge("engine", "qvision", label="structural logic\n(ApexLite / Mini)")
    dot.edge("apexcore_mini", "qvision", label="on-device structural hints")
    dot.edge("qvision", "retail_user", label="overlays + HUD")

    # Broker (optional)
    dot.edge("engine", "broker", label="paper/live envelope")
    dot.edge("pred_advisor", "broker", label="ranked candidates", style="dashed")
    dot.edge("user_research", "broker", label="manual oversight", style="dotted")

    _render(dot, "01_quantracore_apex_ecosystem")


# ================================================================================================
# 2. ENGINE / ML PIPELINE FLOW (DETERMINISTIC → LAB → MODELS → PREDICTIVE)
# ================================================================================================

def generate_pipeline_flow_diagram() -> None:
    """
    Investor-required: shows the core deterministic → ML → advisor pipeline.

    Flow:
      Data → Scanner → Engine (Tier/LP/Omega) → Labels → ApexLab v2 → ApexCore v2 →
      PredictiveAdvisor → Outputs to UI/Broker (ranker-only).
    """
    dot = _new_digraph("Apex_Pipeline", "Deterministic → ML → Advisor Pipeline")

    dot.node("data", "Market Data\n(ApexFeed)")
    dot.node("scanner", "Scanner")
    dot.node("engine", "Deterministic Engine\n(Tier T01–T80,\nLP01–LP25, Ω)")
    dot.node("labels", "Teacher Labels\n& Structural Features")
    dot.node("apexlab", "ApexLab v2\n(Label Factory)")
    dot.node("datasets", "Datasets\n(Parquet/Arrow)")
    dot.node("train", "Training\nApexCore v2")
    dot.node("apexcore", "ApexCore v2\n(Big/Mini/Ensembles)")
    dot.node("advisor", "PredictiveAdvisor\n(Ranker / Triager)")
    dot.node("outputs", "Ranked Candidates\n+ Warnings")

    dot.edge("data", "scanner")
    dot.edge("scanner", "engine")
    dot.edge("engine", "labels")
    dot.edge("labels", "apexlab")
    dot.edge("apexlab", "datasets")
    dot.edge("datasets", "train")
    dot.edge("train", "apexcore")
    dot.edge("engine", "advisor", label="top-N deterministic\ncandidates")
    dot.edge("apexcore", "advisor", label="model signals")
    dot.edge("advisor", "outputs")

    _render(dot, "02_apex_pipeline_deterministic_to_ml")


# ================================================================================================
# 3. APEXLAB V2 LABELING FLOW (DATA → EVENTS → LABELS → DATASETS)
# ================================================================================================

def generate_apexlab_flow_diagram() -> None:
    """
    Investor-required: clear picture of ApexLab v2 offline labeling process.
    """
    dot = _new_digraph("ApexLabV2_Flow", "ApexLab v2 — Labeling & Dataset Builder")

    dot.node("raw_data", "Historical OHLCV\n(from ApexFeed)")
    dot.node("windows", "Window Builder\n(100–300 bars)")
    dot.node("engine", "Apex Engine\n(Teacher)")
    dot.node("labels", "Structural Labels\n(QuantraScore, risk, regime, etc.)")
    dot.node("future", "Future Outcomes\n(returns, drawdown, runup)")
    dot.node("quality", "Quality Tiers\n(A+/A/B/C/D)")
    dot.node("dataset", "Dataset Builder\n(ApexLab v2)")
    dot.node("parquet", "Parquet/Arrow\nDatasets")

    dot.edge("raw_data", "windows")
    dot.edge("windows", "engine")
    dot.edge("engine", "labels")
    dot.edge("windows", "future", label="future window\n(t+N bars)")
    dot.edge("labels", "quality", label="combined\nwith outcomes")
    dot.edge("future", "quality")
    dot.edge("labels", "dataset")
    dot.edge("quality", "dataset")
    dot.edge("dataset", "parquet")

    _render(dot, "03_apexlab_v2_labeling_flow")


# ================================================================================================
# 4. APEXCORE V2 MODEL FAMILY (BIG / MINI / ENSEMBLES)
# ================================================================================================

def generate_apexcore_model_family_diagram() -> None:
    """
    Investor-required: how ApexCore v2 Big & Mini relate to teacher and ensembles.
    """
    dot = _new_digraph("ApexCoreV2_Family", "ApexCore v2 Model Family")

    dot.node("teacher", "Apex Engine\n(Teacher)")
    dot.node("lab", "ApexLab v2\n(Datasets)")
    dot.node("big", "ApexCore v2 Big\n(Desktop / Server)")
    dot.node("mini", "ApexCore v2 Mini\n(Android / Mobile)")
    dot.node("ensemble", "ApexCore v2 Ensemble\n(N models + disagreement)")
    dot.node("manifest", "Model Manifest\n(SHA256, metrics, thresholds)")

    dot.edge("teacher", "lab", label="structural labels")
    dot.edge("lab", "big", label="train\nBig")
    dot.edge("big", "mini", label="distill\nMini")
    dot.edge("big", "ensemble", label="multi-copy\nor variants")
    dot.edge("mini", "ensemble", style="dotted", label="optional")
    dot.edge("ensemble", "manifest", label="metrics, hashes")

    _render(dot, "04_apexcore_v2_model_family")


# ================================================================================================
# 5. PREDICTIVEADVISOR + SAFETY GATING (RANKER-ONLY)
# ================================================================================================

def generate_predictive_safety_diagram() -> None:
    """
    Investor-required: demonstrate predictive layer is RANKER ONLY and fully safety-gated.
    """
    dot = _new_digraph("PredictiveAdvisor_Safety", "PredictiveAdvisor Safety & Gating")

    dot.node("engine", "Apex Engine\n(deterministic authority)")
    dot.node("candidates", "Top-N Candidates\n(from deterministic engine)")
    dot.node("apexcore", "ApexCore v2\n(Ensemble)")
    dot.node("advisor", "PredictiveAdvisor\n(Ranker Only)")
    dot.node("safety", "Safety Gates\n(disagreement, avoid_trade, thresholds)")
    dot.node("final", "Final View\n(Ranked, annotated,\nresearch-only)")

    dot.edge("engine", "candidates")
    dot.edge("candidates", "advisor")
    dot.edge("advisor", "apexcore", label="feature calls")
    dot.edge("apexcore", "advisor", label="model outputs")
    dot.edge("advisor", "safety", label="proposed ranks")
    dot.edge("safety", "final")

    # Safety invariants
    dot.node(
        "invariants",
        "Invariants:\n- Engine & Omega always override\n- No direct 'buy/sell'\n- Ranker-only hints",
        shape="note",
        fillcolor="#111827",
    )
    dot.edge("safety", "invariants", style="dotted")

    _render(dot, "05_predictiveadvisor_safety_gating")


# ================================================================================================
# 6. BROKER SAFETY ENVELOPE (OPTIONAL LAYER, FAIL-CLOSED)
# ================================================================================================

def generate_broker_safety_diagram() -> None:
    """
    Investor-required: shows that broker/execution is OPTIONAL, fail-closed, risk-gated.
    """
    dot = _new_digraph("Broker_Safety", "Broker Layer Safety Envelope")

    dot.node("research", "Research Mode\n(default)")
    dot.node("engine", "Apex Engine\n(Research / Paper Signals)")
    dot.node("risk", "Risk Module\n(max loss, exposure,\nΩ2 kill-switch)")
    dot.node("oms", "OMS\n(Paper/Live envelope)")
    dot.node("broker", "Broker APIs\n(IB, Alpaca, etc.)")
    dot.node("paper", "Paper Trading\n(default)")
    dot.node("live", "Live Ready\n(explicit, locked)")

    dot.edge("research", "engine")
    dot.edge("engine", "risk", label="simulated\norders")
    dot.edge("risk", "oms", label="approved\norders only")
    dot.edge("oms", "paper", label="default route")
    dot.edge("oms", "broker", label="when configured")
    dot.edge("broker", "live", label="only with\nexplicit setup")

    # Safety note
    dot.node(
        "safety",
        "Defaults:\n- broker_mode = disabled\n- execution_enabled = false\n- paper-only shipped",
        shape="note",
        fillcolor="#111827",
    )
    dot.edge("risk", "safety", style="dotted")

    _render(dot, "06_broker_safety_envelope")


# ================================================================================================
# 7. DETERMINISM & PROVENANCE CHAIN (INVESTOR-CRITICAL)
# ================================================================================================

def generate_determinism_chain_diagram() -> None:
    """
    Investor-required: clear visual of determinism + provenance guarantees.
    """
    dot = _new_digraph("Determinism_Chain", "Determinism & Provenance Chain")

    dot.node("code", "Code + Config\n(Versioned)")
    dot.node("hashes", "PROVENANCE.manifest\n(SHA256 for files)")
    dot.node("engine", "Apex Engine\n(Deterministic)")
    dot.node("inputs", "Inputs\n(data, universe,\nengine config)")
    dot.node("outputs", "Outputs\n(QuantraScore,\nlabels, logs)")
    dot.node("proof", "Proof Logs\n(JSONL, traces)")
    dot.node("replay", "Replay Engine\n(Deterministic Re-run)")
    dot.node("auditor", "Auditor / Investor\n(Reads proof)")

    dot.edge("code", "hashes")
    dot.edge("hashes", "engine", label="integrity check")
    dot.edge("inputs", "engine")
    dot.edge("engine", "outputs")
    dot.edge("outputs", "proof")
    dot.edge("proof", "replay")
    dot.edge("replay", "auditor")

    dot.node(
        "guarantees",
        "Guarantees:\n- Same code + inputs → same outputs\n- Hash-verified builds\n- Full trace per decision",
        shape="note",
        fillcolor="#111827",
    )
    dot.edge("proof", "guarantees", style="dotted")

    _render(dot, "07_determinism_and_provenance_chain")


# ================================================================================================
# 8. TRAINING & EVALUATION VISUALS (LOSS, CALIBRATION, QUALITY)
# ================================================================================================

def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_training_loss_curves() -> None:
    """
    Investor-required: at least one visual of training curves (loss over epochs).

    Expected JSON format in metrics/apexcore_v2_training_history.json:
    {
      "epochs": [1, 2, 3, ...],
      "loss_total": [...],
      "loss_quantra_score": [...],
      "loss_runner_prob": [...],
      "loss_avoid_trade": [...]
    }
    """
    history = _load_json(TRAINING_HISTORY_FILE)
    if history is None:
        # Fails gracefully if metrics file not present; investor can add later.
        return

    epochs = history.get("epochs") or []
    if not epochs:
        return

    plt.figure()
    if "loss_total" in history:
        plt.plot(epochs, history["loss_total"], label="Total loss")
    if "loss_quantra_score" in history:
        plt.plot(epochs, history["loss_quantra_score"], label="QuantraScore head")
    if "loss_runner_prob" in history:
        plt.plot(epochs, history["loss_runner_prob"], label="Runner head")
    if "loss_avoid_trade" in history:
        plt.plot(epochs, history["loss_avoid_trade"], label="Avoid-trade head")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ApexCore v2 — Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = OUTPUT_DIR / "08_apexcore_v2_training_loss.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_evaluation_bar_summary() -> None:
    """
    Investor-required: one compact evaluation summary bar chart.

    Expected JSON at metrics/apexlab_v2_eval_summary.json:
    {
      "runner_auc": float,
      "runner_brier": float,
      "quality_accuracy": float,
      "regime_accuracy": float
    }
    """
    summary = _load_json(EVAL_SUMMARY_FILE)
    if summary is None:
        return

    metrics = []
    values = []

    if "runner_auc" in summary:
        metrics.append("Runner AUC")
        values.append(summary["runner_auc"])
    if "quality_accuracy" in summary:
        metrics.append("Quality Accuracy")
        values.append(summary["quality_accuracy"])
    if "regime_accuracy" in summary:
        metrics.append("Regime Accuracy")
        values.append(summary["regime_accuracy"])

    if not metrics:
        return

    plt.figure()
    x = range(len(metrics))
    plt.bar(x, values)
    plt.xticks(x, metrics, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("ApexCore v2 / ApexLab v2 — Evaluation Summary")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.grid(axis="y", alpha=0.2)

    out_path = OUTPUT_DIR / "09_apexlab_v2_eval_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ================================================================================================
# MAIN ENTRYPOINT
# ================================================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = [
        generate_system_architecture_diagram,
        generate_pipeline_flow_diagram,
        generate_apexlab_flow_diagram,
        generate_apexcore_model_family_diagram,
        generate_predictive_safety_diagram,
        generate_broker_safety_diagram,
        generate_determinism_chain_diagram,
        generate_training_loss_curves,
        generate_evaluation_bar_summary,
    ]

    for gen in generators:
        try:
            print(f"[+] Generating: {gen.__name__}")
            gen()
        except Exception as e:
            # Fail soft per asset — investors still get the rest.
            print(f"[!] Failed: {gen.__name__}: {e}")


if __name__ == "__main__":
    main()
