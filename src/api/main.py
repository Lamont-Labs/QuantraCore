from fastapi import FastAPI
from datetime import datetime
from pathlib import Path
import json
from src.core.engine import generate_signal
from src.core.risk_filters import run_filters
from src.core.failsafes import safe_execute

app = FastAPI(title="QuantraCore API", version="v3.7u",
              description="Deterministic demo — no live trading")

@app.get("/health")
def health():
    return {"status": "ok", "version": "v3.7u", "time": datetime.utcnow().isoformat()}

@app.get("/score")
def score(ticker: str = "AAPL", seed: int = 42):
    return safe_execute(generate_signal, ticker, seed)

@app.get("/risk/hud")
def risk_hud(ticker: str = "AAPL", seed: int = 42):
    sig = generate_signal(ticker, seed)
    return {"ticker": ticker, "score": sig["score"], "filters": run_filters(sig["score"])}

@app.get("/audit/export")
def audit_export():
    Path("dist/golden_demo_outputs").mkdir(parents=True, exist_ok=True)
    doc = {"status": "demo_export", "time": datetime.utcnow().isoformat()}
    Path("dist/golden_demo_outputs/audit_export.json").write_text(json.dumps(doc, indent=2))
    return doc
