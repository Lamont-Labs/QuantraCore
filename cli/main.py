import json
from pathlib import Path
import typer
from src.core.engine import generate_signal
from src.core.risk_filters import run_filters
from src.core.failsafes import enforce_bounds

app = typer.Typer(help="QuantraCore™ deterministic CLI")

@app.command("run")
def run(ticker: str = "AAPL", seed: int = 42):
    sig = generate_signal(ticker, seed)
    payload = {"signal": sig, "bounds": enforce_bounds(sig["score"]), "filters": run_filters(sig["score"])}
    outdir = Path("dist/golden_demo_outputs"); outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{ticker}_demo.json"
    path.write_text(json.dumps(payload, indent=2))
    typer.echo(f"Wrote {path}")

if __name__ == "__main__":
    app()
