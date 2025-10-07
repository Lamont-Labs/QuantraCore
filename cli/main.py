import json
from pathlib import Path
import typer
from src.core.engine import generate_signal
from src.core.risk_filters import run_filters
from src.core.failsafes import enforce_bounds

app = typer.Typer(help="QuantraCore™ deterministic CLI")

@app.callback(invoke_without_command=True)
def default(
    ticker: str = typer.Option("AAPL", help="Ticker symbol"),
    seed: int = typer.Option(42, help="Deterministic seed"),
):
    """
    Default command (no subcommand required).
    Produces a deterministic demo artifact at dist/golden_demo_outputs/<TICKER>_demo.json
    """
    sig = generate_signal(ticker, seed)
    payload = {
        "signal": sig,
        "bounds": enforce_bounds(sig["score"]),
        "filters": run_filters(sig["score"]),
    }
    outdir = Path("dist/golden_demo_outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{ticker}_demo.json"
    path.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[QuantraCore] Wrote {path}")

if __name__ == "__main__":
    app()
