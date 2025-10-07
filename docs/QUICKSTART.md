# Quickstart
1) `pip install -r requirements.txt`
2) CLI run: `python -m cli.main run --ticker AAPL --seed 42`
3) Verify: `bash verify.sh` → SBOM/checksums.csv
4) Optional API: `uvicorn src.api.asgi:app --reload`
