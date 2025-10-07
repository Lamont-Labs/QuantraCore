from pathlib import Path
import hashlib, json, sys

p = Path("dist/golden_demo_outputs/AAPL_demo.json")
if not p.exists():
    print("Missing demo output. Run: python -m cli.main run --ticker AAPL --seed 42")
    sys.exit(1)

h = hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({"file": str(p), "sha256": h}, indent=2))
