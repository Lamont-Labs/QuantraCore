import hashlib, csv
from pathlib import Path

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

out = Path("SBOM/checksums.csv")
out.parent.mkdir(parents=True, exist_ok=True)
target = Path("dist/golden_demo_outputs/AAPL_demo.json")
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["file","sha256"])
    if target.exists():
        w.writerow([str(target), sha256(target)])
print(f"Wrote {out}")
