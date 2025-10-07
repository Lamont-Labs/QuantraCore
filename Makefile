.PHONY: demo verify sbom test clean

demo:
\tpython -m cli.main run --ticker AAPL --seed 42

verify:
\tbash verify.sh

sbom:
\tpython3 -c "import json;print(json.dumps({'sbom':'CycloneDX placeholder'},indent=2))" > SBOM/sbom.cdx.json

test:
\tpytest -q

clean:
\trm -rf dist/* SBOM/*.json SBOM/*.csv
