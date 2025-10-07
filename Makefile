.PHONY: demo verify sbom test clean

demo:
	@echo "Running QuantraCore deterministic demo (no subcommand)..."
	python -m cli.main

verify:
	bash verify.sh

sbom:
	python3 -c "import json; print(json.dumps({'sbom':'CycloneDX placeholder'}, indent=2))" > SBOM/sbom.cdx.json

test:
	pytest -q

clean:
	rm -rf dist/* SBOM/*.json SBOM/*.csv
