# Architecture
- Engine (QuantraScore), Risk (ASP-01..20), HUD JSON, API (FastAPI), CLI (Typer).
- Flow: seed → score → bounds → filters → HUD → artifact → checksum.
- Determinism: seeded RNG, no network, stable JSON.
