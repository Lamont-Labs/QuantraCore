# QuantraCore Apex v8.0 — Architecture Summary

## Core Components

- **Apex Engine** — Deterministic core producing QuantraScore and structural analysis
- **Tier Protocols (T01–T80)** — Structural, trait, entropy, suppression, and advanced protocols
- **Learning Protocols (LP01–LP25)** — Trait reinforcement, entropy tuning, pattern memory
- **Omega Directives (Ω1–Ω4)** — Integrity lock, risk kill switch, config guard, compliance gate
- **API Layer** — FastAPI endpoints for integration
- **CLI** — Typer-based command-line interface

## Data Flow

```
Market Data → Normalization → Validation → Trait Extraction →
Tier Protocols → Learning Protocols → Omega Safety →
QuantraScore → Proof Log → Output
```

## Design Principles

- **Determinism** — Seeded operations, no network calls in core logic, reproducible outputs
- **Fail-Closed** — System restricts on uncertainty rather than guessing
- **Proof Logging** — All operations hashed and logged
- **Offline-First** — Zero cloud dependency for core logic

## Model Architecture

- **ApexCore Full** — Desktop structural model (4–20MB)
- **ApexCore Mini** — Mobile structural model (0.5–3MB, distilled from Full)

See [Master Spec v8.0](../QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml) for complete specifications.
