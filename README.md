# QuantraCore™ — AI Trading Intelligence Engine

**Owner:** Jesse J. Lamont  
**Org:** Lamont-Labs  
**Version:** v3.7u  
[![QuantraCore CI](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml)
**Date:** 2025-10-07  
**Status:** Demo-ready, deterministic handoff seed  
**Repo:** https://github.com/Lamont-Labs/QuantraCore

---

## 📖 Purpose
QuantraCore™ is an institutional-grade deterministic AI trading demo.  
It proves that AI signal engines can be **transparent, reproducible, and compliance-safe** without relying on hype.  
Every output is **replayable, hashed, and logged**, so any reviewer can confirm provenance and determinism.  
This is a **truth-only demo** — no live trading, no user data, and no external brokerage.

---

## 🧱 Repository Contents
- /src/core/ — deterministic signal engine and risk filters  
- /src/api/ — FastAPI endpoints: /health, /score, /risk/hud, /audit/export  
- /cli/ — Typer CLI for demo scoring  
- /tests/ — reproducibility and filter tests  
- /docs/ — architecture, quickstart, investor, limitations, security  
- /assets/ — branding placeholders and screenshots  
- /SBOM/ — CycloneDX metadata, provenance JSON, and checksums  
- /dist/ — generated demo outputs  
- verify.sh — deterministic verification script  
- .github/workflows/ci.yml — reproducible GitHub Actions workflow

---

## 🚫 What This Repo Does *Not* Include
- No live brokerage or API keys  
- No real financial data or external model feeds  
- No user accounts or personal info  
- No claims of profit, advice, or market prediction  
All data is **synthetic and demonstrative**.

---

## 🚀 Getting Started

1. **Clone or download**
   \`\`\`bash
   git clone https://github.com/Lamont-Labs/QuantraCore.git
   cd QuantraCore
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Run deterministic demo**
   \`\`\`bash
   python -m cli.main
   \`\`\`

4. **Verify checksums**
   \`\`\`bash
   bash verify.sh
   \`\`\`

All reproducible demo outputs appear in `dist/golden_demo_outputs/`.

---

## 🧠 Architecture Overview
- Core Engine — Generates reproducible AI signals  
- Risk Layer — Fail-closed filters (ASP-01 → ASP-20)  
- Provenance Layer — Logs + hashes every step  
- HUD Layer — Interactive risk visualization  
- CI Layer — Enforces reproducibility on rebuilds  

See `/docs/ARCHITECTURE.md` for system diagrams.

---

## 🔒 Security & Provenance
- Dependencies pinned in `requirements.txt`  
- SBOM + checksums stored under `/SBOM/`  
- No secrets or environment variables included  
- Rebuilds hash-verifiable with `verify.sh`

---

## 🧩 Related Projects
Part of the **Lamont Labs** proof-of-possibility demo suite:  
- TreeMix™ — Mobile Remix Collaboration App  
- MemoryCloud™ — Privacy-First AI Memory Assistant  
- Brightline™ — Education AI Compliance Assistant  
- SpecForge™ — AI Invention Mapper  

---

## 📞 Contact
**Jesse J. Lamont** — Founder, Lamont Labs  
📧 lamontlabs@proton.me  
🌐 https://github.com/Lamont-Labs

---

## ⚖️ Disclaimers
Demo repository only — no trading advice or financial activity.  
All data is synthetic or public domain.  
No production systems are connected.

---

**Persistence = Proof.**  
Every build, every log, every checksum — reproducible by design.
