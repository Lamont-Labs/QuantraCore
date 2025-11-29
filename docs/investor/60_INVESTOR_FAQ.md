# Investor FAQ

**Document Classification:** Investor Due Diligence — FAQ  
**Version:** 9.0-A  
**Date:** November 2025  

---

## Direct Q&A, No Spin

This document answers investor questions in the way they would actually ask them.

---

## Product and Technology

### Q: Is this a live trading system today, or a research engine?

**A:** Research engine only.

The system is designed for market structure analysis and research. Execution capabilities are disabled by default, and the broker layer is spec-only. Ω4 (Compliance Mode) is permanently active, enforcing research-only operation.

*See: [Platform Overview](03_PLATFORM_OVERVIEW.md), [Broker and Execution](14_BROKER_AND_EXECUTION_ENVELOPE.md)*

---

### Q: What's actually implemented vs what's just spec?

**A:** Most core functionality is implemented; execution layer is spec-only.

| Component | Status |
|-----------|--------|
| Deterministic Engine | **Implemented** |
| ApexLab v2 | **Implemented** |
| ApexCore v2 | **Implemented** |
| PredictiveAdvisor | **Implemented** |
| ApexDesk UI | **Implemented** |
| Scanner | **Implemented** |
| FastAPI Backend | **Implemented** |
| Broker Layer | Spec-only |
| Live Execution | Disabled |

*See: [Platform Overview](03_PLATFORM_OVERVIEW.md)*

---

### Q: How much refactor would a senior team need to do to harden this?

**A:** Estimate 6-12 months for production-grade deployment.

Work needed:
- Production infrastructure setup (2-3 months)
- Monitoring and observability (1-2 months)
- Security hardening (1-2 months)
- Load testing and optimization (1-2 months)
- Documentation updates (1 month)

The codebase is clean and well-documented, which reduces onboarding time. Core logic is solid; it's the operational infrastructure that needs work.

*See: [Limitations and Honest Risks](33_LIMITATIONS_AND_HONEST_RISKS.md), [Operations and Runbooks](52_OPERATIONS_AND_RUNBOOKS.md)*

---

### Q: How heavy is this to run (K6 vs server)?

**A:** Runs on desktop; scales on server.

| Target | Specs | Use Case |
|--------|-------|----------|
| K6 Desktop | 4+ cores, 16GB RAM, 50GB SSD | Individual research |
| Institutional Server | 16+ cores, 64GB RAM, 500GB SSD | Full-universe scanning |

CPU-only — no GPU required for inference.

*See: [System Architecture](10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md), [Operations](52_OPERATIONS_AND_RUNBOOKS.md)*

---

### Q: Can we disable the predictive layer and just use deterministic outputs?

**A:** Yes, absolutely.

The system is architected so the predictive layer is optional. You can:
- Use engine-only mode (deterministic outputs only)
- Disable model loading entirely
- Configure advisor to pass-through engine rankings

The engine is the source of truth; the model is an optional assistant.

*See: [Predictive Layer and Safety](13_PREDICTIVE_LAYER_AND_SAFETY.md)*

---

## Models and Data

### Q: How much data did you use for training?

**A:** ~3,000 liquid US equities over 6 years.

| Aspect | Value |
|--------|-------|
| Symbols | ~3,000 liquid US equities |
| Period | 2018-2024 |
| Windows | ~2 million labeled windows |
| Quality Filter | Avg volume >200k, price >$5 |

*See: [Data Sources and Universe](20_DATA_SOURCES_AND_UNIVERSE.md)*

---

### Q: How often would you retrain in production?

**A:** Quarterly evaluation, retrain as needed.

Recommended cadence:
- Weekly: Monitor drift metrics
- Monthly: Evaluate performance vs baseline
- Quarterly: Consider retraining if drift detected
- Annually: Full model refresh regardless

*See: [Training Process](22_TRAINING_PROCESS_AND_HYPERPARAMS.md), [Evaluation and Limitations](23_EVALUATION_AND_LIMITATIONS.md)*

---

### Q: What regimes is it weakest in?

**A:** High volatility and extreme events.

| Regime | Performance |
|--------|-------------|
| Trending | Best (AUC 0.81) |
| Low volatility | Good (AUC 0.76) |
| Ranging | Moderate (AUC 0.73) |
| High volatility | Weakest (AUC 0.68) |

The model also struggles with:
- Black swan events
- Low-liquidity microcaps
- Unusual corporate actions

*See: [Evaluation and Limitations](23_EVALUATION_AND_LIMITATIONS.md)*

---

### Q: How do you protect against overfitting and leakage?

**A:** Strict walk-forward splits with no overlap.

Protections:
1. **Temporal splits:** Train < validation < test (no mixing)
2. **No future features:** Features use only window data
3. **Gap between window and outcome:** Prevents immediate-future leakage
4. **No overlapping windows:** Train/val sets don't share dates
5. **Regularization:** L2, dropout, early stopping
6. **Ensemble disagreement:** High variance triggers caution

*See: [Labeling Methods and Leakage Guards](21_LABELING_METHODS_AND_LEAKAGE_GUARDS.md)*

---

## Risk and Compliance

### Q: Could this be used in a regulated setting? What would be missing?

**A:** Research use is fine; live trading needs significant additional work.

For live trading in a regulated setting, you'd need:
- Production-grade infrastructure
- Real-time monitoring
- Regulatory registration (SEC, FINRA, etc.)
- Compliance program
- Legal counsel approval
- Risk controls beyond current implementation

The system is designed with regulatory awareness (99.25% compliance score), but that's for research use. Execution requires additional layers.

*See: [Compliance and Usage Policies](31_COMPLIANCE_AND_USAGE_POLICIES.md), [Limitations](33_LIMITATIONS_AND_HONEST_RISKS.md)*

---

### Q: How do you prevent the model from being a "signal seller" to retail?

**A:** By design — no buy/sell signals generated.

Safeguards:
1. **Educational framing:** All outputs are structural probabilities, not recommendations
2. **Ω4 always active:** Research-only mode enforced
3. **No execution capability:** Broker layer disabled
4. **Disclaimers:** Every output includes compliance notes
5. **QuantraVision:** Explicitly educational, not advisory

*See: [QuantraVision Overview](16_QUANTRAVISION_APEX_OVERVIEW.md), [Compliance](31_COMPLIANCE_AND_USAGE_POLICIES.md)*

---

## Business and Team

### Q: What exactly do you want from a partner/investor?

**A:** Capital + engineering support to productize.

Open to:
- **IP acquisition:** Sell the system outright
- **Licensing:** Recurring revenue from institutional use
- **Build-out partnership:** Minority investment + senior engineers

The ideal outcome is a partnership where the founder maintains architectural vision while experienced engineers harden and extend the system.

*See: [Deal Summary](05_DEAL_SUMMARY.md), [Roadmap and Capital Use](42_ROADMAP_AND_CAPITAL_USE.md)*

---

### Q: What kind of team would you want around this (first 3-5 hires)?

**A:** Senior engineers with financial domain experience.

Priority hires:
1. **Senior ML/Python Engineer:** Model improvements, production ML
2. **Full-Stack Engineer:** UI/API, developer experience
3. **DevOps/Infrastructure:** Production deployment, monitoring
4. (Optional) **Quant Researcher:** Domain expertise, strategy development
5. (Optional) **Compliance/Ops:** Regulatory navigation

*See: [Roadmap and Capital Use](42_ROADMAP_AND_CAPITAL_USE.md)*

---

### Q: What are your own gaps and limits as founder?

**A:** Honest assessment — I'm an architect, not a scaling engineer.

**Strengths:**
- System architecture and design
- Product vision and direction
- Domain understanding (trading research)
- Documentation and specification
- AI-augmented development

**Gaps:**
- Production infrastructure experience
- Scaling to institutional deployment
- Fundraising and enterprise sales
- Team management at scale

I want to architect and steer, not be the solo coder forever.

*See: [Founder Profile](02_FOUNDER_PROFILE.md)*

---

## Additional Questions

### Q: What's the IP defensibility?

**A:** Novel architecture + comprehensive implementation.

| Moat | Description |
|------|-------------|
| Protocol IP | 145 protocols with unique logic |
| Architecture | Deterministic-to-neural pipeline |
| Documentation | 180-page master spec + investor docs |
| Test Suite | 1145 tests validating correctness |
| Provenance | SBOM, manifests, proof logs |

*See: [Competitive Positioning](04_COMPETITIVE_POSITIONING.md)*

---

### Q: What happens if you're unavailable?

**A:** Comprehensive documentation enables continuity.

The system is extensively documented:
- Master specification (180 pages)
- Investor documentation (30+ files)
- Inline code comments
- Test suite as documentation
- Self-documenting architecture

A new team could onboard within weeks, not months.

*See: [Engineering Overview](50_ENGINEERING_OVERVIEW_AND_PRACTICES.md)*

---

### Q: Why should we believe the metrics?

**A:** They're verifiable.

All metrics are:
- Generated by automated tests
- Reproducible on your machine
- Exposed via API (`/compliance/score`, `/models/metrics`)
- Backed by proof logs

You can run the test suite yourself and verify everything.

*See: [Testing and Coverage](51_TESTING_AND_COVERAGE_SUMMARY.md)*

---

## Still Have Questions?

If your question isn't answered here:

1. Check the [Document Index](README_INVESTOR_PORTAL.md) for relevant deep dives
2. Search the codebase for implementation details
3. Run the API endpoints to see live data
4. Ask directly — honest answers are the policy

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
