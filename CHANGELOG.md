# Changelog

All notable changes to QuantraCore Apex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [9.0.0-A] - 2025-11-30

### Added

#### Security Hardening
- **API Authentication:** Added `X-API-Key` header verification for protected endpoints
- **CORS Restriction:** Changed from wildcard to regex pattern (localhost + Replit domains only)
- **Non-blocking Rate Limiting:** Updated Polygon and Binance adapters with async-compatible delays
- **Timeframe Validation:** Added case-insensitive matching with warning logs for unknown values
- **TTL Cache:** Implemented 1000-entry limit with 5-minute expiration and LRU eviction

#### Frontend Updates
- **Tailwind CSS v4:** Migrated to `@theme` blocks for custom color definitions
- **Custom Design System:** Institutional trading terminal aesthetic with apex/lamont color palette

#### Core System
- Complete deterministic analysis engine with 115 protocols
  - 80 Tier protocols (T01-T80) for structural analysis
  - 25 Learning protocols (LP01-LP25) for label generation
  - 5 MonsterRunner protocols (MR01-MR05) for extreme move detection
  - 5 Omega directives (Ω1-Ω5) for safety overrides
- ApexLab v2 offline labeling environment
  - Window-based feature extraction
  - Teacher labeling via deterministic engine
  - Future outcome calculation with leakage prevention
  - Quality tier assignment (A+/A/B/C/D)
- ApexCore v2 neural model family
  - Big variant (5-model ensemble, AUC 0.782)
  - Mini variant (3-model ensemble, AUC 0.754)
  - 5 prediction heads (quantra_score, runner_prob, quality_tier, avoid_trade, regime)
  - Manifest verification with SHA256 hashes
- PredictiveAdvisor fail-closed ranker
  - Ensemble disagreement detection
  - Avoid-trade safety gating
  - Engine authority preservation

#### Infrastructure
- FastAPI backend with 36 REST endpoints
- React/Vite dashboard (ApexDesk)
- Comprehensive test suite (970 tests)
- Regulatory compliance framework (99.25% score)

#### Documentation
- Complete investor documentation bundle (30+ files)
  - Executive summary and one-pager
  - Technical deep dives
  - Risk and compliance documentation
  - Business and commercial analysis
  - Engineering and operations runbooks
  - Investor FAQ
- Architecture diagrams and visual assets
- SBOM and provenance manifest

### Changed
- Migrated from prototype to production-quality codebase
- Upgraded from v8.x to v9.x architecture
- Enhanced determinism verification (150 iterations, 3x FINRA requirement)

### Security
- No secrets in codebase
- Environment variable isolation for API keys
- Model integrity verification via manifests
- Proof logging for audit trail

---

## [8.0.0] - 2025-10-15

### Added
- Initial ApexCore v2 architecture
- Basic ApexLab labeling
- Prototype scanner implementation

### Changed
- Migrated from v7.x protocol structure
- Unified configuration system

---

## [7.0.0] - 2025-09-01

### Added
- Core engine prototype
- Initial protocol implementations
- Basic test framework

---

## Legend

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

---

## Versioning

QuantraCore Apex uses the following versioning scheme:

- **Major.Minor.Patch-Suffix**
- **Major**: Breaking changes or major architecture shifts
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible
- **Suffix**: Release stage (A = Alpha, B = Beta, RC = Release Candidate)

---

*QuantraCore Apex | Lamont Labs | November 2025*
