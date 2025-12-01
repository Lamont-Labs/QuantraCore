# Changelog

All notable changes to QuantraCore Apex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [9.0.0-A] - 2025-12-01

### Added

#### ApexDesk Dashboard (December 2025)
- **9 Real-Time Panels:** Complete institutional trading dashboard
  - SystemStatusPanel: Real-time system health monitoring
  - PortfolioPanel: Live Alpaca paper trading portfolio
  - TradingSetupsPanel: Top trading opportunities by QuantraScore
  - ModelMetricsPanel: ApexCore V3 model performance (7 prediction heads)
  - AutoTraderPanel: Autonomous swing trade execution
  - SignalsAlertsPanel: Live signals and SMS alert status
  - RunnerScreenerPanel: Low-float penny stock scanner (110 symbols)
  - ContinuousLearningPanel: ML training orchestrator status
  - LogsProvenancePanel: System logs and audit trail
- **Velocity Mode System:** Standard (30s), High Velocity (5s), Turbo (2s) refresh rates
- **Technology Upgrade:** React 18.2, Vite 7.2, Tailwind CSS 4.0, TypeScript
- **Extended Market Hours:** Pre-market (4 AM-9:30 AM), Regular (9:30 AM-4 PM), After-hours (4 PM-8 PM ET)

#### Full Trading Capabilities (November 2025)
- **All Position Types Enabled:** Long, short, margin, intraday, swing, scalping
- **Execution Mode:** Upgraded from RESEARCH to PAPER (live paper trading)
- **Short Selling:** Unblocked for full directional flexibility
- **Margin Trading:** Enabled with 4x max leverage
- **Risk Limits:** $100K max exposure, $10K per symbol, 50 positions max
- **Trading Capabilities Endpoint:** `/trading_capabilities` for real-time config

#### Alpaca Data Adapter
- **New Primary Data Source:** Alpaca Markets API (200 requests/minute - 40x faster than Polygon)
- **Retry Logic:** Exponential backoff with 3 retries for production resilience
- **Rate Limiting:** Smart delays prevent API throttling
- **Error Handling:** Graceful recovery from timeouts, connection errors, server errors

#### Swing Trading Scanner
- **Real Scan Modes API:** Uses `/scan_universe_mode` endpoint with `config/scan_modes.yaml`
- **4 Pre-configured Modes:** Momentum Runners, Mid-Cap Focus, Blue Chips, High Vol Small Caps
- **Data Provider Validation:** Verifies real data sources before scanning
- **No Hardcoded Symbols:** All universes configured via YAML

#### Security Hardening
- **API Authentication:** Added `X-API-Key` header verification for protected endpoints
- **CORS Restriction:** Changed from wildcard to regex pattern (localhost + Replit domains only)
- **Non-blocking Rate Limiting:** Updated Polygon and Binance adapters with async-compatible delays
- **Timeframe Validation:** Added case-insensitive matching with warning logs for unknown values
- **TTL Cache:** Implemented 1000-entry limit with 5-minute expiration and LRU eviction

#### Frontend Updates
- **Tailwind CSS v4:** Migrated to `@theme` blocks for custom color definitions
- **Custom Design System:** Institutional trading terminal aesthetic with apex/lamont color palette

#### System Statistics
- **Total Files:** 516 source files
- **Total Lines:** 121,207 lines of code
- **API Endpoints:** 148 REST endpoints
- **Dashboard Panels:** 9 real-time panels
- **Development Stage:** Beta / Production-Ready (Paper Mode)

#### Core System
- Complete deterministic analysis engine with 145 protocols
  - 80 Tier protocols (T01-T80) for structural analysis
  - 25 Learning protocols (LP01-LP25) for label generation
  - 20 MonsterRunner protocols (MR01-MR20) for extreme move detection
  - 20 Omega directives (Ω1-Ω20) for safety overrides
- ApexLab v2 offline labeling environment
  - Window-based feature extraction
  - Teacher labeling via deterministic engine
  - Future outcome calculation with leakage prevention
  - Quality tier assignment (A+/A/B/C/D)
- ApexCore V3 neural model family
  - 7 prediction heads (quantrascore, runner, quality, avoid, regime, timing, runup)
  - Training samples: 6,085+ | 29 diverse symbols | 32,397 intraday bars
  - Accuracy: runner 94.99%, quality 81.68%, avoid 99.34%, regime 85.95%, timing 88.33%
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
