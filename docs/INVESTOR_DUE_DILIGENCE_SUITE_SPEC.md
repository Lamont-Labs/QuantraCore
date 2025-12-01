# Investor Due Diligence Suite — Complete Specification

**Version:** 1.0  
**Status:** Planning / Scoped  
**Last Updated:** 2025-12-01  
**Purpose:** Institutional-grade investor relations and due diligence platform

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Module 1: Performance Analytics Console](#2-module-1-performance-analytics-console)
3. [Module 2: Trade Intelligence Center](#3-module-2-trade-intelligence-center)
4. [Module 3: Risk Command Center](#4-module-3-risk-command-center)
5. [Module 4: Compliance & Governance Hub](#5-module-4-compliance--governance-hub)
6. [Module 5: Audit Trail & Verification](#6-module-5-audit-trail--verification)
7. [Module 6: Legal & Disclosure Center](#7-module-6-legal--disclosure-center)
8. [Module 7: Secure Data Room](#8-module-7-secure-data-room)
9. [Module 8: Investor Portal](#9-module-8-investor-portal)
10. [Technical Architecture](#10-technical-architecture)
11. [Implementation Phases](#11-implementation-phases)

---

## 1. Executive Summary

### 1.1 Purpose

The Investor Due Diligence Suite provides institutional investors with complete transparency into QuantraCore Apex's trading operations, performance, risk management, and compliance posture. It transforms raw trading data into investor-grade documentation suitable for LP due diligence, regulatory review, and capital allocation decisions.

### 1.2 Target Users

| User Type | Primary Needs |
|-----------|---------------|
| **Limited Partners (LPs)** | Performance verification, risk assessment, governance review |
| **Family Offices** | Track record validation, strategy understanding, fee transparency |
| **Institutional Allocators** | Operational due diligence, compliance verification, benchmark comparison |
| **Auditors** | Trade reconciliation, data integrity, control verification |
| **Regulators** | Compliance documentation, record retention, disclosure verification |

### 1.3 Key Differentiators

- Real-time transparency (not quarterly snapshots)
- AI model explainability built-in
- Automated compliance attestation
- Verifiable track record with broker reconciliation
- Self-service investor portal

---

## 2. Module 1: Performance Analytics Console

### 2.1 Equity Curve Dashboard

| Component | Description | Update Frequency |
|-----------|-------------|------------------|
| Cumulative Returns | Interactive equity curve with drawdown overlay | Real-time |
| Rolling Returns | 1D, 1W, 1M, 3M, 6M, 1Y, YTD, ITD periods | Daily |
| Drawdown Analysis | Current, max, average drawdown with recovery time | Real-time |
| Monthly Returns Grid | Heat map of monthly P&L by year | Daily |

### 2.2 Risk-Adjusted Metrics

| Metric | Calculation | Benchmark |
|--------|-------------|-----------|
| Sharpe Ratio | (Return - Rf) / Std Dev | S&P 500, Russell 2000 |
| Sortino Ratio | (Return - Rf) / Downside Dev | Category peers |
| Calmar Ratio | CAGR / Max Drawdown | Hedge fund indices |
| Information Ratio | Alpha / Tracking Error | Strategy benchmark |
| Omega Ratio | Probability weighted gains/losses | 0% threshold |
| Tail Ratio | 95th percentile gain / 5th percentile loss | - |

### 2.3 Benchmark Comparison

| Feature | Description |
|---------|-------------|
| Multi-Benchmark | Compare vs SPY, QQQ, IWM, custom indices |
| Alpha Attribution | Decompose returns into market beta + alpha |
| Correlation Matrix | Rolling correlation to major indices |
| Beta Exposure | Time-varying market sensitivity |

### 2.4 Performance Attribution

| View | Breakdown |
|------|-----------|
| By Strategy | Long vs Short vs Intraday vs Swing |
| By Sector | Technology, Healthcare, Financials, etc. |
| By Market Cap | Mega, Large, Mid, Small, Micro, Nano, Penny |
| By Signal Type | Runner, Quality, Timing-based entries |
| By Time Period | Best/worst days, months, market conditions |

---

## 3. Module 2: Trade Intelligence Center

### 3.1 Trade History & Analytics

| Feature | Description |
|---------|-------------|
| Complete Trade Log | Every trade with full metadata |
| Advanced Filtering | By date, symbol, strategy, outcome, size |
| Trade Tagging | Custom tags for analysis (earnings play, momentum, etc.) |
| Export Formats | CSV, Excel, PDF, JSON |

### 3.2 Trade Statistics

| Metric | Description |
|--------|-------------|
| Win Rate | % of profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Average Win/Loss | Mean P&L for winning/losing trades |
| Win/Loss Ratio | Average win size / Average loss size |
| Expectancy | Expected P&L per trade |
| Consecutive Wins/Losses | Streak analysis |
| Time in Trade | Average holding period by strategy |

### 3.3 Execution Quality

| Metric | Description |
|--------|-------------|
| Slippage Analysis | Entry/exit price vs signal price |
| Fill Rate | % of orders fully executed |
| Latency Tracking | Time from signal to execution |
| Cost Analysis | Commission, spread, market impact |

### 3.4 Signal Performance

| Component | Description |
|-----------|-------------|
| QuantraScore Accuracy | Predicted vs actual outcomes by score bucket |
| Runner Hit Rate | % of runner predictions that achieved 10%+ |
| Timing Accuracy | Entry timing bucket vs actual move timing |
| Runup Prediction | Predicted top vs actual high |
| Conviction Tier Analysis | Performance by high/medium/low conviction |
| Protocol Attribution | Which protocols drove best/worst trades |

---

## 4. Module 3: Risk Command Center

### 4.1 Real-Time Risk Dashboard

| Component | Description | Alert Threshold |
|-----------|-------------|-----------------|
| Current Exposure | Total $ at risk | 80% of max |
| Position Count | Active positions | 80% of max |
| Sector Concentration | % in any single sector | >30% |
| Symbol Concentration | % in any single symbol | >10% |
| Leverage Utilization | Current vs max leverage | >3x |
| Margin Usage | Maintenance margin buffer | <150% |

### 4.2 Stress Testing

| Scenario | Description |
|----------|-------------|
| Market Crash (-20%) | Portfolio impact of broad selloff |
| Flash Crash | Impact of 5% intraday drop |
| Sector Rotation | Impact if current hot sector reverses |
| Volatility Spike | Impact of VIX doubling |
| Liquidity Crisis | Impact if spreads widen 5x |
| Correlation Breakdown | Impact if diversification fails |
| Black Swan | Historical worst-case scenarios |

### 4.3 VaR & Expected Shortfall

| Metric | Methodology |
|--------|-------------|
| 1-Day VaR (95%) | Historical simulation |
| 1-Day VaR (99%) | Parametric + Monte Carlo |
| Expected Shortfall | Average loss beyond VaR |
| Component VaR | Per-position contribution |
| Marginal VaR | Impact of adding position |

### 4.4 Limit Monitoring

| Limit | Current | Max | Status |
|-------|---------|-----|--------|
| Total Exposure | Dynamic | $100,000 | Green/Yellow/Red |
| Per Symbol | Dynamic | $10,000 | Green/Yellow/Red |
| Position Count | Dynamic | 50 | Green/Yellow/Red |
| Daily Loss | Dynamic | -5% | Green/Yellow/Red |
| Leverage | Dynamic | 4x | Green/Yellow/Red |

---

## 5. Module 4: Compliance & Governance Hub

### 5.1 Policy Management

| Document | Description | Review Cycle |
|----------|-------------|--------------|
| Trading Policy | Approved strategies, instruments, limits | Annual |
| Risk Policy | Risk limits, escalation procedures | Annual |
| Compliance Manual | Regulatory requirements, procedures | Annual |
| Personal Trading Policy | Operator trading restrictions | Annual |
| Data Handling Policy | PII, retention, security | Annual |
| Business Continuity Plan | Disaster recovery procedures | Annual |

### 5.2 Attestation Workflow

| Attestation | Frequency | Responsible |
|-------------|-----------|-------------|
| Daily Reconciliation | Daily | Operations |
| Risk Limit Compliance | Daily | Risk |
| Policy Acknowledgment | Annual | All staff |
| Control Effectiveness | Quarterly | Compliance |
| Incident Review | As needed | Compliance |

### 5.3 Regulatory Readiness

| Regulation | Applicability | Status |
|------------|---------------|--------|
| SEC Rule 17a-4 | Record retention | Tracking |
| Reg D / Reg CF | Capital raising | If applicable |
| AML/KYC | Investor onboarding | Required |
| GDPR/CCPA | Data privacy | Required |
| SMS TCPA | Alert consent | Required |

### 5.4 Incident Management

| Field | Description |
|-------|-------------|
| Incident ID | Unique identifier |
| Timestamp | When detected |
| Severity | Critical / High / Medium / Low |
| Category | Trading / System / Compliance / Data |
| Description | What happened |
| Root Cause | Why it happened |
| Resolution | How it was fixed |
| Prevention | How to prevent recurrence |

---

## 6. Module 5: Audit Trail & Verification

### 6.1 Track Record Verification

| Component | Description |
|-----------|-------------|
| Broker Statement Import | Parse official Alpaca statements |
| Reconciliation Engine | Match internal logs to broker confirms |
| Discrepancy Flagging | Highlight any mismatches |
| Auditor Export | Formatted for CPA review |

### 6.2 Data Integrity

| Check | Description |
|-------|-------------|
| Hash Verification | Cryptographic proof of log integrity |
| Timestamp Validation | Ensure chronological consistency |
| Gap Detection | Flag missing data periods |
| Anomaly Detection | Flag unusual patterns |

### 6.3 Third-Party Verification

| Service | Purpose |
|---------|---------|
| Broker Confirms | Official trade confirmations |
| Bank Statements | Cash movement verification |
| Prime Broker Reports | Position reconciliation |
| Administrator NAV | Independent valuation |

### 6.4 Audit Reports

| Report | Frequency | Audience |
|--------|-----------|----------|
| Monthly Reconciliation | Monthly | Internal |
| Quarterly Performance | Quarterly | Investors |
| Annual Audit Package | Annual | Auditors |
| Regulatory Report | As required | Regulators |

---

## 7. Module 6: Legal & Disclosure Center

### 7.1 Required Disclaimers

```
IMPORTANT DISCLOSURES:

1. NOT INVESTMENT ADVICE: QuantraCore Apex is a trading research and 
   execution system. Nothing herein constitutes investment advice or 
   a recommendation to buy or sell any security.

2. PAST PERFORMANCE: Past performance is not indicative of future 
   results. Trading involves substantial risk of loss.

3. MODEL LIMITATIONS: AI/ML models have inherent limitations including:
   - Training data may not reflect future market conditions
   - Model drift may degrade prediction accuracy over time
   - Black swan events may cause unexpected losses
   - System failures may prevent timely execution

4. RISK OF LOSS: You may lose some or all of your invested capital.
   Only invest funds you can afford to lose entirely.

5. NO GUARANTEES: No representation is made that any account will 
   achieve profits or losses similar to those shown.
```

### 7.2 Risk Factor Disclosures

| Risk Category | Specific Risks |
|---------------|----------------|
| Market Risk | Price movements, volatility, gaps |
| Model Risk | Prediction errors, overfitting, drift |
| Execution Risk | Slippage, partial fills, latency |
| Liquidity Risk | Wide spreads, inability to exit |
| Technology Risk | System failures, connectivity issues |
| Regulatory Risk | Rule changes, restrictions |
| Operational Risk | Human error, process failures |

### 7.3 Fee Disclosure

| Fee Type | Description | Amount |
|----------|-------------|--------|
| Management Fee | Annual fee on AUM | X% |
| Performance Fee | Fee on profits above hurdle | X% |
| Trading Costs | Commissions, spreads | Pass-through |
| Data Costs | Market data fees | Pass-through |

### 7.4 Consent Management

| Consent Type | Purpose | Required |
|--------------|---------|----------|
| SMS Alerts | Trading signal notifications | Yes for SMS |
| Email Communications | Investor updates | Yes |
| Data Sharing | Third-party verification | Optional |
| Marketing | Promotional materials | Optional |

---

## 8. Module 7: Secure Data Room

### 8.1 Document Categories

| Category | Documents |
|----------|-----------|
| **Formation** | Operating Agreement, Subscription Docs, Side Letters |
| **Performance** | Monthly/Quarterly Reports, Audited Financials |
| **Strategy** | Investment Thesis, Strategy Deck, Model Documentation |
| **Risk** | Risk Policy, Stress Test Results, VaR Reports |
| **Compliance** | Compliance Manual, Regulatory Filings, Attestations |
| **Operations** | BCP, Vendor Due Diligence, IT Security Policy |
| **Legal** | PPM, LPA, Advisory Agreements |

### 8.2 Access Control

| Role | Access Level |
|------|--------------|
| Administrator | Full access, user management |
| Investor | Own documents, performance reports |
| Prospect | Limited deck, track record summary |
| Auditor | Full read access, export capability |
| Regulator | Compliance documents, trade records |

### 8.3 Security Features

| Feature | Description |
|---------|-------------|
| Encryption | AES-256 at rest, TLS in transit |
| Watermarking | Dynamic watermarks with viewer ID |
| Access Logging | Complete audit trail of views/downloads |
| Expiring Links | Time-limited document access |
| Download Control | Prevent/allow downloads per document |
| Two-Factor Auth | Required for sensitive documents |

---

## 9. Module 8: Investor Portal

### 9.1 Investor Dashboard

| Component | Description |
|-----------|-------------|
| Account Summary | Current NAV, P&L, IRR |
| Capital Account | Contributions, distributions, balance |
| Performance Chart | Personal equity curve |
| Document Center | Access to statements, K-1s, reports |
| Communication Center | Messages from fund |

### 9.2 Self-Service Features

| Feature | Description |
|---------|-------------|
| Statement Download | Monthly/quarterly statements |
| Tax Documents | K-1, 1099 access |
| Subscription Status | View/update subscription info |
| Contact Update | Update contact information |
| Preference Management | Communication preferences |

### 9.3 Reporting

| Report | Frequency | Delivery |
|--------|-----------|----------|
| Flash Report | Daily | Portal |
| Monthly Letter | Monthly | Email + Portal |
| Quarterly Report | Quarterly | Email + Portal |
| Annual Report | Annual | Email + Portal |
| Tax Documents | Annual | Portal |

---

## 10. Technical Architecture

### 10.1 Frontend Components

```
investor-desk/
├── src/
│   ├── modules/
│   │   ├── performance/       # Performance Analytics Console
│   │   ├── trades/            # Trade Intelligence Center
│   │   ├── risk/              # Risk Command Center
│   │   ├── compliance/        # Compliance & Governance Hub
│   │   ├── audit/             # Audit Trail & Verification
│   │   ├── legal/             # Legal & Disclosure Center
│   │   ├── dataroom/          # Secure Data Room
│   │   └── portal/            # Investor Portal
│   ├── components/
│   │   ├── charts/            # Recharts/D3 visualizations
│   │   ├── tables/            # Data grids with filtering
│   │   ├── forms/             # Input components
│   │   └── common/            # Shared UI components
│   └── services/
│       ├── api/               # Backend API clients
│       ├── auth/              # Authentication
│       └── export/            # PDF/Excel generation
```

### 10.2 Backend API Endpoints

| Endpoint Group | Purpose | Count |
|----------------|---------|-------|
| `/investor/performance/*` | Performance metrics & charts | ~15 |
| `/investor/trades/*` | Trade history & analytics | ~10 |
| `/investor/risk/*` | Risk metrics & stress tests | ~12 |
| `/investor/compliance/*` | Compliance status & attestations | ~8 |
| `/investor/audit/*` | Audit trails & reconciliation | ~6 |
| `/investor/documents/*` | Data room operations | ~10 |
| `/investor/portal/*` | Investor self-service | ~8 |
| **Total** | | **~70 endpoints** |

### 10.3 Data Sources

| Source | Data |
|--------|------|
| `investor_logs/` | Existing trade logs |
| Alpaca API | Account, positions, orders |
| PostgreSQL | Investor records, documents |
| File Storage | Document uploads |

### 10.4 Security Architecture

| Layer | Implementation |
|-------|----------------|
| Authentication | JWT + Refresh tokens |
| Authorization | Role-based access control (RBAC) |
| Encryption | TLS 1.3, AES-256 |
| Audit Logging | Immutable audit trail |
| Session Management | Secure session handling |

---

## 11. Implementation Phases

### Phase 1: Core Analytics (4-6 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Equity Curve Dashboard | 4 | 3 charts | investor_logs parser | 20-25 |
| Risk-Adjusted Metrics | 6 | 2 tables, 1 chart | Metrics calculator | 25-30 |
| Benchmark Comparison | 4 | 2 charts | External data fetch | 15-20 |
| Performance Attribution | 5 | 4 views | Attribution engine | 25-30 |
| **Subtotal** | **19** | **12** | **4** | **85-105** |

### Phase 2: Trade Intelligence (3-4 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Trade History & Analytics | 5 | 2 tables | investor_logs query | 20-25 |
| Trade Statistics | 4 | 3 displays | Stats calculator | 15-20 |
| Execution Quality | 3 | 2 tables | Slippage analyzer | 15-20 |
| Signal Performance | 4 | 4 charts | ApexCore attribution | 20-25 |
| **Subtotal** | **16** | **11** | **4** | **70-90** |

### Phase 3: Risk Command Center (3-4 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Real-Time Risk Dashboard | 6 | 5 displays | Live position feed | 25-30 |
| Stress Testing | 4 | 3 scenarios | Monte Carlo engine | 30-40 |
| VaR & Expected Shortfall | 4 | 2 charts | VaR calculator | 20-25 |
| Limit Monitoring | 3 | 1 status board | Alert system | 10-15 |
| **Subtotal** | **17** | **11** | **4** | **85-110** |

### Phase 4: Compliance & Governance (4-5 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Policy Management | 4 | 2 forms, 1 list | Document storage | 25-30 |
| Attestation Workflow | 5 | 3 forms | Workflow engine | 35-45 |
| Regulatory Readiness | 3 | 2 checklists | Status tracker | 15-20 |
| Incident Management | 4 | 2 forms, 1 list | Incident logger (existing) | 20-25 |
| **Subtotal** | **16** | **11** | **4** | **95-120** |

### Phase 5: Audit Trail & Verification (3-4 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Broker Statement Import | 3 | 1 upload, 1 table | Alpaca statement parser | 30-40 |
| Reconciliation Engine | 4 | 2 diff views | Matching algorithm | 35-45 |
| Data Integrity Checks | 2 | 1 status page | Hash verification | 15-20 |
| Audit Reports | 3 | 2 report views | Report generator | 20-25 |
| **Subtotal** | **12** | **6** | **4** | **100-130** |

### Phase 6: Legal & Disclosure (2-3 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Disclaimer Management | 2 | 2 displays | Template engine | 15-20 |
| Risk Disclosure Generator | 3 | 1 form, 1 preview | Dynamic generator | 20-25 |
| Fee Disclosure | 2 | 1 table | Fee calculator | 10-15 |
| Consent Management | 3 | 2 forms | Consent tracker | 15-20 |
| **Subtotal** | **10** | **7** | **4** | **60-80** |

### Phase 7: Secure Data Room (4-5 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Document Storage | 5 | 3 views | File storage (S3/local) | 30-40 |
| Access Control (RBAC) | 4 | 2 admin panels | Auth integration | 35-45 |
| Watermarking & Security | 2 | 1 preview | PDF processor | 25-30 |
| Audit Logging | 2 | 1 log view | Access logger | 15-20 |
| **Subtotal** | **13** | **7** | **4** | **105-135** |

### Phase 8: Investor Portal (3-4 weeks)

| Component | Endpoints | UI Surfaces | Data Pipelines | Hours |
|-----------|-----------|-------------|----------------|-------|
| Investor Dashboard | 5 | 4 displays | Account aggregator | 25-30 |
| Capital Account | 3 | 2 tables | Transaction tracker | 20-25 |
| Self-Service Features | 4 | 3 forms | Profile manager | 20-25 |
| Report Delivery | 3 | 2 views | Google Docs (existing) | 15-20 |
| **Subtotal** | **15** | **11** | **4** | **80-100** |

---

## Total Estimated Effort (Revised)

| Phase | Endpoints | UI Surfaces | Duration | Hours |
|-------|-----------|-------------|----------|-------|
| Phase 1: Core Analytics | 19 | 12 | 4-6 weeks | 85-105 |
| Phase 2: Trade Intelligence | 16 | 11 | 3-4 weeks | 70-90 |
| Phase 3: Risk Command | 17 | 11 | 3-4 weeks | 85-110 |
| Phase 4: Compliance | 16 | 11 | 4-5 weeks | 95-120 |
| Phase 5: Audit Trail | 12 | 6 | 3-4 weeks | 100-130 |
| Phase 6: Legal | 10 | 7 | 2-3 weeks | 60-80 |
| Phase 7: Data Room | 13 | 7 | 4-5 weeks | 105-135 |
| Phase 8: Portal | 15 | 11 | 3-4 weeks | 80-100 |
| **TOTAL** | **118** | **76** | **26-35 weeks** | **680-870** |

**Note:** Phases can run in parallel where dependencies allow. With 2 parallel tracks:
- Track A: Phases 1, 3, 5, 7 (analytics, risk, audit, data room)
- Track B: Phases 2, 4, 6, 8 (trades, compliance, legal, portal)
- **Parallel Timeline:** 13-18 weeks

---

## Appendix A: Existing Infrastructure Leverage

The following existing components will be reused:

| Component | Location | Reuse |
|-----------|----------|-------|
| React + Vite + Tailwind | `dashboard/` | UI framework |
| FastAPI Backend | `src/quantracore_apex/server/` | API layer |
| Investor Trade Logs | `investor_logs/` | Data source |
| Google Docs Integration | Existing | Report export |
| Twilio SMS | Existing | Alert delivery |
| ApexCore V3 | Existing | Signal attribution |

---

## Appendix B: Integration Map

This map shows how each module leverages existing QuantraCore Apex infrastructure versus new build requirements.

### Existing Services Reused

| Service | Location | Reused By |
|---------|----------|-----------|
| **investor_logs/** | File system | Performance, Trade Intelligence, Audit |
| **ApexCore V3** | `prediction/apexcore_v3.py` | Signal Performance Attribution |
| **Incident Logger** | `hardening/incident_logger.py` | Compliance Hub |
| **Google Docs Export** | `integrations/google_docs/` | Report Delivery |
| **Twilio SMS** | `signals/sms_service.py` | Alert Notifications |
| **FastAPI Auth** | `server/app.py` | RBAC foundation |
| **Risk Engine** | `risk/` | Risk Command Center |
| **Kill Switch** | `hardening/kill_switch.py` | Operational Controls |
| **Protocol Manifest** | `config/protocol_manifest.yaml` | Compliance verification |

### New Build vs Reuse by Module

| Module | Reuse % | New Build | Notes |
|--------|---------|-----------|-------|
| Performance Analytics | 40% | 60% | Reuses investor_logs; new charts, metrics |
| Trade Intelligence | 50% | 50% | Reuses logs + ApexCore; new stats engine |
| Risk Command | 30% | 70% | Reuses risk engine; new VaR, stress tests |
| Compliance Hub | 20% | 80% | Reuses incident logger; mostly new |
| Audit Trail | 10% | 90% | Mostly new reconciliation engine |
| Legal & Disclosure | 0% | 100% | All new |
| Data Room | 10% | 90% | Reuses auth; new storage/security |
| Investor Portal | 30% | 70% | Reuses exports; new self-service |

---

## Appendix C: Broker Reconciliation Workflow

### Data Sources

| Source | Format | Access Method |
|--------|--------|---------------|
| Alpaca Trade Confirms | JSON API | `GET /v2/account/activities` |
| Alpaca Statements | PDF | Manual download or API |
| Internal Trade Logs | JSONL | `investor_logs/*.jsonl` |

### Reconciliation Steps

```
1. INGEST
   ├── Fetch Alpaca trade confirmations (API)
   ├── Parse internal investor_logs
   └── Normalize to common schema

2. MATCH
   ├── Match by order_id (primary key)
   ├── Match by symbol + timestamp + quantity (fallback)
   └── Flag unmatched records

3. VALIDATE
   ├── Compare fill prices (tolerance: $0.01)
   ├── Compare quantities (exact match)
   ├── Compare timestamps (tolerance: 1 second)
   └── Compare fees/commissions

4. REPORT
   ├── Generate reconciliation report
   ├── Flag discrepancies for review
   ├── Calculate match rate metrics
   └── Export for auditor review

5. ATTEST
   ├── Operations reviews discrepancies
   ├── Resolves or documents exceptions
   └── Signs off on reconciliation
```

### Reconciliation Schema

```python
class ReconciliationRecord:
    internal_trade_id: str
    broker_trade_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    internal_price: float
    broker_price: float
    price_diff: float
    internal_timestamp: datetime
    broker_timestamp: datetime
    time_diff_seconds: float
    match_status: str  # matched/unmatched/discrepancy
    discrepancy_type: Optional[str]
    resolution: Optional[str]
    attested_by: Optional[str]
    attested_at: Optional[datetime]
```

### Controls

| Control | Description | Responsible |
|---------|-------------|-------------|
| Daily Reconciliation | All trades reconciled T+1 | Operations |
| Exception Review | Discrepancies investigated within 24h | Operations |
| Weekly Sign-off | Management attestation | Compliance |
| Monthly Audit | Independent review of reconciliation | Internal Audit |

---

**Document Status:** Complete Specification  
**Ready for Implementation:** Yes, on user request  
**Prerequisites:** None (all infrastructure exists)
