# Compliance and Usage Policies

**Document Classification:** Investor Due Diligence — Risk/Compliance  
**Version:** 9.0-A  
**Date:** November 2025  

---

## If We Use This at a Firm, What Policies Need to Be in Place?

This document outlines the compliance considerations and usage policies for different deployment scenarios.

---

## Usage Modes

### Mode 1: Personal Research

**Description:** Individual using the system for personal market research and education.

**Requirements:**
- No regulatory registration required
- Standard research disclaimers
- No client assets involved
- No trading signals shared publicly

**Policies:**
- Maintain personal trading logs
- Acknowledge research-only nature
- Do not represent as investment advice

### Mode 2: Proprietary Trading

**Description:** Trading firm using the system for internal research and proprietary capital.

**Requirements:**
- Firm must be appropriately registered (if required by jurisdiction)
- Internal risk management policies
- Trade surveillance capabilities
- Proper record retention

**Policies:**
- Document decision rationale
- Log all system outputs
- Regular compliance reviews
- Model validation procedures

### Mode 3: Client Assets

**Description:** Using the system to manage client capital or provide investment advice.

**Requirements:**
- SEC/FINRA registration (US)
- Investment adviser compliance program
- Client disclosures and agreements
- Fiduciary duty obligations

**Policies:**
- Full disclosure of methodology
- Client suitability assessments
- Best execution obligations
- Performance reporting standards

**Important:** This mode requires significant additional compliance infrastructure beyond what the system provides.

---

## Regulatory Framework Awareness

### SEC (US Securities and Exchange Commission)

| Regulation | Relevance | System Support |
|------------|-----------|----------------|
| Reg SHO | Short sale rules | N/A (research-only) |
| Reg NMS | Market access | N/A (no execution) |
| Reg SCI | System compliance | Audit logging |
| Rule 15c3-5 | Risk controls | Kill-switch architecture |

### FINRA (Financial Industry Regulatory Authority)

| Regulation | Relevance | System Support |
|------------|-----------|----------------|
| Rule 3110 | Supervision | Trace logging |
| FINRA 15-09 | Algo testing | Determinism verification |
| Rule 4512 | Record retention | Proof logs |

### MiFID II (European Union)

| Regulation | Relevance | System Support |
|------------|-----------|----------------|
| RTS 6 Art 17 | Algo testing | Determinism, stress tests |
| RTS 25 | Clock sync | Timestamp logging |
| RTS 6 Art 18 | Kill functionality | Ω2 kill-switch |

### Basel Committee (Banking)

| Standard | Relevance | System Support |
|----------|-----------|----------------|
| BCBS 239 | Data governance | Provenance chain |

---

## Compliance Score

The system maintains a compliance score based on:

```
Overall Score: 99.25% (EXCEPTIONAL)

Components:
- Determinism: 150 iterations (3x FINRA requirement)
- Stress tests: 5x standard volume
- Latency margins: 50ms buffer
- Documentation: Complete
```

This score is accessible via:
- API: `GET /compliance/score`
- Dashboard: Compliance view

---

## Logging and Retention

### What Gets Logged

| Log Type | Contents | Retention |
|----------|----------|-----------|
| Proof logs | Input/output hashes, traces | 7 years |
| Signal logs | All generated signals | 7 years |
| Audit logs | System access, config changes | 7 years |
| Error logs | Exceptions, failures | 90 days |

### Log Format

```json
{
  "timestamp": "2025-11-29T10:15:00.000Z",
  "event_type": "SIGNAL_GENERATED",
  "symbol": "XYZ",
  "input_hash": "sha256:abc123...",
  "output_hash": "sha256:def456...",
  "quantra_score": 72.4,
  "omega_status": "Ω4_ACTIVE",
  "compliance_mode": "RESEARCH_ONLY"
}
```

### Retention Support

The system supports audit/regulatory requirements through:
- Immutable proof logs
- Timestamp verification
- Hash chain integrity
- Export capabilities

---

## Disclaimers and Disclosures

### Standard Disclaimer (All Modes)

> "QuantraCore Apex is a research and analysis tool that provides structural probability assessments. Outputs are for informational and educational purposes only and do not constitute investment advice, trading signals, or recommendations to buy or sell any security. Past structural patterns do not guarantee future results. All investment decisions should be made in consultation with qualified financial professionals. The system operates in research-only mode by default."

### Required Disclosures for Client-Facing Use

If used in any client-facing capacity, additional disclosures required:

1. **Methodology Disclosure:** How signals are generated
2. **Limitations Disclosure:** Where the system is weak
3. **Risk Disclosure:** Potential for loss
4. **Conflict Disclosure:** Any conflicts of interest
5. **Fee Disclosure:** All costs and fees

---

## Policy Templates

### Acceptable Use Policy Template

```
1. System is used for research and analysis only
2. Outputs are not shared as trading recommendations
3. All regulatory requirements are met
4. Logs are retained per policy
5. Compliance reviews are conducted [quarterly/annually]
6. Any issues are reported immediately
```

### Model Governance Policy Template

```
1. Models are validated before promotion
2. Performance is monitored continuously
3. Drift triggers investigation
4. Retraining follows documented process
5. Changes are logged with rationale
```

---

## Compliance Contacts

For regulatory questions beyond this documentation:
- Consult qualified legal counsel
- Engage compliance consultants
- Contact relevant regulators directly

The system provides infrastructure for compliance; it does not replace legal advice.

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
