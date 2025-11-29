# Operations and Runbooks

**Document Classification:** Investor Due Diligence — Engineering  
**Version:** 9.0-A  
**Date:** November 2025  

---

## If Someone Deployed This, How Do They Run It Day-to-Day?

This document provides operational guidance for running QuantraCore Apex in research and development environments.

---

## System Startup

### K6 Desktop (Research)

**Hardware Target:**
- CPU: 4+ cores
- RAM: 16GB minimum
- Storage: 50GB SSD

**Startup Sequence:**

```bash
# 1. Start backend
cd /path/to/quantracore-apex
source venv/bin/activate
uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000

# 2. Start frontend
cd dashboard
npm run dev

# 3. Verify health
curl http://localhost:8000/health
```

### Institutional Server (Production-like)

**Hardware Target:**
- CPU: 16+ cores
- RAM: 64GB+
- Storage: 500GB+ SSD

**Startup Sequence:**

```bash
# 1. Start with production settings
export APEX_ENV=production
export POLYGON_API_KEY=your_key

# 2. Start backend with Gunicorn
gunicorn src.quantracore_apex.server.app:app \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120

# 3. Start frontend with production build
cd dashboard
npm run build
npm run preview

# 4. Verify all endpoints
./scripts/health_check.sh
```

---

## Daily Operations

### Morning Checklist

| Task | Command | Expected |
|------|---------|----------|
| Health check | `curl /health` | `{"status": "healthy"}` |
| Compliance score | `curl /compliance/score` | Score >95% |
| Data status | `curl /data/status` | All feeds operational |
| Model status | `curl /models/status` | All models loaded |

### Typical Research Workflow

```
08:00 - System startup and health check
08:15 - Review overnight data updates
08:30 - Run universe scan (momentum mode)
09:00 - Review top candidates
10:00 - Deep dive analysis on selected symbols
12:00 - Export proof logs for review
14:00 - Run alternative scan modes
16:00 - End-of-day summary
16:30 - System backup
```

### Weekly Tasks

| Task | Frequency | Purpose |
|------|-----------|---------|
| Log rotation | Weekly | Prevent disk fill |
| Backup | Weekly | Data protection |
| Metric review | Weekly | Performance monitoring |
| Model drift check | Weekly | Ensure model validity |

### Monthly Tasks

| Task | Frequency | Purpose |
|------|-----------|---------|
| Full test suite | Monthly | Regression check |
| Dependency update | Monthly | Security patches |
| Documentation review | Monthly | Keep docs current |
| Model retraining eval | Monthly | Consider refresh |

---

## Common Operations

### Running a Full Universe Scan

```bash
# Via API
curl -X POST http://localhost:8000/scan/universe \
    -H "Content-Type: application/json" \
    -d '{"mode": "momentum", "cap_bucket": "mid"}'

# Expected: List of candidates with QuantraScores
```

### Analyzing a Single Symbol

```bash
# Via API
curl http://localhost:8000/analyze/AAPL

# Expected: Full analysis with protocol traces
```

### Exporting Proof Logs

```bash
# Via API
curl http://localhost:8000/proof/export?symbol=AAPL&date=2025-11-29

# Expected: JSON proof log for the specified analysis
```

### Checking Compliance

```bash
# Via API
curl http://localhost:8000/compliance/report

# Expected: Detailed compliance report
```

---

## Failure Modes and Quick Checks

### Symptom: Backend Not Starting

| Check | Command | Fix |
|-------|---------|-----|
| Port in use | `lsof -i :8000` | Kill conflicting process |
| Python env | `which python` | Activate virtualenv |
| Dependencies | `pip check` | `pip install -r requirements.txt` |
| Config error | Check logs | Fix configuration |

### Symptom: Frontend Not Loading

| Check | Command | Fix |
|-------|---------|-----|
| Node version | `node --version` | Use Node 18+ |
| Dependencies | `npm ls` | `npm install` |
| Build error | Check console | Fix TypeScript errors |
| API unreachable | `curl /health` | Start backend first |

### Symptom: Slow Performance

| Check | Command | Fix |
|-------|---------|-----|
| CPU usage | `top` | Reduce concurrent scans |
| Memory usage | `free -h` | Increase RAM or reduce scope |
| Disk I/O | `iostat` | Use faster storage |
| Data loading | Check logs | Warm cache |

### Symptom: Model Not Loading

| Check | Command | Fix |
|-------|---------|-----|
| File exists | `ls models/` | Restore from backup |
| Manifest valid | Check manifest.json | Regenerate manifest |
| Hash mismatch | Verify hash | Retrain or restore |
| Permissions | `ls -la` | Fix file permissions |

---

## Monitoring

### Key Metrics to Watch

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| API latency | <100ms | >500ms |
| Memory usage | <80% | >90% |
| CPU usage | <70% | >90% |
| Error rate | <0.1% | >1% |
| Scan time | <1 min | >5 min |

### Log Locations

| Log | Path | Retention |
|-----|------|-----------|
| API access | `logs/api_access.log` | 30 days |
| Error | `logs/error.log` | 90 days |
| Proof | `proof_logs/` | 7 years |
| Audit | `logs/audit.log` | 7 years |

---

## Emergency Procedures

### Ω2 Kill-Switch Triggered

1. All activity halted automatically
2. Review trigger condition in logs
3. Investigate root cause
4. Resolve underlying issue
5. Manual reset required to resume

### Data Corruption Detected

1. Stop all processing
2. Identify affected data
3. Restore from backup
4. Verify integrity
5. Resume operations

### Model Integrity Failure

1. System automatically falls back to engine-only
2. Investigate hash mismatch
3. Restore model from backup or retrain
4. Verify new model passes thresholds
5. Promote and resume

---

## Backup and Recovery

### What to Backup

| Data | Frequency | Retention |
|------|-----------|-----------|
| Configuration | Daily | 30 days |
| Proof logs | Daily | 7 years |
| Models | After training | 1 year |
| Datasets | After generation | 6 months |

### Restore Procedure

```bash
# 1. Stop services
systemctl stop apex-backend apex-frontend

# 2. Restore from backup
./scripts/restore_backup.sh latest

# 3. Verify integrity
./scripts/verify_integrity.sh

# 4. Restart services
systemctl start apex-backend apex-frontend

# 5. Health check
curl http://localhost:8000/health
```

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
