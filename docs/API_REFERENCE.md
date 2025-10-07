# API Reference (Demo)
GET /health → {status,version,time}  
GET /score?ticker=AAPL&seed=42 → deterministic score JSON  
GET /risk/hud → {ticker,score,filters[]}  
GET /audit/export → writes dist/golden_demo_outputs/audit_export.json
