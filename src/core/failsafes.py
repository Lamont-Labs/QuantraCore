"""Guardrails that fail closed."""
import logging

def safe_execute(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        logging.error("FAILSAFE: %s", e)
        return {"error": str(e), "status": "fail_closed"}

def enforce_bounds(score: float, lo=0.0, hi=100.0):
    return {"score": score, "status": "valid" if lo <= score <= hi else "fail_closed"}
