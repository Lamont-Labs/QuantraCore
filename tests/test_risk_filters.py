from src.core.risk_filters import run_filters

def test_asp_count():
    assert len(run_filters(10.0)) == 20

def test_fail_closed_negative():
    res = run_filters(-1.0)
    assert all(r["result"] in ("pass","block") for r in res)
