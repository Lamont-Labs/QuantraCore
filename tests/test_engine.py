from src.core.engine import generate_signal

def test_determinism():
    a = generate_signal("AAPL", 42)
    b = generate_signal("AAPL", 42)
    assert a["hash"] == b["hash"]

def test_score_range():
    s = generate_signal("AAPL", 42)
    assert 0 <= s["score"] <= 100
