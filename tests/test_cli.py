from src.core.engine import generate_signal

def test_engine_local_hash():
    s = generate_signal("AAPL", 42)
    assert "hash" in s and isinstance(s["hash"], str)
