from src.core.zde import zero_drawdown_entry as zde
def test_zde_threshold():
    assert zde(60, 50) is True
    assert zde(40, 50) is False
