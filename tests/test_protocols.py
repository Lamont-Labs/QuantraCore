from src.core.protocols import PROTOCOLS
def test_protocols_70():
    assert len(PROTOCOLS) == 70 and PROTOCOLS[0] == "T01" and PROTOCOLS[-1] == "T70"
