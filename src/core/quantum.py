"""Quantum stack stubs Q1..Q7."""
QUANTUM = [f"Q{i}" for i in range(1, 8)]
def list_quantum():
    return [{"id": qid, "status": "stub"} for qid in QUANTUM]
