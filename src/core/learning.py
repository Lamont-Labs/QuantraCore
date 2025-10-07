"""Learning protocol stubs L1..L20."""
LEARNING = [f"L{i}" for i in range(1, 21)]
def list_learning():
    return [{"id": lid, "status": "stub"} for lid in LEARNING]
