"""Protocol layer table T01..T70 (names only; demo)."""
PROTOCOLS = [f"T{i:02}" for i in range(1, 71)]
DESCRIPTIONS = {p: "Deterministic protocol stub" for p in PROTOCOLS}
