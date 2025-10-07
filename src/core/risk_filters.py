"""
ASP-01..ASP-20 fail-closed checks (demo).
"""
ASP = [f"ASP-{i:02}" for i in range(1, 21)]

def run_filters(score: float):
    status = "pass" if score >= 0 else "block"
    return [{"id": f, "result": status} for f in ASP]
