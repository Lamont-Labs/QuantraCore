"""ZDE — Zero Drawdown Entry decision (demo)."""
def zero_drawdown_entry(score: float, threshold: float = 50.0) -> bool:
    return score >= threshold
