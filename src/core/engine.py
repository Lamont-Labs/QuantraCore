"""
QuantraScore — seeded deterministic scoring.
"""
import numpy as np
from hashlib import sha256

def generate_signal(ticker: str, seed: int = 42):
    np.random.seed(seed)
    score = round(float(np.random.random()) * 100, 2)
    h = sha256(f"{ticker}|{seed}|{score}".encode()).hexdigest()
    return {"ticker": ticker, "seed": seed, "score": score, "hash": h}
