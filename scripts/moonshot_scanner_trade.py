#!/usr/bin/env python3
"""
MOONSHOT SCANNER + AUTO TRADER

Scans all stocks in our universe using the latest EOD data,
scores them with the moonshot model, and opens paper positions
for the top runners.
"""

import os
import sys
import gzip
import pickle
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
CACHE_DIR = Path("data/cache/polygon/day")

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

STOCK_UNIVERSE = [
    "AAPL", "AMD", "NVDA", "TSLA", "PLTR", "SOFI", "LCID", "RIVN",
    "GME", "AMC", "BB", "NOK", "BBAI", "CLOV", "MARA", "RIOT",
    "COIN", "HOOD", "UPST", "AFRM", "SAVA", "PLUG", "FCEL", "BLDP",
    "SNAP", "PINS", "DKNG", "PENN", "SPCE", "RKLB", "JOBY", "LILM",
    "LAZR", "VLDR", "OUST", "LIDR", "GOEV", "NKLA", "FSR", "FFIE",
    "WKHS", "RIDE", "HYLN", "XL", "CLNE", "BLNK", "CHPT", "EVGO",
    "DNA", "GINKGO", "QS", "MVST", "DCRC", "SES", "SLDP", "FREYR",
    "ORGN", "STEM", "ENVX", "RUN", "SEDG", "ENPH", "FSLR", "ARRY",
    "NOVA", "SHLS", "CSIQ", "JKS", "DQ", "MAXN", "SPWR", "SUNW",
    "WOLF", "ON", "CREE", "AEHR", "ACHR", "IONQ", "RGTI", "QUBT",
    "KULR", "SMCI", "APP", "MSTR",
    "MRNA", "BNTX", "NVAX", "VXRT", "INO",
    "JD", "PDD", "BIDU", "BILI", "TME",
]

SECTOR_MAP = {
    "AAPL": "tech", "AMD": "tech", "NVDA": "tech", "TSLA": "auto",
    "BBAI": "tech", "CLOV": "health", "MARA": "crypto", "RIOT": "crypto",
    "PLTR": "tech", "SOFI": "fintech", "LCID": "auto", "RIVN": "auto",
    "GME": "retail", "AMC": "entertainment", "BB": "tech", "NOK": "tech",
    "MRNA": "biotech", "BNTX": "biotech", "NVAX": "biotech", "VXRT": "biotech",
    "INO": "biotech", "JD": "china_tech", "PDD": "china_tech", "BIDU": "china_tech",
    "BILI": "china_tech", "TME": "china_tech", "BABA": "china_tech",
    "COIN": "crypto", "HOOD": "fintech", "UPST": "fintech", "AFRM": "fintech",
    "PLUG": "energy", "FCEL": "energy", "ENPH": "energy", "FSLR": "energy",
    "IONQ": "quantum", "RGTI": "quantum", "QUBT": "quantum",
    "SMCI": "tech", "APP": "tech", "MSTR": "crypto",
}


def get_alpaca_headers():
    api_key = os.environ.get("ALPACA_PAPER_API_KEY", "")
    api_secret = os.environ.get("ALPACA_PAPER_API_SECRET", "")
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json",
    }


def get_stock_embedding(symbol: str) -> Dict[str, float]:
    sector = SECTOR_MAP.get(symbol, "unknown")
    sector_list = ["tech", "crypto", "health", "fintech", "auto", "retail", 
                   "cannabis", "entertainment", "aerospace", "biotech", "china_tech", "unknown"]
    vec = [1.0 if s == sector else 0.0 for s in sector_list]
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) / (16**8)
    return {f"sector_{i}": float(v) for i, v in enumerate(vec)} | {
        "symbol_hash": float(sym_hash),
        "is_meme": float(symbol in {"GME", "AMC", "BB", "BBAI", "CLOV", "WISH"}),
        "is_crypto_related": float(symbol in {"MARA", "RIOT", "COIN", "HOOD"}),
        "is_ev": float(symbol in {"TSLA", "LCID", "RIVN"}),
        "is_biotech": float(symbol in {"MRNA", "BNTX", "NVAX", "VXRT", "INO"}),
        "is_china_tech": float(symbol in {"JD", "PDD", "BIDU", "BILI", "TME", "BABA"}),
    }


def extract_features(df: pd.DataFrame, symbol: str) -> Dict[str, float]:
    """Extract features from the most recent data point (for live scoring)."""
    if len(df) < 61:
        return {}
    
    idx = len(df) - 1
    window = df.iloc[max(0, idx-60):idx+1]
    
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    open_p = window["open"].values if "open" in window.columns else close
    
    if close[-1] <= 0:
        return {}
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    f = {}
    
    for p in [5, 10, 20, 50]:
        f[f"sma_{p}"] = np.mean(close[-p:]) if len(close) >= p else np.mean(close)
        f[f"vol_sma_{p}"] = np.mean(volume[-p:]) if len(volume) >= p else np.mean(volume)
    
    for p in [5, 10, 20, 30]:
        r_window = returns[-p:] if len(returns) >= p else returns
        f[f"volatility_{p}"] = np.std(r_window) if len(r_window) > 0 else 0
    
    f["vol_compression_5_20"] = f["volatility_5"] / (f["volatility_20"] + 0.001)
    f["vol_compression_10_30"] = f["volatility_10"] / (f["volatility_30"] + 0.001)
    f["vol_compression_5_30"] = f["volatility_5"] / (f["volatility_30"] + 0.001)
    
    for p in [3, 5, 10, 20]:
        if len(close) > p:
            f[f"momentum_{p}"] = (close[-1] - close[-p-1]) / (close[-p-1] + 0.001)
        else:
            f[f"momentum_{p}"] = 0
    
    f["mom_acceleration"] = f["momentum_3"] - f["momentum_5"]
    f["mom_inflection"] = f["momentum_5"] - f["momentum_10"]
    f["mom_acceleration_2"] = (f["momentum_3"] - f["momentum_5"]) - (f["momentum_5"] - f["momentum_10"])
    
    for p in [5, 10, 20]:
        recent_vol = np.mean(volume[-3:]) if len(volume) >= 3 else volume[-1]
        avg_vol = np.mean(volume[-p:]) if len(volume) >= p else np.mean(volume)
        f[f"vol_ratio_{p}"] = recent_vol / (avg_vol + 1)
    f["vol_acceleration"] = f["vol_ratio_5"] / (f["vol_ratio_10"] + 0.001)
    
    vol_5d_mean = np.mean(volume[-5:]) if len(volume) >= 5 else np.mean(volume)
    vol_20d_mean = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    f["vol_dryup_5d"] = 1.0 if vol_5d_mean < vol_20d_mean * 0.5 else 0.0
    f["vol_dryup_ratio"] = vol_5d_mean / (vol_20d_mean + 1)
    f["vol_spike_today"] = 1.0 if volume[-1] > vol_20d_mean * 2 else 0.0
    f["vol_spike_3d"] = 1.0 if vol_5d_mean > vol_20d_mean * 1.5 else 0.0
    
    for p in [10, 20, 50]:
        f[f"price_vs_sma_{p}"] = close[-1] / (f[f"sma_{p}"] + 0.001)
    
    f["sma_slope_5_10"] = (f["sma_5"] - f["sma_10"]) / (f["sma_10"] + 0.001)
    f["sma_slope_10_20"] = (f["sma_10"] - f["sma_20"]) / (f["sma_20"] + 0.001)
    f["sma_slope_20_50"] = (f["sma_20"] - f["sma_50"]) / (f["sma_50"] + 0.001)
    f["sma_alignment"] = 1.0 if (f["sma_5"] > f["sma_10"] > f["sma_20"]) else 0.0
    
    for p in [10, 20, 30]:
        h_window = high[-p:] if len(high) >= p else high
        l_window = low[-p:] if len(low) >= p else low
        r = np.max(h_window) - np.min(l_window)
        f[f"range_{p}"] = r / (close[-1] + 0.001)
    
    f["range_compression_10_20"] = f["range_10"] / (f["range_20"] + 0.001)
    f["range_compression_10_30"] = f["range_10"] / (f["range_30"] + 0.001)
    
    f["high_20"] = np.max(high[-20:]) if len(high) >= 20 else np.max(high)
    f["low_20"] = np.min(low[-20:]) if len(low) >= 20 else np.min(low)
    f["high_52w"] = np.max(high[-min(252, len(high)):]) if len(high) >= 50 else np.max(high)
    
    f["breakout_proximity"] = close[-1] / (f["high_20"] + 0.001)
    f["breakout_proximity_52w"] = close[-1] / (f["high_52w"] + 0.001)
    f["support_distance"] = (close[-1] - f["low_20"]) / (f["low_20"] + 0.001)
    
    for i in range(-5, 0):
        if len(open_p) > abs(i) and len(close) > abs(i):
            gap = (open_p[i] - close[i-1]) / (close[i-1] + 0.001) if abs(i-1) < len(close) else 0
        else:
            gap = 0
        f[f"gap_day_{abs(i)}"] = gap
    
    if len(open_p) >= 2 and len(close) >= 2:
        f["gap_today"] = (open_p[-1] - close[-2]) / (close[-2] + 0.001)
    else:
        f["gap_today"] = 0
        
    gaps = []
    for i in range(-5, 0):
        if len(open_p) > abs(i) and abs(i-1) < len(close):
            gaps.append(abs((open_p[i] - close[i-1]) / (close[i-1] + 0.001)))
    f["gap_avg_5d"] = np.mean(gaps) if gaps else 0
    f["gap_up_count_5d"] = sum(1 for i in range(-5, 0) if len(open_p) > abs(i) and abs(i-1) < len(close) and open_p[i] > close[i-1])
    
    dr = high[-10:] - low[-10:] if len(high) >= 10 else high - low
    cp = (close[-10:] - low[-10:]) / (dr + 0.001) if len(close) >= 10 else (close - low) / (dr + 0.001)
    f["close_position_avg"] = np.mean(cp)
    f["close_position_today"] = cp[-1] if len(cp) > 0 else 0
    
    o_window = open_p[-10:] if len(open_p) >= 10 else open_p
    c_window = close[-10:] if len(close) >= 10 else close
    h_window = high[-10:] if len(high) >= 10 else high
    l_window = low[-10:] if len(low) >= 10 else low
    dr_w = h_window - l_window
    
    uw = (h_window - np.maximum(c_window, o_window)) / (dr_w + 0.001)
    lw = (np.minimum(c_window, o_window) - l_window) / (dr_w + 0.001)
    f["upper_wick_avg"] = np.mean(uw)
    f["lower_wick_avg"] = np.mean(lw)
    f["wick_shrinking"] = np.mean(uw[-3:]) - np.mean(uw[-7:-3]) if len(uw) >= 7 else 0
    f["wick_ratio"] = np.mean(uw) / (np.mean(lw) + 0.001)
    
    f["up_days_5"] = sum(1 for r in returns[-5:] if r > 0) if len(returns) >= 5 else 0
    f["up_days_10"] = sum(1 for r in returns[-10:] if r > 0) if len(returns) >= 10 else 0
    f["up_days_20"] = sum(1 for r in returns[-20:] if r > 0) if len(returns) >= 20 else 0
    f["tight_days_5"] = sum(1 for r in returns[-5:] if abs(r) < 0.02) if len(returns) >= 5 else 0
    f["tight_days_10"] = sum(1 for r in returns[-10:] if abs(r) < 0.02) if len(returns) >= 10 else 0
    
    f["consecutive_up"] = 0
    for r in reversed(returns[-10:]):
        if r > 0:
            f["consecutive_up"] += 1
        else:
            break
    
    f["consecutive_tight"] = 0
    for r in reversed(returns[-10:]):
        if abs(r) < 0.02:
            f["consecutive_tight"] += 1
        else:
            break
    
    dv = close[-20:] * volume[-20:] if len(close) >= 20 else close * volume
    f["dollar_vol_ratio"] = np.mean(dv[-3:]) / (np.mean(dv) + 1) if len(dv) >= 3 else 1
    f["dollar_vol_today"] = dv[-1] / (np.mean(dv) + 1) if len(dv) > 0 else 1
    
    gains = returns[-14:].copy() if len(returns) >= 14 else returns.copy()
    gains[gains < 0] = 0
    losses = -returns[-14:].copy() if len(returns) >= 14 else -returns.copy()
    losses[losses < 0] = 0
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
    f["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 0.0001)))
    
    if len(close) >= 26:
        ema12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
        f["macd"] = (ema12 - ema26) / (close[-1] + 0.001)
    else:
        f["macd"] = 0
    
    atr_vals = []
    for j in range(-min(14, len(high)-1), 0):
        if abs(j-1) < len(close):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_vals.append(tr)
    f["atr_14"] = np.mean(atr_vals) / (close[-1] + 0.001) if atr_vals else 0
    f["atr_compression"] = atr_vals[-1] / (np.mean(atr_vals) + 0.001) if len(atr_vals) > 1 else 1
    
    bb_close = close[-20:] if len(close) >= 20 else close
    bb_sma = np.mean(bb_close)
    bb_std = np.std(bb_close)
    f["bb_position"] = (close[-1] - bb_sma) / (2 * bb_std + 0.001)
    f["bb_width"] = (4 * bb_std) / (bb_sma + 0.001)
    
    f["current_price"] = close[-1]
    f["volume_today"] = volume[-1]
    f["is_penny"] = 1.0 if close[-1] < 5 else 0.0
    f["is_micro"] = 1.0 if close[-1] < 1 else 0.0
    
    stock_emb = get_stock_embedding(symbol)
    f.update(stock_emb)
    
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}


def fetch_latest_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch last 90 days of bars from Alpaca for each symbol."""
    headers = get_alpaca_headers()
    
    end = datetime.now()
    start = end - timedelta(days=90)
    
    result = {}
    
    symbols_str = ",".join(symbols[:50])
    
    url = f"{ALPACA_DATA_URL}/v2/stocks/bars"
    params = {
        "symbols": symbols_str,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timeframe": "1Day",
        "adjustment": "all",
        "limit": 10000,
        "feed": "iex",
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        bars_by_symbol = data.get("bars", {})
        
        for symbol, bars in bars_by_symbol.items():
            rows = []
            for bar in bars:
                rows.append({
                    "timestamp": bar["t"],
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                })
            if rows:
                df = pd.DataFrame(rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                result[symbol] = df
                
        logger.info(f"Fetched data for {len(result)} symbols from Alpaca")
        
    except Exception as e:
        logger.error(f"Failed to fetch from Alpaca: {e}")
    
    if len(symbols) > 50:
        symbols_str2 = ",".join(symbols[50:])
        params["symbols"] = symbols_str2
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            bars_by_symbol = data.get("bars", {})
            
            for symbol, bars in bars_by_symbol.items():
                rows = []
                for bar in bars:
                    rows.append({
                        "timestamp": bar["t"],
                        "open": float(bar["o"]),
                        "high": float(bar["h"]),
                        "low": float(bar["l"]),
                        "close": float(bar["c"]),
                        "volume": float(bar["v"]),
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")
                    result[symbol] = df
                    
            logger.info(f"Total: {len(result)} symbols with data")
            
        except Exception as e:
            logger.error(f"Failed to fetch batch 2: {e}")
    
    return result


def load_model():
    """Load the best moonshot model."""
    model_path = MODELS_DIR / "moonshot_incremental_v2.pkl.gz"
    
    if not model_path.exists():
        model_path = MODELS_DIR / "moonshot_strict_v2.pkl.gz"
    
    if not model_path.exists():
        raise FileNotFoundError("No moonshot model found!")
    
    with gzip.open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    return data['model'], data.get('feature_names', [])


def score_stocks(bars_data: Dict[str, pd.DataFrame], model, feature_names: List[str]) -> List[Tuple[str, float, float]]:
    """Score all stocks and return sorted by probability."""
    scores = []
    
    for symbol, df in bars_data.items():
        try:
            features = extract_features(df, symbol)
            if not features:
                continue
            
            X = np.array([[features.get(f, 0) for f in feature_names]])
            prob = model.predict_proba(X)[0, 1]
            price = features.get("current_price", 0)
            
            scores.append((symbol, prob, price))
            
        except Exception as e:
            logger.warning(f"Failed to score {symbol}: {e}")
            continue
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_current_positions() -> Dict[str, float]:
    """Get current paper trading positions."""
    headers = get_alpaca_headers()
    
    try:
        response = requests.get(f"{ALPACA_PAPER_URL}/v2/positions", headers=headers, timeout=10)
        response.raise_for_status()
        
        positions = {}
        for pos in response.json():
            positions[pos["symbol"]] = float(pos["qty"])
        
        return positions
        
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return {}


def get_account_info() -> Dict:
    """Get account info."""
    headers = get_alpaca_headers()
    
    try:
        response = requests.get(f"{ALPACA_PAPER_URL}/v2/account", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get account: {e}")
        return {}


def place_order(symbol: str, qty: int, side: str = "buy") -> Dict:
    """Place a paper trading order."""
    headers = get_alpaca_headers()
    
    body = {
        "symbol": symbol.upper(),
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    
    try:
        response = requests.post(
            f"{ALPACA_PAPER_URL}/v2/orders",
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        logger.error(f"Failed to place order for {symbol}: {e}")
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("MOONSHOT SCANNER + AUTO TRADER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n[1/5] Loading moonshot model...")
    model, feature_names = load_model()
    print(f"  Model loaded with {len(feature_names)} features")
    
    print("\n[2/5] Fetching latest EOD data from Alpaca...")
    bars_data = fetch_latest_bars(STOCK_UNIVERSE)
    print(f"  Got data for {len(bars_data)} stocks")
    
    print("\n[3/5] Scoring all stocks with moonshot model...")
    scores = score_stocks(bars_data, model, feature_names)
    
    print("\n  TOP 10 MOONSHOT CANDIDATES:")
    print("  " + "-" * 50)
    print(f"  {'Rank':<5} {'Symbol':<8} {'Probability':<15} {'Price':<10}")
    print("  " + "-" * 50)
    
    for i, (symbol, prob, price) in enumerate(scores[:10], 1):
        prob_str = f"{prob:.1%}"
        print(f"  {i:<5} {symbol:<8} {prob_str:<15} ${price:.2f}")
    
    top_3 = scores[:3]
    
    print("\n[4/5] Checking account and existing positions...")
    account = get_account_info()
    positions = get_current_positions()
    
    buying_power = float(account.get("buying_power", 0))
    equity = float(account.get("equity", 0))
    
    print(f"  Account Equity: ${equity:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")
    print(f"  Existing Positions: {len(positions)}")
    
    position_size = min(1000, buying_power / 4)
    
    print("\n[5/5] Opening paper positions for TOP 3 runners...")
    print("  " + "-" * 60)
    
    orders_placed = []
    
    for symbol, prob, price in top_3:
        if symbol in positions:
            print(f"  [{symbol}] Already have position ({positions[symbol]:.0f} shares) - SKIP")
            continue
        
        if price <= 0:
            print(f"  [{symbol}] Invalid price - SKIP")
            continue
        
        shares = int(position_size / price)
        
        if shares < 1:
            print(f"  [{symbol}] Position too small ({shares} shares) - SKIP")
            continue
        
        print(f"  [{symbol}] Moonshot prob: {prob:.1%} | Price: ${price:.2f} | Buying {shares} shares...")
        
        result = place_order(symbol, shares, "buy")
        
        if "error" not in result:
            print(f"    ORDER PLACED: {result.get('id', 'N/A')[:8]}...")
            orders_placed.append({
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "probability": prob,
                "order_id": result.get("id"),
            })
        else:
            print(f"    FAILED: {result['error']}")
    
    print("\n" + "=" * 70)
    print("SCANNER COMPLETE")
    print("=" * 70)
    
    if orders_placed:
        print(f"\n  Orders Placed: {len(orders_placed)}")
        for o in orders_placed:
            print(f"    - {o['symbol']}: {o['shares']} shares @ ~${o['price']:.2f} (prob: {o['probability']:.1%})")
        
        total_value = sum(o['shares'] * o['price'] for o in orders_placed)
        print(f"\n  Total Value: ${total_value:,.2f}")
    else:
        print("\n  No new orders placed.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
