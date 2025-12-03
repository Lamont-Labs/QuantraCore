#!/usr/bin/env python3
"""
Hot Runner Scanner - Find stocks ready to pop in 1-2 DAYS
Looks for imminent breakout setups with extreme compression
"""

import os
import sys
import gzip
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

ALPACA_PAPER_URL = 'https://paper-api.alpaca.markets'
ALPACA_DATA_URL = 'https://data.alpaca.markets'
MODELS_DIR = Path('data/models')

UNIVERSE = [
    'AMC', 'GME', 'BBAI', 'SOUN', 'PLTR', 'MARA', 'RIOT', 'COIN', 'HOOD',
    'SOFI', 'LCID', 'RIVN', 'TSLA', 'NVDA', 'AMD', 'SMCI', 'QUBT', 'IONQ',
    'RGTI', 'KULR', 'RKLB', 'LUNR', 'RDW', 'ASTR', 'ASTS', 'SPCE', 'JOBY',
    'ACHR', 'EVTL', 'LILM', 'WKHS', 'GOEV', 'NKLA', 'FFIE', 'MULN', 'FSR',
    'CLOV', 'WISH', 'BB', 'NOK', 'SNAP', 'PINS', 'RBLX', 'U', 'DKNG', 'PENN',
    'CHPT', 'BLNK', 'EVGO', 'PLUG', 'FCEL', 'BE', 'BLDP', 'HYLN',
    'TLRY', 'CGC', 'ACB', 'SNDL', 'HEXO', 'OGI', 'VFF', 'CRON',
    'LAZR', 'VLDR', 'OUST', 'AEVA', 'LIDR', 'INVZ', 'CPTN',
    'UPST', 'AFRM', 'LMND', 'ROOT', 'OPEN', 'OPFI', 'SFT',
    'ORGN', 'ARRY', 'ENPH', 'SEDG', 'FSLR', 'RUN', 'NOVA',
    'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'SRNE',
    'JD', 'PDD', 'BIDU', 'BILI', 'TME', 'TAL', 'EDU', 'XPEV', 'LI', 'NIO',
]

SECTOR_MAP = {
    'AAPL': 'tech', 'AMD': 'tech', 'NVDA': 'tech', 'TSLA': 'auto',
    'BBAI': 'tech', 'CLOV': 'health', 'MARA': 'crypto', 'RIOT': 'crypto',
    'PLTR': 'tech', 'SOFI': 'fintech', 'LCID': 'auto', 'RIVN': 'auto',
    'GME': 'retail', 'AMC': 'entertainment', 'BB': 'tech', 'NOK': 'tech',
    'MRNA': 'biotech', 'BNTX': 'biotech', 'NVAX': 'biotech', 'VXRT': 'biotech',
    'INO': 'biotech', 'JD': 'china_tech', 'PDD': 'china_tech', 'BIDU': 'china_tech',
    'BILI': 'china_tech', 'TME': 'china_tech', 'NET': 'tech', 'MSFT': 'tech',
    'TLRY': 'cannabis', 'CGC': 'cannabis', 'SNDL': 'cannabis',
    'WKHS': 'auto', 'SMCI': 'tech', 'SOUN': 'tech', 'QUBT': 'quantum',
    'IONQ': 'quantum', 'RGTI': 'quantum',
}

headers = {
    'APCA-API-KEY-ID': os.environ.get('ALPACA_PAPER_API_KEY', ''),
    'APCA-API-SECRET-KEY': os.environ.get('ALPACA_PAPER_API_SECRET', ''),
}


def get_stock_embedding(symbol):
    sector = SECTOR_MAP.get(symbol, 'unknown')
    sector_list = ['tech', 'crypto', 'health', 'fintech', 'auto', 'retail', 
                   'cannabis', 'entertainment', 'aerospace', 'biotech', 'china_tech', 'quantum', 'unknown']
    vec = [1.0 if s == sector else 0.0 for s in sector_list]
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) / (16**8)
    return {f'sector_{i}': float(v) for i, v in enumerate(vec)} | {
        'symbol_hash': float(sym_hash),
        'is_meme': float(symbol in {'GME', 'AMC', 'BB', 'BBAI', 'CLOV', 'WISH'}),
        'is_crypto_related': float(symbol in {'MARA', 'RIOT', 'COIN', 'HOOD'}),
        'is_ev': float(symbol in {'TSLA', 'LCID', 'RIVN', 'WKHS', 'GOEV', 'NKLA'}),
        'is_biotech': float(symbol in {'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN'}),
        'is_china_tech': float(symbol in {'JD', 'PDD', 'BIDU', 'BILI', 'TME', 'NIO', 'XPEV', 'LI'}),
    }


def extract_features(df, symbol):
    if len(df) < 61:
        return {}
    
    idx = len(df) - 1
    window = df.iloc[max(0, idx-60):idx+1]
    
    close = window['close'].values
    high = window['high'].values
    low = window['low'].values
    volume = window['volume'].values
    open_p = window['open'].values if 'open' in window.columns else close
    
    if close[-1] <= 0:
        return {}
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    f = {}
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = np.mean(close[-p:]) if len(close) >= p else np.mean(close)
        f[f'vol_sma_{p}'] = np.mean(volume[-p:]) if len(volume) >= p else np.mean(volume)
    
    for p in [5, 10, 20, 30]:
        r_window = returns[-p:] if len(returns) >= p else returns
        f[f'volatility_{p}'] = np.std(r_window) if len(r_window) > 0 else 0
    
    f['vol_compression_5_20'] = f['volatility_5'] / (f['volatility_20'] + 0.001)
    f['vol_compression_10_30'] = f['volatility_10'] / (f['volatility_30'] + 0.001)
    f['vol_compression_5_30'] = f['volatility_5'] / (f['volatility_30'] + 0.001)
    
    for p in [3, 5, 10, 20]:
        if len(close) > p:
            f[f'momentum_{p}'] = (close[-1] - close[-p-1]) / (close[-p-1] + 0.001)
        else:
            f[f'momentum_{p}'] = 0
    
    f['mom_acceleration'] = f['momentum_3'] - f['momentum_5']
    f['mom_inflection'] = f['momentum_5'] - f['momentum_10']
    f['mom_acceleration_2'] = (f['momentum_3'] - f['momentum_5']) - (f['momentum_5'] - f['momentum_10'])
    
    for p in [5, 10, 20]:
        recent_vol = np.mean(volume[-3:]) if len(volume) >= 3 else volume[-1]
        avg_vol = np.mean(volume[-p:]) if len(volume) >= p else np.mean(volume)
        f[f'vol_ratio_{p}'] = recent_vol / (avg_vol + 1)
    f['vol_acceleration'] = f['vol_ratio_5'] / (f['vol_ratio_10'] + 0.001)
    
    vol_5d_mean = np.mean(volume[-5:]) if len(volume) >= 5 else np.mean(volume)
    vol_20d_mean = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    f['vol_dryup_5d'] = 1.0 if vol_5d_mean < vol_20d_mean * 0.5 else 0.0
    f['vol_dryup_ratio'] = vol_5d_mean / (vol_20d_mean + 1)
    f['vol_spike_today'] = 1.0 if volume[-1] > vol_20d_mean * 2 else 0.0
    f['vol_spike_3d'] = 1.0 if vol_5d_mean > vol_20d_mean * 1.5 else 0.0
    
    for p in [10, 20, 50]:
        f[f'price_vs_sma_{p}'] = close[-1] / (f[f'sma_{p}'] + 0.001)
    
    f['sma_slope_5_10'] = (f['sma_5'] - f['sma_10']) / (f['sma_10'] + 0.001)
    f['sma_slope_10_20'] = (f['sma_10'] - f['sma_20']) / (f['sma_20'] + 0.001)
    f['sma_slope_20_50'] = (f['sma_20'] - f['sma_50']) / (f['sma_50'] + 0.001)
    f['sma_alignment'] = 1.0 if (f['sma_5'] > f['sma_10'] > f['sma_20']) else 0.0
    
    for p in [10, 20, 30]:
        h_window = high[-p:] if len(high) >= p else high
        l_window = low[-p:] if len(low) >= p else low
        r = np.max(h_window) - np.min(l_window)
        f[f'range_{p}'] = r / (close[-1] + 0.001)
    
    f['range_compression_10_20'] = f['range_10'] / (f['range_20'] + 0.001)
    f['range_compression_10_30'] = f['range_10'] / (f['range_30'] + 0.001)
    
    f['high_20'] = np.max(high[-20:]) if len(high) >= 20 else np.max(high)
    f['low_20'] = np.min(low[-20:]) if len(low) >= 20 else np.min(low)
    f['high_52w'] = np.max(high[-min(252, len(high)):]) if len(high) >= 50 else np.max(high)
    f['high_5'] = np.max(high[-5:]) if len(high) >= 5 else np.max(high)
    f['high_3'] = np.max(high[-3:]) if len(high) >= 3 else np.max(high)
    
    f['breakout_proximity'] = close[-1] / (f['high_20'] + 0.001)
    f['breakout_proximity_52w'] = close[-1] / (f['high_52w'] + 0.001)
    f['breakout_proximity_5d'] = close[-1] / (f['high_5'] + 0.001)
    f['breakout_proximity_3d'] = close[-1] / (f['high_3'] + 0.001)
    f['support_distance'] = (close[-1] - f['low_20']) / (f['low_20'] + 0.001)
    
    for i in range(-5, 0):
        if len(open_p) > abs(i) and len(close) > abs(i):
            gap = (open_p[i] - close[i-1]) / (close[i-1] + 0.001) if abs(i-1) < len(close) else 0
        else:
            gap = 0
        f[f'gap_day_{abs(i)}'] = gap
    
    if len(open_p) >= 2 and len(close) >= 2:
        f['gap_today'] = (open_p[-1] - close[-2]) / (close[-2] + 0.001)
    else:
        f['gap_today'] = 0
        
    gaps = []
    for i in range(-5, 0):
        if len(open_p) > abs(i) and abs(i-1) < len(close):
            gaps.append(abs((open_p[i] - close[i-1]) / (close[i-1] + 0.001)))
    f['gap_avg_5d'] = np.mean(gaps) if gaps else 0
    f['gap_up_count_5d'] = sum(1 for i in range(-5, 0) if len(open_p) > abs(i) and abs(i-1) < len(close) and open_p[i] > close[i-1])
    
    dr = high[-10:] - low[-10:] if len(high) >= 10 else high - low
    cp = (close[-10:] - low[-10:]) / (dr + 0.001) if len(close) >= 10 else (close - low) / (dr + 0.001)
    f['close_position_avg'] = np.mean(cp)
    f['close_position_today'] = cp[-1] if len(cp) > 0 else 0
    
    o_window = open_p[-10:] if len(open_p) >= 10 else open_p
    c_window = close[-10:] if len(close) >= 10 else close
    h_window = high[-10:] if len(high) >= 10 else high
    l_window = low[-10:] if len(low) >= 10 else low
    dr_w = h_window - l_window
    
    uw = (h_window - np.maximum(c_window, o_window)) / (dr_w + 0.001)
    lw = (np.minimum(c_window, o_window) - l_window) / (dr_w + 0.001)
    f['upper_wick_avg'] = np.mean(uw)
    f['lower_wick_avg'] = np.mean(lw)
    f['wick_shrinking'] = np.mean(uw[-3:]) - np.mean(uw[-7:-3]) if len(uw) >= 7 else 0
    f['wick_ratio'] = np.mean(uw) / (np.mean(lw) + 0.001)
    
    f['up_days_5'] = sum(1 for r in returns[-5:] if r > 0) if len(returns) >= 5 else 0
    f['up_days_10'] = sum(1 for r in returns[-10:] if r > 0) if len(returns) >= 10 else 0
    f['up_days_20'] = sum(1 for r in returns[-20:] if r > 0) if len(returns) >= 20 else 0
    f['tight_days_5'] = sum(1 for r in returns[-5:] if abs(r) < 0.02) if len(returns) >= 5 else 0
    f['tight_days_10'] = sum(1 for r in returns[-10:] if abs(r) < 0.02) if len(returns) >= 10 else 0
    
    f['consecutive_up'] = 0
    for r in reversed(returns[-10:]):
        if r > 0:
            f['consecutive_up'] += 1
        else:
            break
    
    f['consecutive_tight'] = 0
    for r in reversed(returns[-10:]):
        if abs(r) < 0.02:
            f['consecutive_tight'] += 1
        else:
            break
    
    dv = close[-20:] * volume[-20:] if len(close) >= 20 else close * volume
    f['dollar_vol_ratio'] = np.mean(dv[-3:]) / (np.mean(dv) + 1) if len(dv) >= 3 else 1
    f['dollar_vol_today'] = dv[-1] / (np.mean(dv) + 1) if len(dv) > 0 else 1
    
    gains = returns[-14:].copy() if len(returns) >= 14 else returns.copy()
    gains[gains < 0] = 0
    losses = -returns[-14:].copy() if len(returns) >= 14 else -returns.copy()
    losses[losses < 0] = 0
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
    f['rsi_14'] = 100 - (100 / (1 + avg_gain / (avg_loss + 0.0001)))
    
    if len(close) >= 26:
        ema12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
        f['macd'] = (ema12 - ema26) / (close[-1] + 0.001)
    else:
        f['macd'] = 0
    
    atr_vals = []
    for j in range(-min(14, len(high)-1), 0):
        if abs(j-1) < len(close):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_vals.append(tr)
    f['atr_14'] = np.mean(atr_vals) / (close[-1] + 0.001) if atr_vals else 0
    f['atr_compression'] = atr_vals[-1] / (np.mean(atr_vals) + 0.001) if len(atr_vals) > 1 else 1
    
    bb_close = close[-20:] if len(close) >= 20 else close
    bb_sma = np.mean(bb_close)
    bb_std = np.std(bb_close)
    f['bb_position'] = (close[-1] - bb_sma) / (2 * bb_std + 0.001)
    f['bb_width'] = (4 * bb_std) / (bb_sma + 0.001)
    
    f['current_price'] = close[-1]
    f['volume_today'] = volume[-1]
    f['is_penny'] = 1.0 if close[-1] < 5 else 0.0
    f['is_micro'] = 1.0 if close[-1] < 1 else 0.0
    
    stock_emb = get_stock_embedding(symbol)
    f.update(stock_emb)
    
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}


def calculate_imminence_score(features, base_prob):
    """Calculate how IMMINENT the breakout is (1-2 days vs 1-2 weeks)"""
    score = 0
    reasons = []
    
    vol_comp = features.get('vol_compression_5_20', 1)
    if vol_comp < 0.3:
        score += 30
        reasons.append(f"EXTREME compression ({vol_comp:.2f})")
    elif vol_comp < 0.5:
        score += 20
        reasons.append(f"Tight compression ({vol_comp:.2f})")
    elif vol_comp < 0.7:
        score += 10
        reasons.append(f"Moderate compression")
    
    bp_5d = features.get('breakout_proximity_5d', 0)
    bp_3d = features.get('breakout_proximity_3d', 0)
    if bp_3d >= 0.98:
        score += 25
        reasons.append(f"AT 3-day high ({bp_3d*100:.0f}%)")
    elif bp_5d >= 0.95:
        score += 20
        reasons.append(f"Near 5-day high ({bp_5d*100:.0f}%)")
    elif bp_5d >= 0.90:
        score += 10
        reasons.append(f"Approaching breakout")
    
    vol_spike = features.get('vol_spike_today', 0)
    vol_ratio = features.get('vol_ratio_5', 1)
    if vol_spike:
        score += 20
        reasons.append("Volume SPIKE today (2x avg)")
    elif vol_ratio > 1.5:
        score += 15
        reasons.append(f"Rising volume ({vol_ratio:.1f}x)")
    elif vol_ratio > 1.2:
        score += 5
        reasons.append("Volume building")
    
    mom_acc = features.get('mom_acceleration', 0)
    mom_3 = features.get('momentum_3', 0)
    if mom_acc > 0.03 and mom_3 > 0:
        score += 15
        reasons.append(f"Momentum ACCELERATING (+{mom_acc*100:.1f}%)")
    elif mom_3 > 0.02:
        score += 10
        reasons.append(f"Positive momentum")
    
    gap = features.get('gap_today', 0)
    if gap > 0.03:
        score += 15
        reasons.append(f"Gap UP today (+{gap*100:.1f}%)")
    elif gap > 0.01:
        score += 5
        reasons.append(f"Small gap up")
    
    consec_up = features.get('consecutive_up', 0)
    if consec_up >= 3:
        score += 10
        reasons.append(f"{consec_up} consecutive green days")
    
    close_pos = features.get('close_position_today', 0.5)
    if close_pos > 0.8:
        score += 10
        reasons.append("Closed near high of day")
    
    rsi = features.get('rsi_14', 50)
    if 40 < rsi < 60:
        score += 5
        reasons.append(f"RSI neutral ({rsi:.0f}) - room to run")
    
    return min(score, 100), reasons


def fetch_bars_batch(symbols, days=90):
    """Fetch historical bars for multiple symbols"""
    all_bars = {}
    batch_size = 50
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        end = datetime.now()
        start = end - timedelta(days=days)
        
        url = f'{ALPACA_DATA_URL}/v2/stocks/bars'
        params = {
            'symbols': ','.join(batch),
            'start': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'timeframe': '1Day',
            'adjustment': 'all',
            'limit': 10000,
            'feed': 'iex',
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()
            bars = data.get('bars', {})
            all_bars.update(bars)
        except Exception as e:
            print(f"  Error fetching batch: {e}")
    
    return all_bars


def main():
    print("=" * 80)
    print("HOT RUNNER SCANNER - Find 1-2 DAY breakouts")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    print("[1/4] Loading moonshot model...")
    model_path = MODELS_DIR / 'moonshot_incremental_v2.pkl.gz'
    with gzip.open(model_path, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    feature_names = data.get('feature_names', [])
    print(f"  Model loaded with {len(feature_names)} features")
    
    print("\n[2/4] Fetching EOD data for universe...")
    bars_data = fetch_bars_batch(UNIVERSE)
    print(f"  Got data for {len(bars_data)} stocks")
    
    print("\n[3/4] Scanning for IMMINENT breakouts...")
    
    results = []
    for symbol in UNIVERSE:
        bars = bars_data.get(symbol, [])
        if not bars or len(bars) < 61:
            continue
        
        rows = []
        for bar in bars:
            rows.append({
                'timestamp': bar['t'],
                'open': float(bar['o']),
                'high': float(bar['h']),
                'low': float(bar['l']),
                'close': float(bar['c']),
                'volume': float(bar['v']),
            })
        df = pd.DataFrame(rows).sort_values('timestamp')
        
        features = extract_features(df, symbol)
        if not features:
            continue
        
        X = np.array([[features.get(f, 0) for f in feature_names]])
        base_prob = model.predict_proba(X)[0, 1]
        
        imminence_score, reasons = calculate_imminence_score(features, base_prob)
        
        combined_score = (base_prob * 0.4 + imminence_score / 100 * 0.6)
        
        results.append({
            'symbol': symbol,
            'base_prob': base_prob,
            'imminence': imminence_score,
            'combined': combined_score,
            'price': features.get('current_price', 0),
            'reasons': reasons,
            'features': features,
        })
    
    results.sort(key=lambda x: x['imminence'], reverse=True)
    
    print("\n[4/4] Results - HOT RUNNERS (ready in 1-2 days)")
    print("=" * 80)
    
    hot_runners = [r for r in results if r['imminence'] >= 50]
    
    if hot_runners:
        print(f"\n{'Symbol':<8} {'Imminence':<12} {'Moonshot%':<12} {'Price':<10} {'Signals'}")
        print("-" * 80)
        
        for r in hot_runners[:15]:
            signals = ' | '.join(r['reasons'][:3])
            print(f"{r['symbol']:<8} {r['imminence']:<12} {r['base_prob']*100:.1f}%{'':<6} ${r['price']:<9.2f} {signals}")
    else:
        print("\nNo hot runners found with 50+ imminence score.")
        print("Showing top 10 by imminence anyway:\n")
        print(f"{'Symbol':<8} {'Imminence':<12} {'Moonshot%':<12} {'Price':<10} {'Signals'}")
        print("-" * 80)
        
        for r in results[:10]:
            signals = ' | '.join(r['reasons'][:3]) if r['reasons'] else 'Building...'
            print(f"{r['symbol']:<8} {r['imminence']:<12} {r['base_prob']*100:.1f}%{'':<6} ${r['price']:<9.2f} {signals}")
    
    print("\n" + "=" * 80)
    print("IMMINENCE SCORE BREAKDOWN:")
    print("-" * 50)
    print("  80-100 = Could break out TODAY or TOMORROW")
    print("  60-79  = Ready within 1-2 DAYS")
    print("  40-59  = Setup forming, 2-3 days")
    print("  20-39  = Still building, 3-5 days")
    print("  <20    = Not imminent")
    print("=" * 80)
    
    print("\nDETAILED ANALYSIS - TOP 5 IMMINENT:")
    print("-" * 80)
    
    for r in results[:5]:
        f = r['features']
        print(f"\n{r['symbol']} - Imminence: {r['imminence']}/100 | Moonshot: {r['base_prob']*100:.1f}%")
        print(f"  Price: ${r['price']:.2f}")
        print(f"  Vol Compression (5/20): {f.get('vol_compression_5_20', 0):.2f} {'(TIGHT!)' if f.get('vol_compression_5_20', 1) < 0.5 else ''}")
        print(f"  Near 3-day High: {f.get('breakout_proximity_3d', 0)*100:.1f}%")
        print(f"  Near 5-day High: {f.get('breakout_proximity_5d', 0)*100:.1f}%")
        print(f"  Volume Ratio: {f.get('vol_ratio_5', 1):.2f}x {'(SPIKE!)' if f.get('vol_spike_today', 0) else ''}")
        print(f"  Momentum (3d): {f.get('momentum_3', 0)*100:+.1f}%")
        print(f"  Gap Today: {f.get('gap_today', 0)*100:+.2f}%")
        print(f"  RSI: {f.get('rsi_14', 50):.0f}")
        print(f"  Signals: {', '.join(r['reasons']) if r['reasons'] else 'Building...'}")
    
    return results


if __name__ == '__main__':
    main()
