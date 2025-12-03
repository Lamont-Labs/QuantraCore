#!/usr/bin/env python3
"""
FRESH SAMPLE FETCHER - Add more stocks to training universe via Alpaca (FREE)

Uses Alpaca's free historical data API to fetch EOD data for new stocks.
No rate limits like Polygon! Can fetch 100+ stocks quickly.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

ALPACA_DATA_URL = 'https://data.alpaca.markets'
CACHE_DIR = Path("data/cache/polygon/day")

headers = {
    'APCA-API-KEY-ID': os.environ.get('ALPACA_PAPER_API_KEY', ''),
    'APCA-API-SECRET-KEY': os.environ.get('ALPACA_PAPER_API_SECRET', ''),
}

NEW_VOLATILE_STOCKS = [
    'QUBT', 'IONQ', 'RGTI', 'KULR', 'QBTS',
    'RKLB', 'LUNR', 'RDW', 'ASTR', 'ASTS', 'SPCE', 'JOBY',
    'ACHR', 'EVTL', 'LILM', 'BLDE', 'EVEX',
    'GOEV', 'NKLA', 'FFIE', 'MULN', 'FSR', 'PSNY', 'LCID', 'RIVN',
    'CHPT', 'BLNK', 'EVGO',
    'PLUG', 'FCEL', 'BE', 'BLDP', 'HYLN', 'NOVA', 'RUN',
    'LAZR', 'VLDR', 'OUST', 'AEVA', 'LIDR', 'INVZ', 'CPTN',
    'UPST', 'AFRM', 'LMND', 'ROOT', 'OPEN', 'OPFI', 'SFT',
    'ORGN', 'ARRY', 'ENPH', 'SEDG', 'FSLR', 'MAXN', 'JKS',
    'U', 'DKNG', 'PENN', 'DRAFT', 'RSI', 'GENI',
    'RBLX', 'MTTR', 'DM', 'PRNT', 'SSYS', 'DDD',
    'PATH', 'AI', 'BBAI', 'BIGC', 'SQSP', 'DOCS', 'DOCN',
    'SNOW', 'MDB', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA',
    'CFLT', 'GTLB', 'FROG', 'SUMO', 'ESTC', 'NEWR',
    'XPEV', 'LI', 'NIO', 'BYDDY', 'GGPI',
    'JD', 'PDD', 'BIDU', 'BILI', 'TME', 'TAL', 'EDU', 'GOTU',
    'BABA', 'KWEB', 'BEKE', 'DIDI', 'FUTU', 'TIGR',
    'OCGN', 'SRNE', 'IBRX', 'AGEN', 'CERE', 'AKRO', 'CRSP',
    'BEAM', 'EDIT', 'NTLA', 'VERV', 'FATE', 'EXAI',
    'CGC', 'ACB', 'SNDL', 'HEXO', 'OGI', 'VFF', 'CRON', 'CURLF',
    'SPWR', 'CSIQ', 'SHLS', 'STEM', 'DCFC',
    'CLSK', 'BITF', 'HUT', 'HIVE', 'BTBT', 'ANY',
    'EBON', 'CAN', 'SOS', 'BTCS', 'GREE', 'CORZ',
    'SOXL', 'FNGU', 'TQQQ', 'UPRO', 'LABU',
    'DWAC', 'PHUN', 'MARK', 'BKKT', 'GBTC',
    'IRNT', 'OPAD', 'TMC', 'PTRA', 'VLD',
    'DNA', 'ME', 'TALK', 'GENI', 'SEAT', 'BROS',
    'HIMS', 'HNST', 'BARK', 'CHWY', 'LOVE', 'PETS',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
    'GRAB', 'SE', 'MELI', 'NU', 'STNE', 'PAGS',
    'COIN', 'HOOD', 'SOFI', 'LC', 'ALLY',
    'RKT', 'UWMC', 'GHLD',
    'NCLH', 'CCL', 'RCL', 'LUV', 'DAL', 'UAL', 'AAL',
    'ABNB', 'EXPE', 'BKNG', 'TCOM',
    'ZM', 'DOCU', 'PTON', 'FVRR', 'UPWK', 'ETSY',
    'BMBL', 'MTCH', 'PINS', 'SNAP', 'TWTR',
    'TTD', 'ROKU', 'FUBO', 'PARA', 'WBD',
    'GME', 'BBBY', 'EXPR', 'KOSS', 'NAKD',
    'WKHS', 'RIDE', 'SOLO', 'AYRO', 'IDEX',
    'CEMI', 'VERB', 'PROG', 'ATER', 'SDC',
    'FAZE', 'NILE', 'IMPP', 'ENDP', 'MNMD',
    'PSFE', 'PAYO', 'FLYW', 'BILL', 'PYPL',
    'SQ', 'SHOP', 'MKTX', 'FOUR',
]

EXISTING_STOCKS = set()


def load_existing_cache():
    """Load list of already cached stocks."""
    global EXISTING_STOCKS
    parquets = list(CACHE_DIR.glob("*.parquet"))
    EXISTING_STOCKS = {p.stem for p in parquets}
    print(f"Found {len(EXISTING_STOCKS)} existing cached stocks")
    return EXISTING_STOCKS


def fetch_bars_batch(symbols: list, years: int = 3) -> dict:
    """Fetch historical bars for multiple symbols from Alpaca."""
    all_bars = {}
    batch_size = 30
    
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
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
            response = requests.get(url, headers=headers, params=params, timeout=60)
            data = response.json()
            
            if 'bars' in data:
                all_bars.update(data['bars'])
            
            next_token = data.get('next_page_token')
            while next_token:
                params['page_token'] = next_token
                response = requests.get(url, headers=headers, params=params, timeout=60)
                data = response.json()
                if 'bars' in data:
                    for sym, bars in data['bars'].items():
                        if sym in all_bars:
                            all_bars[sym].extend(bars)
                        else:
                            all_bars[sym] = bars
                next_token = data.get('next_page_token')
                
        except Exception as e:
            print(f"  Error fetching batch {i//batch_size + 1}: {e}")
        
        print(f"  Fetched batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
        time.sleep(0.5)
    
    return all_bars


def save_to_cache(symbol: str, bars: list) -> bool:
    """Save stock data to parquet cache."""
    if len(bars) < 90:
        return False
    
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
    
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    
    if len(df) < 90:
        return False
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}.parquet"
    df.to_parquet(path, index=False)
    
    return True


def main():
    print("=" * 70)
    print("FRESH SAMPLE FETCHER - Expand Training Universe")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n[1/3] Loading existing cache...")
    existing = load_existing_cache()
    
    new_stocks = [s for s in NEW_VOLATILE_STOCKS if s not in existing]
    new_stocks = list(dict.fromkeys(new_stocks))
    
    print(f"\n[2/3] Fetching {len(new_stocks)} NEW stocks from Alpaca...")
    print(f"  (Skipping {len(NEW_VOLATILE_STOCKS) - len(new_stocks)} already cached)")
    
    if not new_stocks:
        print("\n  All stocks already cached!")
        return
    
    all_bars = fetch_bars_batch(new_stocks, years=3)
    
    print(f"\n[3/3] Saving to cache...")
    saved = 0
    skipped = 0
    
    for symbol, bars in all_bars.items():
        if save_to_cache(symbol, bars):
            saved += 1
            print(f"  Saved {symbol}: {len(bars)} days")
        else:
            skipped += 1
    
    final_count = len(list(CACHE_DIR.glob("*.parquet")))
    
    print("\n" + "=" * 70)
    print("FETCH COMPLETE!")
    print("=" * 70)
    print(f"\n  New stocks saved: {saved}")
    print(f"  Skipped (insufficient data): {skipped}")
    print(f"  Total stocks in cache: {final_count}")
    print(f"\n  Run: python scripts/incremental_train.py")
    print(f"  To retrain with expanded universe!")
    print("=" * 70)


if __name__ == '__main__':
    main()
