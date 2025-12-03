#!/usr/bin/env python3
"""
EXPAND STOCK UNIVERSE - Add 50-100 more volatile stocks for training

High-value buckets:
1. Biotech/Pharma trial runners
2. EV/Battery/Energy metals
3. Crypto miners/exchanges
4. China/EM tech ADRs
5. Recent IPOs/SPACs
6. Meme/retail sentiment rotation
7. Momentum mid-caps anchor set
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BIOTECH_PHARMA = [
    "MRNA", "BNTX", "NVAX", "VXRT", "INO", "OCGN", "SRNE", "ATOS",
    "SAVA", "DVAX", "IMVT", "CRSP", "NTLA", "BEAM", "EDIT", "VERV",
    "ARCT", "PRTA", "FATE", "CRISPR"
]

EV_BATTERY_METALS = [
    "LAC", "LTHM", "ALB", "SQM", "ENVX", "QS", "MVST", "DCRC",
    "CHPT", "BLNK", "EVGO", "PLUG", "FCEL", "BE", "BLDP", "HYLN",
    "GOEV", "FSR", "WKHS", "NKLA"
]

CRYPTO_MINERS_EXCHANGES = [
    "HUT", "BITF", "CLSK", "HIVE", "BTBT", "CAN", "EBON", "DMGI",
    "BITO", "GBTC", "ETHE", "BTCS", "SI", "SBNY"
]

CHINA_EM_TECH_ADRS = [
    "BABA", "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI",
    "TME", "DIDI", "GRAB", "SE", "MELI", "NU", "STNE", "PAGS"
]

IPO_SPAC_RECENT = [
    "SOFI", "DKNG", "PLTR", "SNOW", "AI", "RBLX", "U", "ABNB",
    "DASH", "COIN", "AFRM", "UPST", "IONQ", "JOBY", "LILM", "EVTL"
]

MEME_ROTATION = [
    "BBBY", "KOSS", "EXPR", "NAKD", "CENN", "MULN", "APRN", "ATER",
    "PROG", "PHUN", "DWAC", "BENE", "BGFV", "REV", "NEGG", "CARV"
]

MOMENTUM_ANCHORS = [
    "SQ", "ROKU", "SHOP", "TTD", "NET", "CRWD", "ZS", "DDOG",
    "OKTA", "PANW", "SNOW", "MNDY", "CFLT", "S", "MDB", "GTLB"
]

def get_all_new_tickers() -> List[str]:
    """Get deduplicated list of all new tickers to fetch."""
    all_tickers = (
        BIOTECH_PHARMA + 
        EV_BATTERY_METALS + 
        CRYPTO_MINERS_EXCHANGES + 
        CHINA_EM_TECH_ADRS + 
        IPO_SPAC_RECENT + 
        MEME_ROTATION + 
        MOMENTUM_ANCHORS
    )
    
    existing = set(p.stem for p in CACHE_DIR.glob("*.parquet"))
    logger.info(f"Existing cached stocks: {len(existing)}")
    
    new_tickers = [t for t in all_tickers if t not in existing]
    new_tickers = list(dict.fromkeys(new_tickers))
    
    logger.info(f"New tickers to fetch: {len(new_tickers)}")
    return new_tickers

def fetch_stock_data(symbol: str, api_key: str) -> bool:
    """Fetch EOD data for a single stock from Polygon."""
    import pandas as pd
    import httpx
    
    cache_path = CACHE_DIR / f"{symbol}.parquet"
    if cache_path.exists():
        logger.info(f"  {symbol}: Already cached, skipping")
        return True
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url, params={"apiKey": api_key, "limit": 500})
            
            if response.status_code == 429:
                logger.warning(f"  {symbol}: Rate limited, waiting...")
                return False
            
            if response.status_code != 200:
                logger.warning(f"  {symbol}: API error {response.status_code}")
                return False
            
            data = response.json()
            
            if data.get("resultsCount", 0) == 0:
                logger.warning(f"  {symbol}: No data returned")
                return False
            
            results = data.get("results", [])
            
            if len(results) < 90:
                logger.warning(f"  {symbol}: Only {len(results)} bars, need 90+")
                return False
            
            df = pd.DataFrame(results)
            df = df.rename(columns={
                "o": "open",
                "h": "high", 
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "timestamp"
            })
            
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("date").reset_index(drop=True)
            
            df.to_parquet(cache_path)
            logger.info(f"  {symbol}: Saved {len(df)} bars")
            return True
            
    except Exception as e:
        logger.error(f"  {symbol}: Error - {e}")
        return False

def main():
    print("=" * 70)
    print("EXPAND STOCK UNIVERSE - Fetch 50-100 Additional Volatile Stocks")
    print("=" * 70)
    
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set")
        return
    
    new_tickers = get_all_new_tickers()
    
    if not new_tickers:
        print("No new tickers to fetch - universe is already expanded!")
        return
    
    print(f"\nFetching {len(new_tickers)} new stocks...")
    print("Categories:")
    print(f"  - Biotech/Pharma: {len([t for t in BIOTECH_PHARMA if t in new_tickers])}")
    print(f"  - EV/Battery: {len([t for t in EV_BATTERY_METALS if t in new_tickers])}")
    print(f"  - Crypto miners: {len([t for t in CRYPTO_MINERS_EXCHANGES if t in new_tickers])}")
    print(f"  - China/EM ADRs: {len([t for t in CHINA_EM_TECH_ADRS if t in new_tickers])}")
    print(f"  - IPO/SPAC: {len([t for t in IPO_SPAC_RECENT if t in new_tickers])}")
    print(f"  - Meme rotation: {len([t for t in MEME_ROTATION if t in new_tickers])}")
    print(f"  - Momentum anchors: {len([t for t in MOMENTUM_ANCHORS if t in new_tickers])}")
    
    success_count = 0
    failed = []
    rate_limited = []
    
    for i, symbol in enumerate(new_tickers):
        logger.info(f"[{i+1}/{len(new_tickers)}] Fetching {symbol}...")
        
        result = fetch_stock_data(symbol, api_key)
        
        if result:
            success_count += 1
        elif result is False:
            if "rate" in str(result).lower():
                rate_limited.append(symbol)
            else:
                failed.append(symbol)
        
        time.sleep(0.3)
        
        if (i + 1) % 5 == 0:
            time.sleep(1)
    
    print("\n" + "=" * 70)
    print("UNIVERSE EXPANSION COMPLETE")
    print("=" * 70)
    
    total_cached = len(list(CACHE_DIR.glob("*.parquet")))
    print(f"\nResults:")
    print(f"  - New stocks fetched: {success_count}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Total cached stocks: {total_cached}")
    
    if failed:
        print(f"\n  Failed tickers: {', '.join(failed[:20])}")
    
    print("\nNext steps:")
    print("  1. Run: python scripts/strict_moonshot_validation.py")
    print("  2. Compare stock-split precision to 65.1% baseline")
    print("=" * 70)

if __name__ == "__main__":
    main()
