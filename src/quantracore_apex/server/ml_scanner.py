"""
ML Scanner API - Live trading signals from trained models.

Provides endpoints for:
- Runner detection (5%+ gains)
- Mega runner detection (10%+ gains)
- Moonshot detection (50%+/100%+ gains)
- Portfolio tracking
- Auto position sizing
"""

import os
import io
import gzip
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import joblib
import psycopg2

logger = logging.getLogger(__name__)

_model_cache: Dict[str, Any] = {}


def load_models_from_database() -> Dict[str, Any]:
    """Load ML models from PostgreSQL database."""
    global _model_cache
    
    if _model_cache:
        return _model_cache
    
    try:
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            logger.warning("DATABASE_URL not set, trying disk models")
            return load_models_from_disk()
        
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        cur.execute("SELECT model_name, model_data, metrics FROM ml_models")
        rows = cur.fetchall()
        
        for name, data, metrics in rows:
            decompressed = gzip.decompress(bytes(data))
            buffer = io.BytesIO(decompressed)
            models = joblib.load(buffer)
            _model_cache[name] = {
                'models': models,
                'metrics': metrics,
                'loaded_at': datetime.now()
            }
            logger.info(f"Loaded model {name} from database")
        
        cur.close()
        conn.close()
        return _model_cache
    except Exception as e:
        logger.error(f"Failed to load from database: {e}")
        return load_models_from_disk()


def load_models_from_disk() -> Dict[str, Any]:
    """Fallback: Load models from disk."""
    global _model_cache
    
    model_paths = {
        'apex_production': 'models/production/apex_ensemble.pkl',
        'mega_runners': 'models/mega_runners/mega_ensemble.pkl',
        'moonshots': 'models/moonshots/moonshot_ensemble.pkl',
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models = joblib.load(path)
                _model_cache[name] = {
                    'models': models,
                    'metrics': models.get('metrics', {}),
                    'loaded_at': datetime.now()
                }
                logger.info(f"Loaded model {name} from disk: {path}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
    
    return _model_cache


def get_model(name: str) -> Optional[Dict]:
    """Get a specific model by name."""
    if not _model_cache:
        load_models_from_database()
    return _model_cache.get(name)


@dataclass
class ScanSignal:
    symbol: str
    confidence: float
    model_type: str
    price: float
    week_change: float
    month_change: float
    volume_surge: float
    features_used: int


def scan_for_runners(symbols: List[str], model_type: str = 'apex_production') -> List[Dict]:
    """
    Scan symbols for runner candidates using trained ML models.
    
    Args:
        symbols: List of stock symbols to scan
        model_type: 'apex_production', 'mega_runners', or 'moonshots'
    
    Returns:
        List of signals sorted by confidence
    """
    from src.quantracore_apex.apexlab.features import SwingFeatureExtractor
    from src.quantracore_apex.data_layer.adapters.alpaca_data_adapter import AlpacaDataAdapter
    from src.quantracore_apex.core.schemas import OhlcvWindow
    
    model_data = get_model(model_type)
    if not model_data:
        raise ValueError(f"Model {model_type} not loaded")
    
    models = model_data['models']
    
    if model_type == 'apex_production':
        model = models.get('runner_detector')
    elif model_type == 'mega_runners':
        model = models.get('runner_10')
    elif model_type == 'moonshots':
        model = models.get('model_50')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model is None:
        raise ValueError(f"Model not found in {model_type}")
    
    extractor = SwingFeatureExtractor(enable_enrichment=False)
    alpaca = AlpacaDataAdapter()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    
    signals = []
    
    for symbol in symbols:
        try:
            bars = alpaca.fetch_ohlcv(symbol, start_date, end_date, '1d')
            if not bars or len(bars) < 30:
                continue
            
            window_bars = bars[-30:]
            window = OhlcvWindow(symbol=symbol, timeframe='day', bars=window_bars)
            
            features = extractor.extract(window)
            
            closes = np.array([b.close for b in window_bars])
            volumes = np.array([b.volume for b in window_bars])
            
            mom_5d = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0
            mom_10d = (closes[-1] - closes[-10]) / closes[-10] if len(closes) > 10 and closes[-10] > 0 else 0
            vol_surge = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
            
            if model_type == 'moonshots':
                augmented = np.append(features, [mom_5d, mom_10d, vol_surge])
            else:
                augmented = np.append(features, [mom_5d])
            
            augmented = np.nan_to_num(augmented, nan=0, posinf=0, neginf=0)
            
            proba = model.predict_proba([augmented])[0][1]
            
            current_price = closes[-1]
            week_change = mom_5d * 100
            month_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
            
            signals.append({
                'symbol': symbol,
                'confidence': float(proba),
                'model_type': model_type,
                'price': float(current_price),
                'week_change': float(week_change),
                'month_change': float(month_change),
                'volume_surge': float(vol_surge),
                'features_used': len(features),
            })
        except Exception as e:
            logger.debug(f"Failed to scan {symbol}: {e}")
            continue
    
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


def scan_with_intraday_model(symbols: List[str], probability_threshold: float = 0.6) -> List[Dict]:
    """
    Scan symbols using the trained intraday moonshot model.
    
    Uses 1-minute bar patterns to detect stocks ready for significant moves.
    Requires Alpha Vantage API for live data.
    
    Args:
        symbols: List of stock symbols to scan
        probability_threshold: Minimum probability to flag as candidate
    
    Returns:
        List of predictions sorted by probability
    """
    try:
        from src.quantracore_apex.ml.intraday_predictor import get_intraday_predictor
    except ImportError as e:
        logger.error(f"Failed to import intraday predictor: {e}")
        return []
    
    try:
        predictor = get_intraday_predictor(probability_threshold=probability_threshold)
    except FileNotFoundError:
        logger.warning("Intraday model not found - skipping intraday scan")
        return []
    
    predictions = predictor.scan_symbols(symbols, use_cache=True)
    
    results = []
    for p in predictions:
        results.append({
            'symbol': p.symbol,
            'confidence': p.probability,
            'model_type': 'intraday_moonshot',
            'confidence_tier': p.confidence_tier,
            'is_candidate': p.is_candidate,
            'bars_analyzed': p.bars_analyzed,
            'timestamp': p.timestamp.isoformat(),
        })
    
    return results


def combined_moonshot_scan(
    symbols: List[str],
    include_eod: bool = True,
    include_intraday: bool = True,
) -> Dict[str, Any]:
    """
    Combine EOD and intraday model signals for more robust moonshot detection.
    
    This function runs both models and combines their signals:
    - EOD model: Daily patterns, fundamental/technical convergence
    - Intraday model: 1-minute microstructure patterns
    
    A symbol flagged by BOTH models has higher conviction.
    
    Args:
        symbols: List of stock symbols to scan
        include_eod: Include EOD moonshot model
        include_intraday: Include intraday model
    
    Returns:
        Combined analysis with both model outputs
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbols_scanned': len(symbols),
        'eod_signals': [],
        'intraday_signals': [],
        'combined_candidates': [],
        'high_conviction': [],
    }
    
    eod_candidates = set()
    intraday_candidates = set()
    
    if include_eod:
        try:
            eod_signals = scan_for_runners(symbols, model_type='moonshots')
            results['eod_signals'] = eod_signals
            eod_candidates = {s['symbol'] for s in eod_signals if s['confidence'] >= 0.6}
        except Exception as e:
            logger.error(f"EOD scan failed: {e}")
    
    if include_intraday:
        try:
            intraday_signals = scan_with_intraday_model(symbols, probability_threshold=0.5)
            results['intraday_signals'] = intraday_signals
            intraday_candidates = {s['symbol'] for s in intraday_signals if s['is_candidate']}
        except Exception as e:
            logger.error(f"Intraday scan failed: {e}")
    
    all_candidates = eod_candidates | intraday_candidates
    high_conviction = eod_candidates & intraday_candidates
    
    for symbol in all_candidates:
        eod_conf = next((s['confidence'] for s in results['eod_signals'] if s['symbol'] == symbol), 0)
        intraday_conf = next((s['confidence'] for s in results['intraday_signals'] if s['symbol'] == symbol), 0)
        
        combined_score = (eod_conf * 0.6 + intraday_conf * 0.4) if eod_conf and intraday_conf else max(eod_conf, intraday_conf)
        
        results['combined_candidates'].append({
            'symbol': symbol,
            'eod_confidence': eod_conf,
            'intraday_confidence': intraday_conf,
            'combined_score': combined_score,
            'signal_sources': ['eod'] * (symbol in eod_candidates) + ['intraday'] * (symbol in intraday_candidates),
            'high_conviction': symbol in high_conviction,
        })
    
    results['combined_candidates'].sort(key=lambda x: x['combined_score'], reverse=True)
    results['high_conviction'] = [c for c in results['combined_candidates'] if c['high_conviction']]
    
    return results


def get_alpaca_positions() -> List[Dict]:
    """Get current positions from Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        
        api_key = os.environ.get('ALPACA_PAPER_API_KEY')
        api_secret = os.environ.get('ALPACA_PAPER_API_SECRET')
        
        if not api_key or not api_secret:
            return []
        
        client = TradingClient(api_key, api_secret, paper=True)
        positions = client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc) * 100,
                'side': 'long' if float(pos.qty) > 0 else 'short',
            })
        
        return result
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return []


def get_alpaca_account() -> Dict:
    """Get Alpaca account info."""
    try:
        from alpaca.trading.client import TradingClient
        
        api_key = os.environ.get('ALPACA_PAPER_API_KEY')
        api_secret = os.environ.get('ALPACA_PAPER_API_SECRET')
        
        if not api_key or not api_secret:
            return {}
        
        client = TradingClient(api_key, api_secret, paper=True)
        account = client.get_account()
        
        return {
            'account_number': account.account_number,
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'last_equity': float(account.last_equity),
            'day_trade_count': account.daytrade_count,
            'status': str(account.status),
        }
    except Exception as e:
        logger.error(f"Failed to get account: {e}")
        return {}


def calculate_position_size(
    symbol: str,
    confidence: float,
    account_value: float,
    max_position_pct: float = 0.05,
    confidence_scaling: bool = True
) -> Dict:
    """
    Calculate position size based on confidence and risk management.
    
    Args:
        symbol: Stock symbol
        confidence: Model confidence (0-1)
        account_value: Total account value
        max_position_pct: Maximum position as % of account (default 5%)
        confidence_scaling: Scale position by confidence
    
    Returns:
        Position sizing recommendation
    """
    from src.quantracore_apex.data_layer.adapters.alpaca_data_adapter import AlpacaDataAdapter
    
    alpaca = AlpacaDataAdapter()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    
    try:
        bars = alpaca.fetch_ohlcv(symbol, start_date, end_date, '1d')
        if not bars:
            return {'error': 'No price data'}
        
        current_price = bars[-1].close
        
        base_position = account_value * max_position_pct
        
        if confidence_scaling:
            if confidence > 0.8:
                scale = 1.0
            elif confidence > 0.6:
                scale = 0.75
            elif confidence > 0.4:
                scale = 0.5
            else:
                scale = 0.25
            position_value = base_position * scale
        else:
            position_value = base_position
        
        shares = int(position_value / current_price)
        actual_value = shares * current_price
        
        return {
            'symbol': symbol,
            'confidence': confidence,
            'current_price': current_price,
            'recommended_shares': shares,
            'position_value': actual_value,
            'position_pct': (actual_value / account_value) * 100,
            'confidence_scale': scale if confidence_scaling else 1.0,
        }
    except Exception as e:
        return {'error': str(e)}


QUICK_UNIVERSE = [
    # Quantum/AI - Hottest sector 2025
    'QUBT', 'RGTI', 'QBTS', 'IONQ', 'BBAI', 'SOUN',
    # Crypto miners
    'MARA', 'RIOT', 'BITF', 'CLSK', 'COIN',
    # High-momentum small-caps
    'SMCI', 'SOFI', 'PLTR', 'FUBO', 'HIMS',
]

RUNNER_UNIVERSE = [
    # Quantum/AI plays
    'QUBT', 'RGTI', 'QBTS', 'IONQ', 'BBAI', 'SOUN', 'POET',
    # Crypto miners - move with BTC
    'MARA', 'RIOT', 'BITF', 'CLSK', 'HIVE', 'COIN',
    # High short interest squeeze candidates (20%+ SI)
    'BYND', 'LCID', 'HIMS', 'NVTS', 'SYM', 'MVIS', 'OCGN',
    # EV/Clean energy volatility plays
    'PLUG', 'FCEL', 'QS', 'BLNK', 'CHPT', 'NIO', 'RIVN', 'XPEV',
    # Tech momentum
    'SMCI', 'SOFI', 'PLTR', 'FUBO', 'OPEN', 'UPST', 'HOOD', 'AFRM',
    # Biotech with catalysts
    'SAVA', 'VXRT', 'INO', 'MRNA', 'BNTX',
    # Cannabis
    'TLRY', 'SNDL', 'CGC',
    # Retail favorites with squeeze potential
    'GME', 'AMC', 'CLOV',
]

MOONSHOT_UNIVERSE = [
    # TIER 1: Quantum/AI - Highest volatility, 1000%+ runners in 2024-2025
    'QUBT', 'RGTI', 'QBTS', 'IONQ', 'BBAI', 'SOUN', 'POET', 'SERV',
    # TIER 2: Crypto miners - Move 50%+ with Bitcoin rallies
    'MARA', 'RIOT', 'BITF', 'CLSK', 'HIVE', 'COIN',
    # TIER 3: High short interest (20-40% SI) - Squeeze candidates
    'BYND', 'LCID', 'HIMS', 'NVTS', 'SYM', 'MVIS', 'OCGN', 'FCEL',
    # TIER 4: EV/Battery/Clean energy - Catalyst-driven moves
    'PLUG', 'QS', 'BLNK', 'CHPT', 'NIO', 'RIVN', 'XPEV',
    # TIER 5: Tech momentum with breakout potential
    'SMCI', 'SOFI', 'PLTR', 'FUBO', 'OPEN', 'UPST', 'HOOD', 'AFRM',
    # TIER 6: Biotech with binary catalysts
    'SAVA', 'VXRT', 'INO', 'MRNA', 'BNTX',
    # TIER 7: Cannabis sector
    'TLRY', 'SNDL', 'CGC',
    # TIER 8: Retail favorites - Proven squeeze history
    'GME', 'AMC', 'CLOV',
]


def get_realtime_status() -> Dict[str, Any]:
    """
    Get real-time scanner status and upgrade info.
    
    Returns current mode (EOD vs real-time) and what's
    unlocked with Algo Trader Plus subscription.
    """
    from src.quantracore_apex.server.realtime_scanner import get_realtime_scanner
    
    scanner = get_realtime_scanner()
    return scanner.get_status()


def get_trading_modes() -> Dict[str, Any]:
    """
    Show available trading modes based on data subscription.
    
    Free tier: EOD only → Swing trades
    Algo Trader Plus: Real-time → All trading types
    """
    realtime_enabled = os.getenv("ALPACA_REALTIME_ENABLED", "false").lower() in ("true", "1", "yes")
    
    return {
        "current_tier": "Algo Trader Plus ($99/mo)" if realtime_enabled else "Free (EOD)",
        "data_refresh": "Real-time streaming" if realtime_enabled else "Once per day (EOD)",
        "trading_modes": {
            "swing_trading": {
                "enabled": True,
                "description": "2-10 day holds",
                "data_required": "EOD (free)",
                "status": "ACTIVE"
            },
            "position_trading": {
                "enabled": True,
                "description": "Weeks to months",
                "data_required": "EOD (free)",
                "status": "ACTIVE"
            },
            "day_trading": {
                "enabled": realtime_enabled,
                "description": "Buy morning, sell afternoon",
                "data_required": "Real-time ($99/mo)",
                "status": "ACTIVE" if realtime_enabled else "UPGRADE REQUIRED"
            },
            "scalping": {
                "enabled": realtime_enabled,
                "description": "1-5 minute trades",
                "data_required": "Real-time ($99/mo)",
                "status": "ACTIVE" if realtime_enabled else "UPGRADE REQUIRED"
            },
            "intraday_swing": {
                "enabled": realtime_enabled,
                "description": "Multiple trades per day",
                "data_required": "Real-time ($99/mo)",
                "status": "ACTIVE" if realtime_enabled else "UPGRADE REQUIRED"
            }
        },
        "features": {
            "ml_scanner": {"status": "ACTIVE", "refresh": "3-30 seconds"},
            "portfolio_tracking": {"status": "ACTIVE", "refresh": "5 seconds"},
            "breakout_alerts": {
                "status": "ACTIVE" if realtime_enabled else "LIMITED",
                "refresh": "instant" if realtime_enabled else "EOD"
            },
            "live_streaming": {
                "status": "ACTIVE" if realtime_enabled else "DISABLED",
                "refresh": "sub-second" if realtime_enabled else "N/A"
            }
        },
        "upgrade_path": {
            "current_cost": "$0/month",
            "upgrade_cost": "$99/month",
            "upgrade_url": "https://app.alpaca.markets/brokerage/dashboard/overview",
            "or_free_with": "$100,000+ account balance (Alpaca Elite tier)",
            "what_you_get": [
                "Full SIP real-time data",
                "100% US market coverage",
                "Unlimited API requests",
                "WebSocket streaming",
                "Day trading enabled",
                "Scalping enabled",
                "Instant breakout alerts"
            ]
        }
    }
