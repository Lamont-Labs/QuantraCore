"""
Intraday Moonshot Predictor

Uses the trained intraday model to predict moonshot candidates
from real-time or recent 1-minute bar data.
"""

import logging
import gzip
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.quantracore_apex.data.intraday_features import IntradayFeatureExtractor
from src.quantracore_apex.data.intraday_pipeline import AlphaVantageIntradayFetcher
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow

logger = logging.getLogger(__name__)


@dataclass
class MoonshotPrediction:
    """Prediction result for a symbol."""
    symbol: str
    probability: float
    confidence_tier: str
    is_candidate: bool
    top_features: Dict[str, float]
    timestamp: datetime
    bars_analyzed: int


class IntradayMoonshotPredictor:
    """
    Predicts moonshot candidates using the trained intraday model.
    
    Uses Alpha Vantage 1-minute data to make predictions on individual stocks.
    """
    
    def __init__(
        self,
        model_path: str = "models/intraday_moonshot_v1.pkl.gz",
        probability_threshold: float = 0.6,
    ):
        self.model_path = model_path
        self.probability_threshold = probability_threshold
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_extractor = IntradayFeatureExtractor()
        self.data_fetcher = AlphaVantageIntradayFetcher()
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with gzip.open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']
        
        logger.info(f"Loaded intraday model from {self.model_path}")
        logger.info(f"Training samples: {self.model_data['training_samples']:,}")
        logger.info(f"Model precision: {self.model_data['metrics']['precision']:.3f}")
    
    def predict_from_bars(
        self,
        symbol: str,
        bars: List[OhlcvBar],
    ) -> Optional[MoonshotPrediction]:
        """
        Make prediction from a list of OhlcvBar objects.
        
        Args:
            symbol: Stock symbol
            bars: List of 1-minute bars (minimum 100)
        
        Returns:
            MoonshotPrediction or None if insufficient data
        """
        if len(bars) < 100:
            logger.warning(f"{symbol}: Need at least 100 bars, got {len(bars)}")
            return None
        
        window = OhlcvWindow(
            symbol=symbol,
            timeframe="1min",
            bars=bars[-100:],
        )
        
        features = self.feature_extractor.extract(window)
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        if probability >= 0.8:
            confidence_tier = "HIGH"
        elif probability >= 0.6:
            confidence_tier = "MEDIUM"
        elif probability >= 0.4:
            confidence_tier = "LOW"
        else:
            confidence_tier = "NONE"
        
        is_candidate = probability >= self.probability_threshold
        
        feature_importance = self.model_data['feature_importance']
        top_features = dict(list(feature_importance.items())[:5])
        
        return MoonshotPrediction(
            symbol=symbol,
            probability=float(probability),
            confidence_tier=confidence_tier,
            is_candidate=is_candidate,
            top_features=top_features,
            timestamp=datetime.now(),
            bars_analyzed=len(bars),
        )
    
    def predict_from_dataframe(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Optional[MoonshotPrediction]:
        """
        Make prediction from a DataFrame with OHLCV columns.
        
        Expected columns: timestamp, open, high, low, close, volume
        """
        if len(df) < 100:
            logger.warning(f"{symbol}: Need at least 100 bars, got {len(df)}")
            return None
        
        bars = []
        for _, row in df.tail(100).iterrows():
            ts = row['timestamp']
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            
            bar = OhlcvBar(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            )
            bars.append(bar)
        
        return self.predict_from_bars(symbol, bars)
    
    def predict_live(
        self,
        symbol: str,
    ) -> Optional[MoonshotPrediction]:
        """
        Fetch live 1-minute data and make prediction.
        
        Uses Alpha Vantage to fetch recent intraday data.
        """
        df = self.data_fetcher.fetch_intraday(symbol, outputsize="compact")
        
        if df is None or len(df) < 100:
            logger.warning(f"{symbol}: Insufficient data from Alpha Vantage")
            return None
        
        return self.predict_from_dataframe(symbol, df)
    
    def scan_symbols(
        self,
        symbols: List[str],
        use_cache: bool = True,
    ) -> List[MoonshotPrediction]:
        """
        Scan multiple symbols for moonshot candidates.
        
        Args:
            symbols: List of symbols to scan
            use_cache: Whether to use cached data
        
        Returns:
            List of predictions sorted by probability
        """
        predictions = []
        
        for symbol in symbols:
            try:
                if use_cache:
                    df = self.data_fetcher.load_cached(symbol)
                else:
                    df = None
                
                if df is None or len(df) < 100:
                    df = self.data_fetcher.fetch_intraday(symbol, outputsize="compact")
                
                if df is None or len(df) < 100:
                    logger.debug(f"{symbol}: Skipping - insufficient data")
                    continue
                
                prediction = self.predict_from_dataframe(symbol, df)
                if prediction:
                    predictions.append(prediction)
                    
                    if prediction.is_candidate:
                        logger.info(
                            f"{symbol}: MOONSHOT CANDIDATE "
                            f"({prediction.probability:.1%} probability, "
                            f"{prediction.confidence_tier} confidence)"
                        )
            except Exception as e:
                logger.error(f"{symbol}: Prediction failed - {e}")
                continue
        
        predictions.sort(key=lambda p: p.probability, reverse=True)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "training_samples": self.model_data['training_samples'],
            "positive_samples": self.model_data['positive_samples'],
            "precision": self.model_data['metrics']['precision'],
            "recall": self.model_data['metrics']['recall'],
            "f1_score": self.model_data['metrics']['f1_score'],
            "precision_at_60": self.model_data['metrics'].get('precision_at_0.6', 0),
            "precision_at_70": self.model_data['metrics'].get('precision_at_0.7', 0),
            "precision_at_80": self.model_data['metrics'].get('precision_at_0.8', 0),
            "timestamp": self.model_data['timestamp'],
            "feature_count": len(self.feature_names),
            "top_features": list(self.model_data['feature_importance'].items())[:10],
        }


def get_intraday_predictor(
    model_path: str = "models/intraday_moonshot_v1.pkl.gz",
    probability_threshold: float = 0.6,
) -> IntradayMoonshotPredictor:
    """
    Get an instance of the intraday moonshot predictor.
    
    Args:
        model_path: Path to the trained model
        probability_threshold: Minimum probability to flag as candidate
    
    Returns:
        Configured IntradayMoonshotPredictor
    """
    return IntradayMoonshotPredictor(
        model_path=model_path,
        probability_threshold=probability_threshold,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    predictor = get_intraday_predictor()
    
    print("\n" + "=" * 60)
    print("INTRADAY MOONSHOT PREDICTOR")
    print("=" * 60)
    
    info = predictor.get_model_info()
    print(f"\nModel Info:")
    print(f"  Training samples: {info['training_samples']:,}")
    print(f"  Precision: {info['precision']:.3f}")
    print(f"  Precision @ 70%: {info['precision_at_70']:.3f}")
    
    print(f"\nTop Features:")
    for name, importance in info['top_features'][:5]:
        print(f"  {name}: {importance:.4f}")
    
    test_symbols = ["AAPL", "TSLA", "SPY"]
    print(f"\nScanning symbols: {test_symbols}")
    
    predictions = predictor.scan_symbols(test_symbols)
    
    print(f"\nResults:")
    for p in predictions:
        status = "CANDIDATE" if p.is_candidate else "---"
        print(f"  {p.symbol}: {p.probability:.1%} ({p.confidence_tier}) {status}")
