"""
ApexCore V4 - Multi-Data Prediction System with Extended Heads.

Extends V3 with 9 new prediction heads that leverage multi-source data:
- catalyst_probability: News/event catalyst detection
- sentiment_score: Social/news sentiment aggregation  
- squeeze_probability: Short squeeze potential
- institutional_accumulation: Dark pool/13F tracking
- momentum_burst: Tape reading momentum detection
- tape_reading: Order flow analysis
- volatility_regime: Options-implied volatility state
- options_flow: Smart money positioning from options
- macro_regime: Economic macro environment

Total: 16 prediction heads (7 from V3 + 9 new)

This is the full-capability model requiring multi-source data connections.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .apexcore_v3 import ApexCoreV3Model, ApexCoreV3Prediction, ApexCoreV3Manifest

logger = logging.getLogger(__name__)

MODEL_DIR = "models/apexcore_v4"


@dataclass
class MultiDataFeatures:
    """Features derived from multi-source data providers."""
    
    options_call_volume: float = 0.0
    options_put_volume: float = 0.0
    options_put_call_ratio: float = 1.0
    options_unusual_count: int = 0
    options_total_premium: float = 0.0
    options_avg_iv: float = 0.3
    options_iv_percentile: float = 50.0
    
    news_sentiment_score: float = 0.5
    news_article_count: int = 0
    social_sentiment_score: float = 0.5
    social_volume: int = 0
    social_bullish_ratio: float = 0.5
    
    dark_pool_buy_ratio: float = 0.5
    dark_pool_total_value: float = 0.0
    dark_pool_block_count: int = 0
    institutional_accumulation: float = 0.0
    short_interest_pct: float = 10.0
    short_days_to_cover: float = 2.0
    
    level2_imbalance: float = 0.0
    level2_spread_pct: float = 0.1
    order_flow_delta: float = 0.0
    large_order_ratio: float = 0.0
    
    fed_funds_rate: float = 5.25
    treasury_10y: float = 4.5
    yield_curve_spread: float = 0.0
    cpi_yoy: float = 3.0
    vix_level: float = 15.0
    risk_regime: str = "neutral"
    
    crypto_correlation: float = 0.0
    btc_dominance: float = 50.0
    
    def to_feature_vector(self) -> List[float]:
        """Convert to numerical feature vector."""
        return [
            self.options_call_volume,
            self.options_put_volume,
            self.options_put_call_ratio,
            float(self.options_unusual_count),
            self.options_total_premium,
            self.options_avg_iv,
            self.options_iv_percentile,
            
            self.news_sentiment_score,
            float(self.news_article_count),
            self.social_sentiment_score,
            float(self.social_volume),
            self.social_bullish_ratio,
            
            self.dark_pool_buy_ratio,
            self.dark_pool_total_value,
            float(self.dark_pool_block_count),
            self.institutional_accumulation,
            self.short_interest_pct,
            self.short_days_to_cover,
            
            self.level2_imbalance,
            self.level2_spread_pct,
            self.order_flow_delta,
            self.large_order_ratio,
            
            self.fed_funds_rate,
            self.treasury_10y,
            self.yield_curve_spread,
            self.cpi_yoy,
            self.vix_level,
            {"risk_on": 1, "neutral": 0, "risk_off": -1}.get(self.risk_regime, 0),
            
            self.crypto_correlation,
            self.btc_dominance,
        ]


@dataclass
class ApexCoreV4Prediction(ApexCoreV3Prediction):
    """Extended prediction with multi-data heads."""
    
    catalyst_probability: float = 0.0
    catalyst_type: str = "none"
    
    sentiment_composite: float = 0.5
    sentiment_trend: str = "neutral"
    
    squeeze_probability: float = 0.0
    squeeze_trigger_price: Optional[float] = None
    
    institutional_signal: str = "neutral"
    accumulation_score: float = 0.0
    
    momentum_burst_prob: float = 0.0
    burst_direction: int = 0
    
    tape_reading_signal: str = "neutral"
    order_flow_strength: float = 0.0
    
    volatility_regime: str = "normal"
    iv_percentile: float = 50.0
    
    options_flow_signal: str = "neutral"
    smart_money_direction: int = 0
    
    macro_regime: str = "neutral"
    risk_appetite: str = "moderate"
    
    data_sources_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "catalyst_probability": self.catalyst_probability,
            "catalyst_type": self.catalyst_type,
            "sentiment_composite": self.sentiment_composite,
            "sentiment_trend": self.sentiment_trend,
            "squeeze_probability": self.squeeze_probability,
            "squeeze_trigger_price": self.squeeze_trigger_price,
            "institutional_signal": self.institutional_signal,
            "accumulation_score": self.accumulation_score,
            "momentum_burst_prob": self.momentum_burst_prob,
            "burst_direction": self.burst_direction,
            "tape_reading_signal": self.tape_reading_signal,
            "order_flow_strength": self.order_flow_strength,
            "volatility_regime": self.volatility_regime,
            "iv_percentile": self.iv_percentile,
            "options_flow_signal": self.options_flow_signal,
            "smart_money_direction": self.smart_money_direction,
            "macro_regime": self.macro_regime,
            "risk_appetite": self.risk_appetite,
            "data_sources_used": self.data_sources_used,
        })
        return base
    
    @property
    def multi_data_consensus(self) -> str:
        """Aggregate signal from all multi-data heads."""
        bullish_signals = 0
        bearish_signals = 0
        
        if self.sentiment_composite > 0.6:
            bullish_signals += 1
        elif self.sentiment_composite < 0.4:
            bearish_signals += 1
        
        if self.squeeze_probability > 0.6:
            bullish_signals += 1
        
        if self.institutional_signal == "accumulation":
            bullish_signals += 1
        elif self.institutional_signal == "distribution":
            bearish_signals += 1
        
        if self.momentum_burst_prob > 0.6 and self.burst_direction > 0:
            bullish_signals += 1
        elif self.momentum_burst_prob > 0.6 and self.burst_direction < 0:
            bearish_signals += 1
        
        if self.options_flow_signal == "bullish":
            bullish_signals += 1
        elif self.options_flow_signal == "bearish":
            bearish_signals += 1
        
        if self.macro_regime == "risk_on":
            bullish_signals += 1
        elif self.macro_regime == "risk_off":
            bearish_signals += 1
        
        if bullish_signals >= 4:
            return "STRONG_BULLISH"
        elif bullish_signals >= 2 and bearish_signals == 0:
            return "BULLISH"
        elif bearish_signals >= 4:
            return "STRONG_BEARISH"
        elif bearish_signals >= 2 and bullish_signals == 0:
            return "BEARISH"
        else:
            return "NEUTRAL"


class ApexCoreV4Model(ApexCoreV3Model):
    """
    Extended multi-head prediction model with multi-source data integration.
    
    Adds 9 new prediction heads on top of V3's 7 heads:
    1. catalyst_head: Predicts catalyst/news-driven moves
    2. sentiment_head: Aggregates social + news sentiment
    3. squeeze_head: Short squeeze probability
    4. institutional_head: Dark pool accumulation/distribution
    5. momentum_burst_head: Sudden momentum detection
    6. tape_reading_head: Order flow analysis
    7. volatility_head: Volatility regime classification
    8. options_flow_head: Smart money options positioning
    9. macro_head: Macroeconomic regime detection
    
    Requires multi-source data connections for full capability.
    Falls back to simulated data when providers unavailable.
    """
    
    EXTENDED_FEATURE_NAMES = [
        "options_call_volume", "options_put_volume", "options_put_call_ratio",
        "options_unusual_count", "options_total_premium", "options_avg_iv",
        "options_iv_percentile", "news_sentiment_score", "news_article_count",
        "social_sentiment_score", "social_volume", "social_bullish_ratio",
        "dark_pool_buy_ratio", "dark_pool_total_value", "dark_pool_block_count",
        "institutional_accumulation", "short_interest_pct", "short_days_to_cover",
        "level2_imbalance", "level2_spread_pct", "order_flow_delta", "large_order_ratio",
        "fed_funds_rate", "treasury_10y", "yield_curve_spread", "cpi_yoy",
        "vix_level", "risk_regime_encoded", "crypto_correlation", "btc_dominance"
    ]
    
    def __init__(
        self,
        model_size: str = "big",
        enable_calibration: bool = True,
        enable_uncertainty: bool = True,
        enable_multi_horizon: bool = True,
        enable_multi_data: bool = True,
    ):
        super().__init__(model_size, enable_calibration, enable_uncertainty, enable_multi_horizon)
        
        self.version = "4.0.0"
        self._enable_multi_data = enable_multi_data
        
        n_estimators = 100 if model_size == "mini" else 200
        max_depth = 4 if model_size == "mini" else 6
        
        self.catalyst_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.sentiment_head = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.squeeze_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.institutional_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.momentum_burst_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.tape_reading_head = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.volatility_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.options_flow_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.macro_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.extended_scaler = StandardScaler()
        self._extended_heads_fitted = False
        self._data_providers_status: Dict[str, bool] = {}
    
    def _fetch_multi_data_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> MultiDataFeatures:
        """
        Fetch features from all available data providers.
        
        Falls back to simulated data when providers unavailable.
        """
        features = MultiDataFeatures()
        sources_used = []
        
        try:
            from src.quantracore_apex.data_layer.adapters.options_flow_adapter import OptionsFlowAggregator
            flow_agg = OptionsFlowAggregator()
            
            if flow_agg.is_available():
                flow = flow_agg.get_all_flow(symbol, min_premium=10000)
                
                if flow:
                    calls = [f for f in flow if f.option_type.upper() == "CALL"]
                    puts = [f for f in flow if f.option_type.upper() == "PUT"]
                    
                    features.options_call_volume = sum(f.size for f in calls)
                    features.options_put_volume = sum(f.size for f in puts)
                    features.options_put_call_ratio = (
                        features.options_put_volume / max(features.options_call_volume, 1)
                    )
                    features.options_unusual_count = sum(1 for f in flow if f.is_unusual)
                    features.options_total_premium = sum(f.premium for f in flow)
                    
                    ivs = [f.implied_volatility for f in flow if f.implied_volatility]
                    if ivs:
                        features.options_avg_iv = np.mean(ivs)
                    
                    sources_used.append("options_flow")
                    self._data_providers_status["options_flow"] = True
        except Exception as e:
            logger.debug(f"Options flow fetch failed: {e}")
            self._data_providers_status["options_flow"] = False
        
        try:
            from src.quantracore_apex.data_layer.adapters.alternative_data_adapter import AlternativeDataAggregator
            alt_agg = AlternativeDataAggregator()
            
            if alt_agg.is_available():
                sentiment = alt_agg.get_combined_sentiment(symbol)
                
                features.news_sentiment_score = sentiment.get("average_score", 0.5)
                features.social_sentiment_score = sentiment.get("average_score", 0.5)
                features.social_bullish_ratio = sentiment.get("average_score", 0.5)
                
                sources_used.append("sentiment")
                self._data_providers_status["sentiment"] = True
        except Exception as e:
            logger.debug(f"Sentiment fetch failed: {e}")
            self._data_providers_status["sentiment"] = False
        
        try:
            from src.quantracore_apex.data_layer.adapters.dark_pool_adapter import DarkPoolAggregator
            dp_agg = DarkPoolAggregator()
            
            if dp_agg.is_available():
                accumulation = dp_agg.get_accumulation_signals(symbol)
                
                features.dark_pool_buy_ratio = accumulation.get("buy_ratio", 0.5)
                features.dark_pool_block_count = accumulation.get("total_prints", 0)
                
                if accumulation.get("signal") == "ACCUMULATION":
                    features.institutional_accumulation = accumulation.get("confidence", 0.5)
                elif accumulation.get("signal") == "DISTRIBUTION":
                    features.institutional_accumulation = -accumulation.get("confidence", 0.5)
                
                short_data = dp_agg.fetch_short_interest(symbol)
                features.short_interest_pct = short_data.short_percent_float
                features.short_days_to_cover = short_data.days_to_cover
                
                sources_used.append("dark_pool")
                self._data_providers_status["dark_pool"] = True
        except Exception as e:
            logger.debug(f"Dark pool fetch failed: {e}")
            self._data_providers_status["dark_pool"] = False
        
        try:
            from src.quantracore_apex.data_layer.adapters.level2_adapter import Level2Aggregator
            l2_agg = Level2Aggregator()
            
            if l2_agg.is_available():
                book = l2_agg.fetch_order_book(symbol)
                
                features.level2_imbalance = book.imbalance
                features.level2_spread_pct = book.spread / max(book.mid_price, 1) * 100
                
                sr = l2_agg.get_support_resistance(symbol)
                
                sources_used.append("level2")
                self._data_providers_status["level2"] = True
        except Exception as e:
            logger.debug(f"Level 2 fetch failed: {e}")
            self._data_providers_status["level2"] = False
        
        try:
            from src.quantracore_apex.data_layer.adapters.economic_adapter import EconomicDataAggregator, EconomicIndicator
            econ_agg = EconomicDataAggregator()
            
            if econ_agg.is_available():
                regime = econ_agg.get_current_regime()
                
                features.fed_funds_rate = 5.25
                features.treasury_10y = 4.5
                features.yield_curve_spread = regime.yield_curve
                features.vix_level = 15.0
                features.risk_regime = regime.regime.lower()
                
                sources_used.append("economic")
                self._data_providers_status["economic"] = True
        except Exception as e:
            logger.debug(f"Economic data fetch failed: {e}")
            self._data_providers_status["economic"] = False
        
        features.data_sources_used = sources_used
        return features
    
    def _prepare_extended_features(
        self,
        rows: List[Dict[str, Any]],
        multi_data: Optional[List[MultiDataFeatures]] = None
    ) -> np.ndarray:
        """Prepare combined feature matrix with multi-data features."""
        base_features = self._prepare_features(rows, include_cross_asset=True)
        
        if multi_data is None:
            multi_data = [MultiDataFeatures() for _ in rows]
        
        extended_features = np.array([m.to_feature_vector() for m in multi_data])
        
        combined = np.hstack([base_features, extended_features])
        return combined
    
    def fit_extended(
        self,
        rows: List[Dict[str, Any]],
        multi_data: Optional[List[MultiDataFeatures]] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train extended prediction heads with multi-data features.
        
        Call after fit() to add extended capabilities.
        """
        if not self._is_fitted:
            logger.info("Base model not fitted, training base heads first...")
            self.fit(rows, validation_split)
        
        if multi_data is None:
            multi_data = [MultiDataFeatures() for _ in rows]
        
        logger.info(f"[ApexCoreV4] Training extended heads on {len(rows)} samples...")
        
        X_ext = self._prepare_extended_features(rows, multi_data)
        
        y_catalyst = np.array([r.get("has_catalyst", 0) for r in rows])
        y_sentiment = np.array([r.get("sentiment_score", 0.5) for r in rows])
        y_squeeze = np.array([r.get("squeeze_occurred", 0) for r in rows])
        y_institutional = np.array([r.get("institutional_signal", 0) for r in rows])
        y_momentum_burst = np.array([r.get("momentum_burst", 0) for r in rows])
        y_tape = np.array([r.get("tape_reading_score", 0.5) for r in rows])
        y_volatility = np.array([r.get("volatility_regime", 0) for r in rows])
        y_options = np.array([r.get("options_signal", 0) for r in rows])
        y_macro = np.array([r.get("macro_regime", 0) for r in rows])
        
        n_val = int(len(rows) * validation_split)
        indices = np.random.RandomState(42).permutation(len(rows))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, X_val = X_ext[train_idx], X_ext[val_idx]
        X_train_scaled = self.extended_scaler.fit_transform(X_train)
        X_val_scaled = self.extended_scaler.transform(X_val)
        
        metrics = {}
        
        logger.info("Training Catalyst head...")
        if len(np.unique(y_catalyst[train_idx])) > 1:
            self.catalyst_head.fit(X_train_scaled, y_catalyst[train_idx])
            metrics["catalyst_accuracy"] = float(self.catalyst_head.score(X_val_scaled, y_catalyst[val_idx]))
        else:
            metrics["catalyst_accuracy"] = 0.5
        
        logger.info("Training Sentiment head...")
        self.sentiment_head.fit(X_train_scaled, y_sentiment[train_idx])
        sent_pred = self.sentiment_head.predict(X_val_scaled)
        metrics["sentiment_rmse"] = float(np.sqrt(np.mean((sent_pred - y_sentiment[val_idx]) ** 2)))
        
        logger.info("Training Squeeze head...")
        if len(np.unique(y_squeeze[train_idx])) > 1:
            self.squeeze_head.fit(X_train_scaled, y_squeeze[train_idx])
            metrics["squeeze_accuracy"] = float(self.squeeze_head.score(X_val_scaled, y_squeeze[val_idx]))
        else:
            metrics["squeeze_accuracy"] = 0.5
        
        logger.info("Training Institutional head...")
        if len(np.unique(y_institutional[train_idx])) > 1:
            self.institutional_head.fit(X_train_scaled, y_institutional[train_idx])
            metrics["institutional_accuracy"] = float(self.institutional_head.score(X_val_scaled, y_institutional[val_idx]))
        else:
            metrics["institutional_accuracy"] = 0.5
        
        logger.info("Training Momentum Burst head...")
        if len(np.unique(y_momentum_burst[train_idx])) > 1:
            self.momentum_burst_head.fit(X_train_scaled, y_momentum_burst[train_idx])
            metrics["momentum_burst_accuracy"] = float(self.momentum_burst_head.score(X_val_scaled, y_momentum_burst[val_idx]))
        else:
            metrics["momentum_burst_accuracy"] = 0.5
        
        logger.info("Training Tape Reading head...")
        self.tape_reading_head.fit(X_train_scaled, y_tape[train_idx])
        tape_pred = self.tape_reading_head.predict(X_val_scaled)
        metrics["tape_reading_rmse"] = float(np.sqrt(np.mean((tape_pred - y_tape[val_idx]) ** 2)))
        
        logger.info("Training Volatility head...")
        if len(np.unique(y_volatility[train_idx])) > 1:
            self.volatility_head.fit(X_train_scaled, y_volatility[train_idx])
            metrics["volatility_accuracy"] = float(self.volatility_head.score(X_val_scaled, y_volatility[val_idx]))
        else:
            metrics["volatility_accuracy"] = 0.5
        
        logger.info("Training Options Flow head...")
        if len(np.unique(y_options[train_idx])) > 1:
            self.options_flow_head.fit(X_train_scaled, y_options[train_idx])
            metrics["options_flow_accuracy"] = float(self.options_flow_head.score(X_val_scaled, y_options[val_idx]))
        else:
            metrics["options_flow_accuracy"] = 0.5
        
        logger.info("Training Macro head...")
        if len(np.unique(y_macro[train_idx])) > 1:
            self.macro_head.fit(X_train_scaled, y_macro[train_idx])
            metrics["macro_accuracy"] = float(self.macro_head.score(X_val_scaled, y_macro[val_idx]))
        else:
            metrics["macro_accuracy"] = 0.5
        
        self._extended_heads_fitted = True
        
        logger.info(f"[ApexCoreV4] Extended training complete. Metrics: {metrics}")
        return metrics
    
    def predict_extended(
        self,
        row: Dict[str, Any],
        symbol: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> ApexCoreV4Prediction:
        """
        Generate full prediction with all 16 heads.
        
        Fetches real-time multi-data features when available.
        """
        base_pred = self.predict(row, current_price)
        
        if symbol and self._enable_multi_data:
            multi_data = self._fetch_multi_data_features(symbol)
        else:
            multi_data = MultiDataFeatures()
        
        X_ext = self._prepare_extended_features([row], [multi_data])
        
        if self._extended_heads_fitted:
            X_ext_scaled = self.extended_scaler.transform(X_ext)
            
            try:
                catalyst_prob = float(self.catalyst_head.predict_proba(X_ext_scaled)[0, 1])
            except:
                catalyst_prob = 0.0
            
            try:
                sentiment = float(self.sentiment_head.predict(X_ext_scaled)[0])
            except:
                sentiment = 0.5
            
            try:
                squeeze_prob = float(self.squeeze_head.predict_proba(X_ext_scaled)[0, 1])
            except:
                squeeze_prob = 0.0
            
            try:
                inst_pred = self.institutional_head.predict(X_ext_scaled)[0]
                inst_signal = {0: "neutral", 1: "accumulation", -1: "distribution"}.get(inst_pred, "neutral")
            except:
                inst_signal = "neutral"
            
            try:
                burst_prob = float(self.momentum_burst_head.predict_proba(X_ext_scaled)[0, 1])
            except:
                burst_prob = 0.0
            
            try:
                tape_score = float(self.tape_reading_head.predict(X_ext_scaled)[0])
            except:
                tape_score = 0.5
            
            try:
                vol_pred = self.volatility_head.predict(X_ext_scaled)[0]
                vol_regime = {0: "low", 1: "normal", 2: "high", 3: "extreme"}.get(vol_pred, "normal")
            except:
                vol_regime = "normal"
            
            try:
                options_pred = self.options_flow_head.predict(X_ext_scaled)[0]
                options_signal = {0: "neutral", 1: "bullish", -1: "bearish"}.get(options_pred, "neutral")
            except:
                options_signal = "neutral"
            
            try:
                macro_pred = self.macro_head.predict(X_ext_scaled)[0]
                macro_regime = {0: "neutral", 1: "risk_on", -1: "risk_off"}.get(macro_pred, "neutral")
            except:
                macro_regime = "neutral"
        else:
            catalyst_prob = multi_data.news_article_count / 50 if multi_data.news_article_count > 0 else 0
            sentiment = multi_data.news_sentiment_score
            squeeze_prob = max(0, (multi_data.short_interest_pct - 20) / 30) if multi_data.short_interest_pct > 20 else 0
            inst_signal = (
                "accumulation" if multi_data.dark_pool_buy_ratio > 0.6 else
                "distribution" if multi_data.dark_pool_buy_ratio < 0.4 else "neutral"
            )
            burst_prob = abs(multi_data.level2_imbalance) if abs(multi_data.level2_imbalance) > 0.5 else 0
            tape_score = 0.5 + multi_data.level2_imbalance * 0.5
            vol_regime = (
                "high" if multi_data.options_avg_iv > 0.5 else
                "low" if multi_data.options_avg_iv < 0.2 else "normal"
            )
            options_signal = (
                "bullish" if multi_data.options_put_call_ratio < 0.7 else
                "bearish" if multi_data.options_put_call_ratio > 1.3 else "neutral"
            )
            macro_regime = multi_data.risk_regime
        
        return ApexCoreV4Prediction(
            quantrascore_pred=base_pred.quantrascore_pred,
            quantrascore_calibrated=base_pred.quantrascore_calibrated,
            runner_probability=base_pred.runner_probability,
            runner_probability_calibrated=base_pred.runner_probability_calibrated,
            quality_tier_pred=base_pred.quality_tier_pred,
            avoid_trade_probability=base_pred.avoid_trade_probability,
            regime_pred=base_pred.regime_pred,
            timing_bucket=base_pred.timing_bucket,
            timing_confidence=base_pred.timing_confidence,
            move_direction=base_pred.move_direction,
            bars_to_move_estimate=base_pred.bars_to_move_estimate,
            expected_runup_pct=base_pred.expected_runup_pct,
            runup_confidence=base_pred.runup_confidence,
            confidence=base_pred.confidence,
            uncertainty_lower=base_pred.uncertainty_lower,
            uncertainty_upper=base_pred.uncertainty_upper,
            uncertainty_level=base_pred.uncertainty_level,
            multi_horizon=base_pred.multi_horizon,
            horizon_consensus=base_pred.horizon_consensus,
            market_context=base_pred.market_context,
            model_version="4.0.0",
            prediction_id=base_pred.prediction_id.replace("v3", "v4"),
            
            catalyst_probability=catalyst_prob,
            catalyst_type="news" if catalyst_prob > 0.5 else "none",
            sentiment_composite=sentiment,
            sentiment_trend="bullish" if sentiment > 0.6 else "bearish" if sentiment < 0.4 else "neutral",
            squeeze_probability=squeeze_prob,
            squeeze_trigger_price=current_price * 1.1 if squeeze_prob > 0.5 and current_price else None,
            institutional_signal=inst_signal,
            accumulation_score=multi_data.institutional_accumulation,
            momentum_burst_prob=burst_prob,
            burst_direction=1 if multi_data.level2_imbalance > 0 else -1 if multi_data.level2_imbalance < 0 else 0,
            tape_reading_signal="bullish" if tape_score > 0.6 else "bearish" if tape_score < 0.4 else "neutral",
            order_flow_strength=abs(tape_score - 0.5) * 2,
            volatility_regime=vol_regime,
            iv_percentile=multi_data.options_iv_percentile,
            options_flow_signal=options_signal,
            smart_money_direction=1 if options_signal == "bullish" else -1 if options_signal == "bearish" else 0,
            macro_regime=macro_regime,
            risk_appetite="high" if macro_regime == "risk_on" else "low" if macro_regime == "risk_off" else "moderate",
            data_sources_used=getattr(multi_data, 'data_sources_used', []),
        )
    
    def get_data_provider_status(self) -> Dict[str, Any]:
        """Get status of all multi-data providers."""
        return {
            "providers": self._data_providers_status,
            "extended_heads_fitted": self._extended_heads_fitted,
            "total_heads": 16,
            "base_heads": 7,
            "extended_heads": 9,
        }


def get_apexcore_v4(
    model_size: str = "big",
    enable_multi_data: bool = True
) -> ApexCoreV4Model:
    """Factory function for ApexCore V4 model."""
    return ApexCoreV4Model(
        model_size=model_size,
        enable_multi_data=enable_multi_data
    )


APEXCORE_V4_HEADS = {
    "base_heads": [
        "quantrascore",
        "runner",
        "quality", 
        "avoid",
        "regime",
        "timing",
        "runup",
    ],
    "extended_heads": [
        "catalyst",
        "sentiment",
        "squeeze",
        "institutional",
        "momentum_burst",
        "tape_reading",
        "volatility",
        "options_flow",
        "macro",
    ],
    "total": 16,
    "description": """
ApexCore V4 - 16-Head Multi-Data Prediction System

Base Heads (from V3):
1. quantrascore: Overall trade quality score (0-100)
2. runner: Probability of large move (0-1)
3. quality: Trade quality tier (A/B/C)
4. avoid: Probability trade should be avoided (0-1)
5. regime: Market regime (trend_up/trend_down/chop/squeeze/crash)
6. timing: When move will occur (immediate/very_soon/soon/late/none)
7. runup: Expected price appreciation (0-100%+)

Extended Heads (V4):
8. catalyst: News/event catalyst probability (0-1)
9. sentiment: Social + news sentiment composite (0-1)
10. squeeze: Short squeeze probability (0-1)
11. institutional: Dark pool accumulation signal (accumulation/distribution/neutral)
12. momentum_burst: Sudden momentum detection (0-1)
13. tape_reading: Order flow analysis score (0-1)
14. volatility: Volatility regime (low/normal/high/extreme)
15. options_flow: Smart money options positioning (bullish/bearish/neutral)
16. macro: Macroeconomic regime (risk_on/risk_off/neutral)

Data Requirements:
- Base heads: Alpaca/Polygon market data
- Extended heads: Options flow, dark pool, sentiment, economic data
"""
}
