"""
Forward Validator - Records predictions BEFORE outcomes are known.

This provides TRUE, unbiased validation of model accuracy by:
1. Recording predictions with entry prices at time of prediction
2. Checking outcomes 5 trading days later
3. Calculating real precision with no look-ahead bias
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading
import time

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single forward prediction record."""
    symbol: str
    prediction_date: date
    model_score: float
    consensus_count: int
    avg_confidence: float
    entry_price: float
    price_source: str


@dataclass 
class ValidationStats:
    """Aggregated validation statistics."""
    total_predictions: int
    checked_outcomes: int
    pending_outcomes: int
    hits: int
    misses: int
    precision: float
    avg_gain_on_hits: float
    avg_loss_on_misses: float
    best_gain: float
    worst_loss: float
    avg_days_to_peak: float


class ForwardValidator:
    """
    Records predictions before outcomes are known and validates accuracy.
    
    This is the ONLY way to get truly unbiased accuracy metrics.
    """
    
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._polygon = None
        self._predictor = None
        
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)
    
    def _init_dependencies(self):
        """Lazy init dependencies to avoid circular imports."""
        if self._polygon is None:
            try:
                from src.quantracore_apex.data_layer.polygon_adapter import PolygonAdapter
                self._polygon = PolygonAdapter()
            except Exception as e:
                logger.warning(f"Could not init Polygon: {e}")
                
        if self._predictor is None:
            try:
                from src.quantracore_apex.ml.massive_predictor import MassivePredictor
                self._predictor = MassivePredictor()
            except Exception as e:
                logger.warning(f"Could not init predictor: {e}")
    
    def record_prediction(
        self,
        symbol: str,
        model_score: float,
        consensus_count: int = 0,
        avg_confidence: float = 0.0,
        entry_price: Optional[float] = None,
        price_source: str = "polygon"
    ) -> bool:
        """
        Record a prediction BEFORE the outcome is known.
        
        Args:
            symbol: Stock ticker
            model_score: QuantraScore (0-100)
            consensus_count: Number of models agreeing
            avg_confidence: Average confidence across models
            entry_price: Current price (will fetch if not provided)
            price_source: Where we got the price
            
        Returns:
            True if recorded successfully
        """
        today = date.today()
        
        if entry_price is None:
            self._init_dependencies()
            if self._polygon:
                try:
                    bars = self._polygon.get_daily_bars(symbol, days=1)
                    if bars and len(bars) > 0:
                        entry_price = bars[-1].get("close", 0)
                except Exception as e:
                    logger.warning(f"Could not get price for {symbol}: {e}")
        
        if entry_price is None or entry_price <= 0:
            logger.warning(f"No valid price for {symbol}, skipping prediction record")
            return False
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO forward_predictions 
                    (symbol, prediction_date, model_score, consensus_count, 
                     avg_confidence, entry_price, price_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, prediction_date) 
                    DO UPDATE SET 
                        model_score = EXCLUDED.model_score,
                        consensus_count = EXCLUDED.consensus_count,
                        avg_confidence = EXCLUDED.avg_confidence,
                        entry_price = EXCLUDED.entry_price
                """, (symbol, today, model_score, consensus_count, 
                      avg_confidence, entry_price, price_source))
                conn.commit()
            conn.close()
            logger.info(f"[ForwardValidator] Recorded prediction: {symbol} @ ${entry_price:.2f}, score={model_score}")
            return True
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            return False
    
    def record_batch_predictions(self, predictions: List[Dict[str, Any]]) -> int:
        """Record multiple predictions at once."""
        recorded = 0
        for pred in predictions:
            if self.record_prediction(
                symbol=pred.get("symbol"),
                model_score=pred.get("score", pred.get("model_score", 0)),
                consensus_count=pred.get("consensus_count", 0),
                avg_confidence=pred.get("avg_confidence", 0),
                entry_price=pred.get("entry_price"),
                price_source=pred.get("price_source", "polygon")
            ):
                recorded += 1
        return recorded
    
    def check_outcomes(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Check outcomes for predictions made 5+ trading days ago.
        
        Returns summary of what was checked.
        """
        self._init_dependencies()
        
        cutoff_date = date.today() - timedelta(days=lookback_days)
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM forward_predictions 
                    WHERE outcome_checked = FALSE 
                    AND prediction_date <= %s
                    ORDER BY prediction_date
                """, (cutoff_date,))
                pending = cur.fetchall()
            
            results = {"checked": 0, "hits": 0, "misses": 0, "errors": 0}
            
            for pred in pending:
                try:
                    outcome = self._check_single_outcome(pred)
                    if outcome:
                        with conn.cursor() as cur:
                            cur.execute("""
                                UPDATE forward_predictions 
                                SET outcome_checked = TRUE,
                                    outcome_date = %s,
                                    max_gain_pct = %s,
                                    actual_outcome = %s,
                                    exit_price = %s,
                                    days_to_peak = %s
                                WHERE id = %s
                            """, (
                                date.today(),
                                outcome["max_gain_pct"],
                                outcome["outcome"],
                                outcome["exit_price"],
                                outcome["days_to_peak"],
                                pred["id"]
                            ))
                            conn.commit()
                        
                        results["checked"] += 1
                        if outcome["outcome"] == "HIT":
                            results["hits"] += 1
                        else:
                            results["misses"] += 1
                except Exception as e:
                    logger.warning(f"Error checking outcome for {pred['symbol']}: {e}")
                    results["errors"] += 1
            
            conn.close()
            
            if results["checked"] > 0:
                logger.info(f"[ForwardValidator] Checked {results['checked']} outcomes: "
                           f"{results['hits']} hits, {results['misses']} misses")
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking outcomes: {e}")
            return {"error": str(e)}
    
    def _check_single_outcome(self, pred: Dict) -> Optional[Dict]:
        """Check outcome for a single prediction."""
        if not self._polygon:
            return None
            
        symbol = pred["symbol"]
        pred_date = pred["prediction_date"]
        entry_price = float(pred["entry_price"])
        target_gain = float(pred.get("target_gain_pct", 50.0))
        lookforward = int(pred.get("lookforward_days", 5))
        
        try:
            end_date = pred_date + timedelta(days=lookforward + 5)
            bars = self._polygon.get_daily_bars(
                symbol, 
                start_date=pred_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            if not bars or len(bars) < 2:
                return None
            
            max_high = 0
            max_gain_pct = -100
            days_to_peak = 0
            exit_price = entry_price
            
            for i, bar in enumerate(bars[1:lookforward+1], 1):
                high = bar.get("high", 0)
                if high > max_high:
                    max_high = high
                    days_to_peak = i
                    
            if max_high > 0 and entry_price > 0:
                max_gain_pct = ((max_high - entry_price) / entry_price) * 100
                exit_price = max_high
            
            outcome = "HIT" if max_gain_pct >= target_gain else "MISS"
            
            return {
                "max_gain_pct": round(max_gain_pct, 2),
                "outcome": outcome,
                "exit_price": round(exit_price, 4),
                "days_to_peak": days_to_peak
            }
            
        except Exception as e:
            logger.warning(f"Error getting bars for {symbol}: {e}")
            return None
    
    def get_stats(self, days: int = 30) -> ValidationStats:
        """Get validation statistics for the past N days."""
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cutoff = date.today() - timedelta(days=days)
                
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE outcome_checked = TRUE) as checked,
                        COUNT(*) FILTER (WHERE outcome_checked = FALSE) as pending,
                        COUNT(*) FILTER (WHERE actual_outcome = 'HIT') as hits,
                        COUNT(*) FILTER (WHERE actual_outcome = 'MISS') as misses,
                        AVG(max_gain_pct) FILTER (WHERE actual_outcome = 'HIT') as avg_gain_hits,
                        AVG(max_gain_pct) FILTER (WHERE actual_outcome = 'MISS') as avg_gain_misses,
                        MAX(max_gain_pct) as best_gain,
                        MIN(max_gain_pct) FILTER (WHERE outcome_checked = TRUE) as worst,
                        AVG(days_to_peak) FILTER (WHERE actual_outcome = 'HIT') as avg_days
                    FROM forward_predictions
                    WHERE prediction_date >= %s
                """, (cutoff,))
                
                row = cur.fetchone()
            conn.close()
            
            total = row["total"] or 0
            checked = row["checked"] or 0
            hits = row["hits"] or 0
            misses = row["misses"] or 0
            
            precision = (hits / checked * 100) if checked > 0 else 0.0
            
            return ValidationStats(
                total_predictions=total,
                checked_outcomes=checked,
                pending_outcomes=row["pending"] or 0,
                hits=hits,
                misses=misses,
                precision=round(precision, 2),
                avg_gain_on_hits=round(row["avg_gain_hits"] or 0, 2),
                avg_loss_on_misses=round(row["avg_gain_misses"] or 0, 2),
                best_gain=round(row["best_gain"] or 0, 2),
                worst_loss=round(row["worst"] or 0, 2),
                avg_days_to_peak=round(row["avg_days"] or 0, 1)
            )
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return ValidationStats(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """Get recent predictions with their outcomes."""
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT symbol, prediction_date, model_score, consensus_count,
                           entry_price, outcome_checked, actual_outcome, 
                           max_gain_pct, days_to_peak
                    FROM forward_predictions
                    ORDER BY prediction_date DESC, model_score DESC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def record_todays_top_predictions(self, min_score: float = 70, top_n: int = 20) -> int:
        """
        Record today's top predictions from the scanner.
        Called daily to capture predictions before outcomes are known.
        """
        self._init_dependencies()
        
        if not self._predictor:
            logger.warning("Predictor not available")
            return 0
        
        try:
            from src.quantracore_apex.scanning.universe_scanner import UniverseScanner
            scanner = UniverseScanner()
            setups = scanner.scan_universe(top_n=top_n * 2, min_score=min_score)
            
            recorded = 0
            for setup in setups[:top_n]:
                if self.record_prediction(
                    symbol=setup.get("symbol"),
                    model_score=setup.get("score", 0),
                    consensus_count=setup.get("consensus_count", 0),
                    avg_confidence=setup.get("avg_confidence", 0),
                    entry_price=setup.get("entry_price", setup.get("close")),
                ):
                    recorded += 1
            
            logger.info(f"[ForwardValidator] Recorded {recorded} top predictions for today")
            return recorded
            
        except Exception as e:
            logger.error(f"Error recording today's predictions: {e}")
            return 0
    
    def start_scheduler(self):
        """Start background scheduler for daily validation tasks."""
        if self._running:
            return
            
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("[ForwardValidator] Scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("[ForwardValidator] Scheduler stopped")
    
    def _scheduler_loop(self):
        """Background loop that runs validation tasks."""
        last_record_date = None
        last_check_date = None
        
        while self._running:
            try:
                now = datetime.now()
                today = date.today()
                
                if now.hour >= 16 and now.hour < 17:
                    if last_record_date != today:
                        self.record_todays_top_predictions()
                        last_record_date = today
                
                if now.hour >= 17 and now.hour < 18:
                    if last_check_date != today:
                        self.check_outcomes()
                        last_check_date = today
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            time.sleep(300)


_validator_instance: Optional[ForwardValidator] = None


def get_validator() -> ForwardValidator:
    """Get singleton ForwardValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ForwardValidator()
    return _validator_instance
