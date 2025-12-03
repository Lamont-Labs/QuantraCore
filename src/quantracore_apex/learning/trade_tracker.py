"""
Trade Tracker - Records trade outcomes for model learning

Tracks:
- Entry signals (what the model predicted)
- Trade outcomes (what actually happened)
- Feature snapshots at entry time
- Performance metrics
"""

import os
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import Json
    HAS_DB = True
except ImportError:
    HAS_DB = False


class TradeTracker:
    """Tracks trade outcomes to enable model learning from real results."""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.trades_dir = Path('data/learning/trades')
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables for trade tracking."""
        if not HAS_DB or not self.db_url:
            return
            
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    entry_date TIMESTAMP NOT NULL,
                    exit_date TIMESTAMP,
                    entry_price FLOAT NOT NULL,
                    exit_price FLOAT,
                    quantity FLOAT NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    model_probability FLOAT,
                    model_votes INT,
                    predicted_gain FLOAT,
                    actual_gain FLOAT,
                    max_gain FLOAT,
                    max_drawdown FLOAT,
                    was_correct BOOLEAN,
                    features JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id SERIAL PRIMARY KEY,
                    session_date TIMESTAMP NOT NULL,
                    trades_analyzed INT,
                    correct_predictions INT,
                    incorrect_predictions INT,
                    precision_before FLOAT,
                    precision_after FLOAT,
                    auc_before FLOAT,
                    auc_after FLOAT,
                    model_version VARCHAR(50),
                    training_samples INT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            print("[TradeTracker] Database tables initialized")
        except Exception as e:
            print(f"[TradeTracker] DB init warning: {e}")
    
    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        model_probability: float,
        model_votes: int,
        features: Dict,
        metadata: Optional[Dict] = None
    ) -> int:
        """Record a new trade entry with model predictions."""
        
        trade = {
            'symbol': symbol,
            'entry_date': datetime.now().isoformat(),
            'entry_price': entry_price,
            'quantity': quantity,
            'side': side,
            'model_probability': model_probability,
            'model_votes': model_votes,
            'features': features,
            'metadata': metadata or {},
            'status': 'open'
        }
        
        trade_id = None
        
        if HAS_DB and self.db_url:
            try:
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO trade_outcomes 
                    (symbol, entry_date, entry_price, quantity, side, 
                     model_probability, model_votes, features, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    symbol, datetime.now(), entry_price, quantity, side,
                    model_probability, model_votes, 
                    Json(features), Json(metadata or {})
                ))
                
                trade_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[TradeTracker] DB error: {e}")
        
        # Also save to file
        file_path = self.trades_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        trade['id'] = trade_id
        with open(file_path, 'w') as f:
            json.dump(trade, f, indent=2)
        
        return trade_id or hash(file_path.name)
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        max_price: float,
        min_price: float,
        trade_id: Optional[int] = None
    ) -> Dict:
        """Record trade exit and calculate outcomes."""
        
        result = None
        
        if HAS_DB and self.db_url:
            try:
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                
                # Find the most recent open trade for this symbol
                cur.execute("""
                    SELECT id, entry_price, model_probability, side
                    FROM trade_outcomes
                    WHERE symbol = %s AND exit_date IS NULL
                    ORDER BY entry_date DESC LIMIT 1
                """, (symbol,))
                
                row = cur.fetchone()
                if row:
                    trade_id, entry_price, prob, side = row
                    
                    # Calculate gains
                    if side == 'buy':
                        actual_gain = (exit_price - entry_price) / entry_price
                        max_gain = (max_price - entry_price) / entry_price
                        max_drawdown = (entry_price - min_price) / entry_price
                    else:
                        actual_gain = (entry_price - exit_price) / entry_price
                        max_gain = (entry_price - min_price) / entry_price
                        max_drawdown = (max_price - entry_price) / entry_price
                    
                    # Determine if prediction was correct (50%+ gain target)
                    was_correct = max_gain >= 0.50
                    
                    cur.execute("""
                        UPDATE trade_outcomes
                        SET exit_date = %s, exit_price = %s, 
                            actual_gain = %s, max_gain = %s, max_drawdown = %s,
                            was_correct = %s
                        WHERE id = %s
                    """, (
                        datetime.now(), exit_price,
                        actual_gain, max_gain, max_drawdown,
                        was_correct, trade_id
                    ))
                    
                    result = {
                        'trade_id': trade_id,
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'actual_gain': actual_gain,
                        'max_gain': max_gain,
                        'was_correct': was_correct,
                        'model_probability': prob
                    }
                    
                    conn.commit()
                
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[TradeTracker] Exit error: {e}")
        
        return result
    
    def get_learning_data(self, days: int = 30) -> Tuple[List[Dict], Dict]:
        """Get trade outcomes for model learning."""
        
        trades = []
        stats = {'total': 0, 'correct': 0, 'incorrect': 0}
        
        if HAS_DB and self.db_url:
            try:
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                
                cutoff = datetime.now() - timedelta(days=days)
                
                cur.execute("""
                    SELECT symbol, entry_price, exit_price, 
                           model_probability, model_votes, features,
                           actual_gain, max_gain, was_correct
                    FROM trade_outcomes
                    WHERE exit_date IS NOT NULL AND entry_date > %s
                    ORDER BY entry_date DESC
                """, (cutoff,))
                
                for row in cur.fetchall():
                    trade = {
                        'symbol': row[0],
                        'entry_price': row[1],
                        'exit_price': row[2],
                        'model_probability': row[3],
                        'model_votes': row[4],
                        'features': row[5] or {},
                        'actual_gain': row[6],
                        'max_gain': row[7],
                        'was_correct': row[8]
                    }
                    trades.append(trade)
                    
                    stats['total'] += 1
                    if trade['was_correct']:
                        stats['correct'] += 1
                    else:
                        stats['incorrect'] += 1
                
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[TradeTracker] Get data error: {e}")
        
        if stats['total'] > 0:
            stats['precision'] = stats['correct'] / stats['total']
        else:
            stats['precision'] = 0
        
        return trades, stats
    
    def get_hard_negatives(self) -> List[Dict]:
        """Get trades where model was confident but wrong (for retraining)."""
        
        hard_negatives = []
        
        if HAS_DB and self.db_url:
            try:
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT features, model_probability, actual_gain
                    FROM trade_outcomes
                    WHERE was_correct = FALSE 
                      AND model_probability >= 0.5
                      AND features IS NOT NULL
                    ORDER BY model_probability DESC
                """)
                
                for row in cur.fetchall():
                    if row[0]:
                        hard_negatives.append({
                            'features': row[0],
                            'probability': row[1],
                            'actual_gain': row[2],
                            'label': 0  # These are negatives
                        })
                
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[TradeTracker] Hard negatives error: {e}")
        
        return hard_negatives
    
    def record_learning_session(
        self,
        trades_analyzed: int,
        correct: int,
        incorrect: int,
        precision_before: float,
        precision_after: float,
        auc_before: float,
        auc_after: float,
        model_version: str,
        training_samples: int,
        notes: str = ""
    ):
        """Record a learning/retraining session."""
        
        if HAS_DB and self.db_url:
            try:
                conn = psycopg2.connect(self.db_url)
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO learning_sessions
                    (session_date, trades_analyzed, correct_predictions, 
                     incorrect_predictions, precision_before, precision_after,
                     auc_before, auc_after, model_version, training_samples, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now(), trades_analyzed, correct, incorrect,
                    precision_before, precision_after, auc_before, auc_after,
                    model_version, training_samples, notes
                ))
                
                conn.commit()
                cur.close()
                conn.close()
                print(f"[TradeTracker] Learning session recorded")
            except Exception as e:
                print(f"[TradeTracker] Session error: {e}")
