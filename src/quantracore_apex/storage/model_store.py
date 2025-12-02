"""
Database-backed Model Storage for Persistent Learning.

This module provides durable storage for ML models in PostgreSQL,
ensuring trained models survive republishes and restarts.

Features:
- Compressed binary storage (gzip) for efficient space usage
- Version history with rollback capability
- Automatic migration from file-based models
- Thread-safe operations
"""

import os
import io
import gzip
import json
import hashlib
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import joblib
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class StoredModel:
    """Represents a model stored in the database."""
    id: int
    model_name: str
    model_size: str
    version: str
    component_name: str
    data_hash: str
    created_at: datetime
    training_samples: int
    manifest: Optional[Dict]
    is_active: bool


class DatabaseModelStore:
    """
    PostgreSQL-backed model storage with compression and versioning.
    
    Stores serialized model components (heads, encoders, scalers) in the database
    with gzip compression for efficient storage.
    """
    
    _instance: Optional["DatabaseModelStore"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            if not self._db_url and os.environ.get("DATABASE_URL"):
                self._db_url = os.environ.get("DATABASE_URL")
                self._ensure_tables()
                logger.info("[DatabaseModelStore] Re-initialized with PostgreSQL storage (env was delayed)")
            return
        
        self._db_url = os.environ.get("DATABASE_URL")
        self._conn_lock = threading.Lock()
        self._initialized = True
        
        if self._db_url:
            self._ensure_tables()
            logger.info("[DatabaseModelStore] Initialized with PostgreSQL storage")
        else:
            logger.warning("[DatabaseModelStore] No DATABASE_URL - falling back to file storage only")
    
    def _get_connection(self):
        """Get a database connection."""
        if not self._db_url:
            self._db_url = os.environ.get("DATABASE_URL")
        if not self._db_url:
            return None
        return psycopg2.connect(self._db_url)
    
    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        try:
            conn = self._get_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        model_size VARCHAR(50) NOT NULL,
                        version VARCHAR(100) NOT NULL,
                        component_name VARCHAR(100) NOT NULL,
                        data_compressed BYTEA NOT NULL,
                        data_hash VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        training_samples INTEGER DEFAULT 0,
                        manifest JSONB,
                        is_active BOOLEAN DEFAULT TRUE,
                        UNIQUE(model_name, model_size, component_name, version)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_ml_models_active 
                    ON ml_models(model_name, model_size, is_active);
                    
                    CREATE INDEX IF NOT EXISTS idx_ml_models_version
                    ON ml_models(model_name, model_size, version);
                    
                    CREATE TABLE IF NOT EXISTS ml_model_versions (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        model_size VARCHAR(50) NOT NULL,
                        version VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        training_samples INTEGER DEFAULT 0,
                        manifest JSONB,
                        is_active BOOLEAN DEFAULT TRUE,
                        notes TEXT,
                        UNIQUE(model_name, model_size, version)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_ml_versions_active
                    ON ml_model_versions(model_name, model_size, is_active);
                """)
                conn.commit()
                logger.info("[DatabaseModelStore] Tables created/verified")
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error creating tables: {e}")
        finally:
            if conn:
                conn.close()
    
    def _compress_data(self, obj: Any) -> Tuple[bytes, str]:
        """Serialize and compress an object, return compressed bytes and hash."""
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        raw_data = buffer.getvalue()
        
        compressed = gzip.compress(raw_data, compresslevel=6)
        data_hash = hashlib.sha256(raw_data).hexdigest()
        
        compression_ratio = len(compressed) / len(raw_data) * 100
        logger.debug(f"[DatabaseModelStore] Compressed {len(raw_data)} -> {len(compressed)} bytes ({compression_ratio:.1f}%)")
        
        return compressed, data_hash
    
    def _decompress_data(self, compressed: bytes) -> Any:
        """Decompress and deserialize data."""
        raw_data = gzip.decompress(compressed)
        buffer = io.BytesIO(raw_data)
        return joblib.load(buffer)
    
    def save_component(
        self,
        model_name: str,
        model_size: str,
        version: str,
        component_name: str,
        component_data: Any,
        training_samples: int = 0,
        manifest: Optional[Dict] = None
    ) -> bool:
        """
        Save a model component to the database.
        
        Args:
            model_name: Name of the model (e.g., "apexcore_v3")
            model_size: Size variant (e.g., "big", "mini")
            version: Version string (e.g., "2025-12-02T12:00:00")
            component_name: Component name (e.g., "quantrascore_head")
            component_data: The actual model component to save
            training_samples: Number of training samples used
            manifest: Optional manifest data
        
        Returns:
            True if saved successfully
        """
        if not self._db_url:
            logger.warning("[DatabaseModelStore] No database connection - cannot save")
            return False
        
        try:
            compressed, data_hash = self._compress_data(component_data)
            
            with self._conn_lock:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ml_model_versions
                        SET is_active = FALSE
                        WHERE model_name = %s AND model_size = %s AND version != %s
                    """, (model_name, model_size, version))
                    
                    cur.execute("""
                        INSERT INTO ml_model_versions
                        (model_name, model_size, version, training_samples, manifest, is_active)
                        VALUES (%s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (model_name, model_size, version)
                        DO UPDATE SET 
                            is_active = TRUE,
                            training_samples = EXCLUDED.training_samples,
                            manifest = EXCLUDED.manifest
                    """, (
                        model_name, model_size, version, training_samples,
                        json.dumps(manifest) if manifest else None
                    ))
                    
                    cur.execute("""
                        UPDATE ml_models
                        SET is_active = FALSE
                        WHERE model_name = %s AND model_size = %s AND version != %s
                    """, (model_name, model_size, version))
                    
                    cur.execute("""
                        INSERT INTO ml_models 
                        (model_name, model_size, version, component_name, 
                         data_compressed, data_hash, training_samples, manifest, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                        ON CONFLICT (model_name, model_size, component_name, version)
                        DO UPDATE SET 
                            data_compressed = EXCLUDED.data_compressed,
                            data_hash = EXCLUDED.data_hash,
                            training_samples = EXCLUDED.training_samples,
                            manifest = EXCLUDED.manifest,
                            is_active = TRUE,
                            created_at = CURRENT_TIMESTAMP
                    """, (
                        model_name, model_size, version, component_name,
                        psycopg2.Binary(compressed), data_hash, 
                        training_samples, json.dumps(manifest) if manifest else None
                    ))
                    conn.commit()
                conn.close()
            
            logger.debug(f"[DatabaseModelStore] Saved {model_name}/{model_size}/{component_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error saving component: {e}")
            return False
    
    def load_component(
        self,
        model_name: str,
        model_size: str,
        component_name: str,
        version: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load a model component from the database.
        
        Args:
            model_name: Name of the model
            model_size: Size variant
            component_name: Component to load
            version: Specific version (None = latest active)
        
        Returns:
            The deserialized model component, or None if not found
        """
        if not self._db_url:
            return None
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if version:
                    cur.execute("""
                        SELECT data_compressed FROM ml_models
                        WHERE model_name = %s AND model_size = %s 
                        AND component_name = %s AND version = %s
                    """, (model_name, model_size, component_name, version))
                else:
                    cur.execute("""
                        SELECT data_compressed FROM ml_models
                        WHERE model_name = %s AND model_size = %s 
                        AND component_name = %s AND is_active = TRUE
                        ORDER BY created_at DESC LIMIT 1
                    """, (model_name, model_size, component_name))
                
                row = cur.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return self._decompress_data(bytes(row["data_compressed"]))
            
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error loading component: {e}")
            return None
    
    def save_model_version(
        self,
        model_name: str,
        model_size: str,
        version: str,
        components: Dict[str, Any],
        training_samples: int = 0,
        manifest: Optional[Dict] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Save all components of a model version atomically.
        
        Args:
            model_name: Name of the model
            model_size: Size variant
            version: Version string
            components: Dict of component_name -> component_data
            training_samples: Number of training samples
            manifest: Model manifest
            notes: Optional notes about this version
        
        Returns:
            True if all components saved successfully
        """
        if not self._db_url:
            return False
        
        try:
            with self._conn_lock:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE ml_model_versions
                            SET is_active = FALSE
                            WHERE model_name = %s AND model_size = %s
                        """, (model_name, model_size))
                        
                        cur.execute("""
                            UPDATE ml_models
                            SET is_active = FALSE
                            WHERE model_name = %s AND model_size = %s
                        """, (model_name, model_size))
                        
                        cur.execute("""
                            INSERT INTO ml_model_versions
                            (model_name, model_size, version, training_samples, manifest, notes, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                            ON CONFLICT (model_name, model_size, version)
                            DO UPDATE SET 
                                training_samples = EXCLUDED.training_samples,
                                manifest = EXCLUDED.manifest,
                                notes = EXCLUDED.notes,
                                is_active = TRUE,
                                created_at = CURRENT_TIMESTAMP
                        """, (
                            model_name, model_size, version, training_samples,
                            json.dumps(manifest) if manifest else None, notes
                        ))
                    
                    for comp_name, comp_data in components.items():
                        compressed, data_hash = self._compress_data(comp_data)
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO ml_models 
                                (model_name, model_size, version, component_name,
                                 data_compressed, data_hash, training_samples, manifest, is_active)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                                ON CONFLICT (model_name, model_size, component_name, version)
                                DO UPDATE SET 
                                    data_compressed = EXCLUDED.data_compressed,
                                    data_hash = EXCLUDED.data_hash,
                                    training_samples = EXCLUDED.training_samples,
                                    manifest = EXCLUDED.manifest,
                                    is_active = TRUE,
                                    created_at = CURRENT_TIMESTAMP
                            """, (
                                model_name, model_size, version, comp_name,
                                psycopg2.Binary(compressed), data_hash,
                                training_samples, json.dumps(manifest) if manifest else None
                            ))
                    
                    conn.commit()
                    logger.info(f"[DatabaseModelStore] Saved model version {model_name}/{model_size} v{version} ({len(components)} components)")
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error saving model version: {e}")
            return False
    
    def load_model_components(
        self,
        model_name: str,
        model_size: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load all components of a model version.
        
        Args:
            model_name: Name of the model
            model_size: Size variant
            version: Specific version (None = latest active)
        
        Returns:
            Dict of component_name -> component_data, or None if not found
        """
        if not self._db_url:
            return None
        
        try:
            conn = self._get_connection()
            
            if not version:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT version FROM ml_model_versions
                        WHERE model_name = %s AND model_size = %s AND is_active = TRUE
                        ORDER BY created_at DESC LIMIT 1
                    """, (model_name, model_size))
                    row = cur.fetchone()
                    if row:
                        version = row["version"]
            
            if not version:
                conn.close()
                return None
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT component_name, data_compressed FROM ml_models
                    WHERE model_name = %s AND model_size = %s AND version = %s
                """, (model_name, model_size, version))
                rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            components = {}
            for row in rows:
                comp_name = row["component_name"]
                comp_data = self._decompress_data(bytes(row["data_compressed"]))
                components[comp_name] = comp_data
            
            logger.info(f"[DatabaseModelStore] Loaded {len(components)} components for {model_name}/{model_size} v{version}")
            return components
            
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error loading model components: {e}")
            return None
    
    def get_version_history(
        self,
        model_name: str,
        model_size: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get version history for a model."""
        if not self._db_url:
            return []
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT version, created_at, training_samples, manifest, is_active, notes
                    FROM ml_model_versions
                    WHERE model_name = %s AND model_size = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (model_name, model_size, limit))
                rows = cur.fetchall()
            conn.close()
            
            return [
                {
                    "version": row["version"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "training_samples": row["training_samples"],
                    "manifest": row["manifest"],
                    "is_active": row["is_active"],
                    "notes": row["notes"]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error getting version history: {e}")
            return []
    
    def set_active_version(
        self,
        model_name: str,
        model_size: str,
        version: str
    ) -> bool:
        """Set a specific version as the active version."""
        if not self._db_url:
            return False
        
        try:
            with self._conn_lock:
                conn = self._get_connection()
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ml_model_versions
                        SET is_active = FALSE
                        WHERE model_name = %s AND model_size = %s
                    """, (model_name, model_size))
                    
                    cur.execute("""
                        UPDATE ml_model_versions
                        SET is_active = TRUE
                        WHERE model_name = %s AND model_size = %s AND version = %s
                    """, (model_name, model_size, version))
                    
                    cur.execute("""
                        UPDATE ml_models
                        SET is_active = FALSE
                        WHERE model_name = %s AND model_size = %s
                    """, (model_name, model_size))
                    
                    cur.execute("""
                        UPDATE ml_models
                        SET is_active = TRUE
                        WHERE model_name = %s AND model_size = %s AND version = %s
                    """, (model_name, model_size, version))
                    
                    conn.commit()
                conn.close()
            
            logger.info(f"[DatabaseModelStore] Set active version: {model_name}/{model_size} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error setting active version: {e}")
            return False
    
    def has_model(self, model_name: str, model_size: str) -> bool:
        """Check if a model exists in the database."""
        if not self._db_url:
            return False
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM ml_model_versions
                    WHERE model_name = %s AND model_size = %s AND is_active = TRUE
                    LIMIT 1
                """, (model_name, model_size))
                exists = cur.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error checking model existence: {e}")
            return False
    
    def get_active_version(self, model_name: str, model_size: str) -> Optional[Dict]:
        """Get info about the current active version."""
        if not self._db_url:
            return None
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT version, created_at, training_samples, manifest, notes
                    FROM ml_model_versions
                    WHERE model_name = %s AND model_size = %s AND is_active = TRUE
                    ORDER BY created_at DESC LIMIT 1
                """, (model_name, model_size))
                row = cur.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return {
                "version": row["version"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "training_samples": row["training_samples"],
                "manifest": row["manifest"],
                "notes": row["notes"]
            }
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error getting active version: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._db_url:
            return {"available": False}
        
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT model_name || model_size) as model_count,
                        COUNT(DISTINCT version) as version_count,
                        COUNT(*) as component_count,
                        SUM(LENGTH(data_compressed)) as total_compressed_bytes
                    FROM ml_models
                """)
                stats = cur.fetchone()
            conn.close()
            
            return {
                "available": True,
                "models": stats["model_count"] or 0,
                "versions": stats["version_count"] or 0,
                "components": stats["component_count"] or 0,
                "storage_bytes": stats["total_compressed_bytes"] or 0,
                "storage_mb": round((stats["total_compressed_bytes"] or 0) / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"[DatabaseModelStore] Error getting stats: {e}")
            return {"available": False, "error": str(e)}


_store: Optional[DatabaseModelStore] = None

def get_model_store() -> DatabaseModelStore:
    """Get global DatabaseModelStore instance."""
    global _store
    if _store is None:
        _store = DatabaseModelStore()
    return _store
