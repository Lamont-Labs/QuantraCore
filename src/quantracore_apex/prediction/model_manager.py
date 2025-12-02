"""
Unified Model Manager with Hot-Reload and Database Persistence.

This module provides centralized model management with automatic hot-reload
when new models are trained. Supports both database and file-based storage
for durable persistence across republishes.
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MODEL_BASE_DIR = Path("models/apexcore_v3")
MODEL_NAME = "apexcore_v3"

COMPONENT_NAMES = [
    "quantrascore_head", "runner_head", "quality_head", "avoid_head",
    "regime_head", "timing_head", "runup_head", "scaler",
    "quality_encoder", "regime_encoder", "timing_encoder"
]


@dataclass
class ModelVersion:
    """Tracks model version information."""
    version: str
    trained_at: str
    training_samples: int
    manifest_hash: str
    storage_source: str = "file"
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "manifest_hash": self.manifest_hash,
            "storage_source": self.storage_source,
            "loaded_at": self.loaded_at.isoformat(),
        }


class ModelManager:
    """
    Centralized model manager with hot-reload and database persistence.
    
    Features:
    - Database storage for durable persistence (survives republishes)
    - Automatic file-based fallback when database unavailable
    - Automatic model version tracking
    - Hot-reload when new models are detected
    - Subscriber notification on model updates
    - Thread-safe operations
    - Cache management
    """
    
    _instance: Optional["ModelManager"] = None
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
            return
        
        self._model_cache: Dict[str, Any] = {}
        self._version_cache: Dict[str, ModelVersion] = {}
        self._subscribers: List[Callable[[str, ModelVersion], None]] = []
        self._cache_lock = threading.RLock()
        self._check_interval = 30
        self._last_check: Dict[str, float] = {}
        self._db_store = None
        self._initialized = True
        
        self._init_db_store()
        logger.info("[ModelManager] Initialized with hot-reload and database persistence")
    
    def _init_db_store(self) -> None:
        """Initialize database store if available."""
        try:
            from ..storage import get_model_store
            self._db_store = get_model_store()
            if os.environ.get("DATABASE_URL"):
                logger.info("[ModelManager] Database storage available")
            else:
                logger.info("[ModelManager] No database - using file storage only")
        except Exception as e:
            logger.warning(f"[ModelManager] Could not initialize database store: {e}")
            self._db_store = None
    
    def _get_manifest_path(self, model_size: str = "big") -> Path:
        """Get path to model manifest."""
        return MODEL_BASE_DIR / model_size / "manifest.json"
    
    def _load_manifest(self, model_size: str = "big") -> Optional[Dict]:
        """Load manifest from disk."""
        manifest_path = self._get_manifest_path(model_size)
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[ModelManager] Error loading manifest: {e}")
            return None
    
    def _compute_manifest_hash(self, manifest: Dict) -> str:
        """Compute hash of manifest for version comparison."""
        import hashlib
        content = json.dumps(manifest, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _check_for_updates(self, model_size: str = "big") -> bool:
        """Check if model has been updated since last load."""
        manifest = self._load_manifest(model_size)
        if not manifest:
            return False
        
        current_hash = self._compute_manifest_hash(manifest)
        cached_version = self._version_cache.get(model_size)
        
        if cached_version is None:
            return True
        
        return current_hash != cached_version.manifest_hash
    
    def _should_check(self, model_size: str = "big") -> bool:
        """Determine if we should check for updates (rate limiting)."""
        now = time.time()
        last_check = self._last_check.get(model_size, 0)
        
        if now - last_check > self._check_interval:
            self._last_check[model_size] = now
            return True
        
        return False
    
    def get_model(self, model_size: str = "big", force_reload: bool = False):
        """
        Get model instance with automatic hot-reload support.
        
        Args:
            model_size: Model variant ('big' or 'mini')
            force_reload: Force reload even if cached
        
        Returns:
            Loaded ApexCoreV3Model instance
        """
        with self._cache_lock:
            should_reload = force_reload
            
            if not force_reload and self._should_check(model_size):
                should_reload = self._check_for_updates(model_size)
                if should_reload:
                    logger.info(f"[ModelManager] New model detected for {model_size}, reloading...")
            
            if should_reload or model_size not in self._model_cache:
                self._load_model(model_size)
            
            return self._model_cache.get(model_size)
    
    def _load_model(self, model_size: str = "big", notify: bool = True) -> None:
        """Load model from database or disk, with database preferred."""
        storage_source = "file"
        
        try:
            from .apexcore_v3 import ApexCoreV3Model, clear_model_cache
            
            clear_model_cache()
            
            model = None
            
            if self._db_store and self._db_store.has_model(MODEL_NAME, model_size):
                model = self._load_from_database(model_size)
                if model:
                    storage_source = "database"
                    logger.info(f"[ModelManager] Loaded model {model_size} from database")
            
            if model is None:
                model = ApexCoreV3Model.load(model_size=model_size, use_cache=False)
                storage_source = "file"
                logger.info(f"[ModelManager] Loaded model {model_size} from file")
                
                if self._db_store and os.environ.get("DATABASE_URL"):
                    self._migrate_to_database(model, model_size)
            
            manifest = self._load_manifest(model_size)
            if manifest:
                version = ModelVersion(
                    version=manifest.get("version", "unknown"),
                    trained_at=manifest.get("trained_at", "unknown"),
                    training_samples=manifest.get("training_samples", 0),
                    manifest_hash=self._compute_manifest_hash(manifest),
                    storage_source=storage_source,
                )
                
                old_version = self._version_cache.get(model_size)
                self._version_cache[model_size] = version
                
                is_new_version = (old_version is None) or (old_version.manifest_hash != version.manifest_hash)
                if is_new_version:
                    if old_version:
                        logger.info(f"[ModelManager] Model {model_size} updated: {old_version.trained_at} -> {version.trained_at}")
                    else:
                        logger.info(f"[ModelManager] Model {model_size} loaded for first time: {version.trained_at}")
                    
                    if notify:
                        self._notify_subscribers(model_size, version)
            
            self._model_cache[model_size] = model
            logger.info(f"[ModelManager] Model {model_size} loaded successfully from {storage_source}")
            
        except Exception as e:
            logger.error(f"[ModelManager] Error loading model {model_size}: {e}")
            raise
    
    def _load_from_database(self, model_size: str) -> Optional[Any]:
        """Load model components from database and reconstruct model."""
        try:
            components = self._db_store.load_model_components(MODEL_NAME, model_size)
            if not components:
                return None
            
            from .apexcore_v3 import ApexCoreV3Model
            
            model = ApexCoreV3Model(model_size=model_size)
            
            if "quantrascore_head" in components:
                model.quantrascore_head = components["quantrascore_head"]
            if "runner_head" in components:
                model.runner_head = components["runner_head"]
            if "quality_head" in components:
                model.quality_head = components["quality_head"]
            if "avoid_head" in components:
                model.avoid_head = components["avoid_head"]
            if "regime_head" in components:
                model.regime_head = components["regime_head"]
            if "timing_head" in components:
                model.timing_head = components["timing_head"]
                model._timing_head_available = True
            if "runup_head" in components:
                model.runup_head = components["runup_head"]
                model._runup_head_available = True
            if "scaler" in components:
                model.scaler = components["scaler"]
            if "quality_encoder" in components:
                model.quality_encoder = components["quality_encoder"]
            if "regime_encoder" in components:
                model.regime_encoder = components["regime_encoder"]
            if "timing_encoder" in components:
                model.timing_encoder = components["timing_encoder"]
            
            model._is_fitted = True
            
            manifest = self._load_manifest(model_size)
            if manifest:
                from .apexcore_v3 import ApexCoreV3Manifest
                model._manifest = ApexCoreV3Manifest(**manifest)
            
            return model
            
        except Exception as e:
            logger.error(f"[ModelManager] Error loading from database: {e}")
            return None
    
    def _migrate_to_database(self, model: Any, model_size: str) -> bool:
        """Migrate a file-based model to database storage."""
        try:
            if not self._db_store:
                return False
            
            manifest = self._load_manifest(model_size)
            version = manifest.get("trained_at", datetime.utcnow().isoformat()) if manifest else datetime.utcnow().isoformat()
            training_samples = manifest.get("training_samples", 0) if manifest else 0
            
            components = {
                "quantrascore_head": model.quantrascore_head,
                "runner_head": model.runner_head,
                "quality_head": model.quality_head,
                "avoid_head": model.avoid_head,
                "regime_head": model.regime_head,
                "scaler": model.scaler,
                "quality_encoder": model.quality_encoder,
                "regime_encoder": model.regime_encoder,
            }
            
            if hasattr(model, 'timing_head') and model.timing_head is not None:
                components["timing_head"] = model.timing_head
            if hasattr(model, 'timing_encoder') and model.timing_encoder is not None:
                components["timing_encoder"] = model.timing_encoder
            if hasattr(model, 'runup_head') and model.runup_head is not None:
                components["runup_head"] = model.runup_head
            
            success = self._db_store.save_model_version(
                model_name=MODEL_NAME,
                model_size=model_size,
                version=version,
                components=components,
                training_samples=training_samples,
                manifest=manifest,
                notes="Migrated from file storage"
            )
            
            if success:
                logger.info(f"[ModelManager] Migrated model {model_size} to database storage")
            
            return success
            
        except Exception as e:
            logger.error(f"[ModelManager] Error migrating to database: {e}")
            return False
    
    def save_to_database(self, model: Any, model_size: str, version: Optional[str] = None) -> bool:
        """
        Save a trained model to database storage.
        
        Call this after training completes to persist the model durably.
        
        Args:
            model: The trained ApexCoreV3Model instance
            model_size: Size variant ('big' or 'mini')
            version: Optional version string (defaults to current timestamp)
        
        Returns:
            True if saved successfully
        """
        if not self._db_store or not os.environ.get("DATABASE_URL"):
            logger.warning("[ModelManager] No database available for persistence")
            return False
        
        try:
            manifest = model._manifest.to_dict() if hasattr(model, '_manifest') and model._manifest else None
            version = version or manifest.get("trained_at", datetime.utcnow().isoformat()) if manifest else datetime.utcnow().isoformat()
            training_samples = manifest.get("training_samples", 0) if manifest else 0
            
            components = {
                "quantrascore_head": model.quantrascore_head,
                "runner_head": model.runner_head,
                "quality_head": model.quality_head,
                "avoid_head": model.avoid_head,
                "regime_head": model.regime_head,
                "scaler": model.scaler,
                "quality_encoder": model.quality_encoder,
                "regime_encoder": model.regime_encoder,
            }
            
            if hasattr(model, 'timing_head') and model.timing_head is not None:
                components["timing_head"] = model.timing_head
            if hasattr(model, 'timing_encoder') and model.timing_encoder is not None:
                components["timing_encoder"] = model.timing_encoder
            if hasattr(model, 'runup_head') and model.runup_head is not None:
                components["runup_head"] = model.runup_head
            
            success = self._db_store.save_model_version(
                model_name=MODEL_NAME,
                model_size=model_size,
                version=version,
                components=components,
                training_samples=training_samples,
                manifest=manifest,
                notes=f"Trained at {version}"
            )
            
            if success:
                logger.info(f"[ModelManager] Saved model {model_size} v{version} to database")
                
                if model_size in self._version_cache:
                    self._version_cache[model_size].storage_source = "database"
            
            return success
            
        except Exception as e:
            logger.error(f"[ModelManager] Error saving to database: {e}")
            return False
    
    def subscribe(self, callback: Callable[[str, ModelVersion], None]) -> None:
        """Subscribe to model update notifications."""
        with self._cache_lock:
            self._subscribers.append(callback)
            logger.debug(f"[ModelManager] New subscriber added, total: {len(self._subscribers)}")
    
    def unsubscribe(self, callback: Callable[[str, ModelVersion], None]) -> None:
        """Unsubscribe from model update notifications."""
        with self._cache_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def _notify_subscribers(self, model_size: str, version: ModelVersion) -> None:
        """Notify all subscribers of model update."""
        for callback in self._subscribers:
            try:
                callback(model_size, version)
            except Exception as e:
                logger.error(f"[ModelManager] Error notifying subscriber: {e}")
    
    def force_reload_all(self) -> Dict[str, ModelVersion]:
        """Force reload all models and return new versions."""
        results = {}
        for model_size in ["big", "mini"]:
            manifest_path = self._get_manifest_path(model_size)
            has_db_model = self._db_store and self._db_store.has_model(MODEL_NAME, model_size)
            
            if manifest_path.exists() or has_db_model:
                try:
                    self._load_model(model_size)
                    results[model_size] = self._version_cache.get(model_size)
                except Exception as e:
                    logger.error(f"[ModelManager] Failed to reload {model_size}: {e}")
        
        return results
    
    def get_version_info(self, model_size: str = "big") -> Optional[Dict]:
        """Get current version information."""
        version = self._version_cache.get(model_size)
        return version.to_dict() if version else None
    
    def get_all_versions(self) -> Dict[str, Dict]:
        """Get version info for all loaded models."""
        return {
            size: version.to_dict() 
            for size, version in self._version_cache.items()
        }
    
    def get_version_history(self, model_size: str = "big", limit: int = 10) -> List[Dict]:
        """Get version history from database."""
        if not self._db_store:
            return []
        return self._db_store.get_version_history(MODEL_NAME, model_size, limit)
    
    def rollback_to_version(self, model_size: str, version: str) -> bool:
        """Rollback to a specific model version."""
        if not self._db_store:
            return False
        
        success = self._db_store.set_active_version(MODEL_NAME, model_size, version)
        if success:
            self._load_model(model_size)
            logger.info(f"[ModelManager] Rolled back {model_size} to version {version}")
        return success
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._db_store:
            return {"database_available": False}
        
        stats = self._db_store.get_storage_stats()
        stats["database_available"] = bool(os.environ.get("DATABASE_URL"))
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached models (forces reload on next access)."""
        with self._cache_lock:
            from .apexcore_v3 import clear_model_cache
            clear_model_cache()
            self._model_cache.clear()
            self._last_check.clear()
            logger.info("[ModelManager] All caches cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status."""
        db_available = bool(os.environ.get("DATABASE_URL"))
        
        return {
            "initialized": self._initialized,
            "models_loaded": list(self._model_cache.keys()),
            "versions": self.get_all_versions(),
            "subscribers_count": len(self._subscribers),
            "check_interval_seconds": self._check_interval,
            "last_checks": {k: datetime.fromtimestamp(v).isoformat() for k, v in self._last_check.items()},
            "database_persistence": {
                "available": db_available,
                "stats": self.get_storage_stats() if db_available else None
            }
        }


_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get global ModelManager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


def get_model(model_size: str = "big", force_reload: bool = False):
    """Convenience function to get model from manager."""
    return get_model_manager().get_model(model_size, force_reload)


def notify_model_updated(model_size: str = "big") -> None:
    """Call this after training a new model to trigger hot-reload."""
    manager = get_model_manager()
    manager._load_model(model_size)
    logger.info(f"[ModelManager] Model update notification processed for {model_size}")


def save_model_to_database(model: Any, model_size: str = "big") -> bool:
    """Save a trained model to database for persistent storage."""
    return get_model_manager().save_to_database(model, model_size)
