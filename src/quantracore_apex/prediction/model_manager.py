"""
Unified Model Manager with Hot-Reload Support.

This module provides centralized model management with automatic hot-reload
when new models are trained. All services should use this manager instead
of directly loading models.
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


@dataclass
class ModelVersion:
    """Tracks model version information."""
    version: str
    trained_at: str
    training_samples: int
    manifest_hash: str
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "manifest_hash": self.manifest_hash,
            "loaded_at": self.loaded_at.isoformat(),
        }


class ModelManager:
    """
    Centralized model manager with hot-reload capability.
    
    Features:
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
        self._initialized = True
        
        logger.info("[ModelManager] Initialized with hot-reload support")
    
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
        """Load model from disk and update cache."""
        try:
            from .apexcore_v3 import ApexCoreV3Model, clear_model_cache
            
            clear_model_cache()
            
            model = ApexCoreV3Model.load(model_size=model_size, use_cache=False)
            
            manifest = self._load_manifest(model_size)
            if manifest:
                version = ModelVersion(
                    version=manifest.get("version", "unknown"),
                    trained_at=manifest.get("trained_at", "unknown"),
                    training_samples=manifest.get("training_samples", 0),
                    manifest_hash=self._compute_manifest_hash(manifest),
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
            logger.info(f"[ModelManager] Model {model_size} loaded successfully")
            
        except Exception as e:
            logger.error(f"[ModelManager] Error loading model {model_size}: {e}")
            raise
    
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
            if manifest_path.exists():
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
        return {
            "initialized": self._initialized,
            "models_loaded": list(self._model_cache.keys()),
            "versions": self.get_all_versions(),
            "subscribers_count": len(self._subscribers),
            "check_interval_seconds": self._check_interval,
            "last_checks": {k: datetime.fromtimestamp(v).isoformat() for k, v in self._last_check.items()},
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
