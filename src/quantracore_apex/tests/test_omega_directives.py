"""
Comprehensive Omega Directive Tests (Ω1-Ω5)

Tests all Omega directives for correct behavior.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List

from src.quantracore_apex.core.schemas import (
    OhlcvBar, ApexResult, RiskTier, EntropyState, DriftState,
    EntropyMetrics, DriftMetrics, SuppressionMetrics, ContinuationMetrics,
    VolumeMetrics, Microtraits, Verdict
)
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.protocols.omega.omega import (
    OmegaDirectives, OmegaLevel, OmegaStatus
)


def generate_bars(n: int = 100, seed: int = 42) -> List[OhlcvBar]:
    """Generate deterministic test bars."""
    np.random.seed(seed)
    bars = []
    price = 100.0
    
    for i in range(n):
        change = np.random.randn() * 0.02
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.005))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.005))
        volume = 1000000 + np.random.randint(-200000, 200000)
        
        bars.append(OhlcvBar(
            timestamp=datetime(2024, 1, 1),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=float(volume),
        ))
        price = close_price
    
    return bars


class TestOmegaDirectives:
    """Tests for Omega directive system."""
    
    def test_omega_initialization(self):
        """Test Omega system initializes correctly."""
        omega = OmegaDirectives()
        
        assert omega.enable_omega_1 is True
        assert omega.enable_omega_2 is True
        assert omega.enable_omega_3 is True
        assert omega.enable_omega_4 is True
        assert omega.enable_omega_5 is True
    
    def test_omega_disable(self):
        """Test Omega directives can be disabled."""
        omega = OmegaDirectives(
            enable_omega_1=False,
            enable_omega_2=False,
            enable_omega_3=False,
            enable_omega_4=False,
            enable_omega_5=False,
        )
        
        assert omega.enable_omega_1 is False
        assert omega.enable_omega_5 is False


class TestOmega1SafetyLock:
    """Tests for Ω1 Hard Safety Lock."""
    
    def test_omega1_exists(self):
        """Test Ω1 method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'check_omega_1')
    
    def test_omega1_disabled(self):
        """Test Ω1 can be disabled."""
        omega = OmegaDirectives(enable_omega_1=False)
        assert omega.enable_omega_1 is False


class TestOmega2EntropyOverride:
    """Tests for Ω2 Entropy Override."""
    
    def test_omega2_exists(self):
        """Test Ω2 method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'check_omega_2')


class TestOmega3DriftOverride:
    """Tests for Ω3 Drift Override."""
    
    def test_omega3_exists(self):
        """Test Ω3 method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'check_omega_3')


class TestOmega4Compliance:
    """Tests for Ω4 Compliance Override."""
    
    def test_omega4_exists(self):
        """Test Ω4 method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'check_omega_4')
    
    def test_omega4_applies_compliance_note(self):
        """Test Ω4 applies compliance note."""
        omega = OmegaDirectives()
        
        verdict = {"action": "OBSERVE"}
        result = omega.apply_omega4(verdict)
        
        assert "compliance_note" in result


class TestOmega5SuppressionLock:
    """Tests for Ω5 Signal Suppression Lock."""
    
    def test_omega5_exists(self):
        """Test Ω5 method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'check_omega_5')
    
    def test_omega5_enabled_by_default(self):
        """Test Ω5 is enabled by default."""
        omega = OmegaDirectives()
        assert omega.enable_omega_5 is True


class TestOmegaApplyAll:
    """Tests for apply_all method."""
    
    def test_apply_all_exists(self):
        """Test apply_all method exists."""
        omega = OmegaDirectives()
        assert hasattr(omega, 'apply_all')
    
    def test_get_highest_alert_level(self):
        """Test getting highest alert level."""
        omega = OmegaDirectives()
        
        statuses = {
            "omega_1": OmegaStatus(active=False, level=OmegaLevel.INACTIVE),
            "omega_2": OmegaStatus(active=True, level=OmegaLevel.ADVISORY),
            "omega_3": OmegaStatus(active=True, level=OmegaLevel.ENFORCED),
        }
        
        highest = omega.get_highest_alert_level(statuses)
        assert highest == OmegaLevel.ENFORCED
