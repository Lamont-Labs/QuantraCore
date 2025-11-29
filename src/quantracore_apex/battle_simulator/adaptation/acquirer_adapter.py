"""
Acquirer Adaptation Layer.

Provides infrastructure abstraction for M&A compatibility.

PURPOSE:
- Allow QuantraCore to adapt to any acquiring firm's infrastructure
- Abstract data feeds, risk frameworks, and compliance systems
- Provide configurable mappings for different systems
- Enable seamless integration with Bloomberg, Refinitiv, or proprietary systems

SUPPORTED INFRASTRUCTURES:
- Bloomberg Terminal
- Refinitiv Eikon
- Custom/Proprietary
- QuantraCore Native
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..models import AdaptationProfile, ComplianceStatus

logger = logging.getLogger(__name__)


@dataclass
class InfrastructureSpec:
    """Specification for a target infrastructure."""
    name: str
    data_format: str
    api_style: str
    risk_framework: str
    compliance_requirements: List[str]
    supported_asset_classes: List[str]


SUPPORTED_INFRASTRUCTURES: Dict[str, InfrastructureSpec] = {
    "bloomberg": InfrastructureSpec(
        name="Bloomberg Terminal",
        data_format="BLPAPI",
        api_style="request_response",
        risk_framework="bloomberg_port",
        compliance_requirements=["MIFID2", "SEC", "FINRA"],
        supported_asset_classes=["equities", "fixed_income", "derivatives", "fx", "commodities"],
    ),
    "refinitiv": InfrastructureSpec(
        name="Refinitiv Eikon",
        data_format="Elektron",
        api_style="streaming",
        risk_framework="refinitiv_risk",
        compliance_requirements=["MIFID2", "SEC", "ESMA"],
        supported_asset_classes=["equities", "fixed_income", "fx"],
    ),
    "quantracore_native": InfrastructureSpec(
        name="QuantraCore Native",
        data_format="JSON/OHLCV",
        api_style="rest_websocket",
        risk_framework="apex_risk",
        compliance_requirements=["research_only"],
        supported_asset_classes=["equities", "etfs"],
    ),
    "custom": InfrastructureSpec(
        name="Custom/Proprietary",
        data_format="configurable",
        api_style="configurable",
        risk_framework="configurable",
        compliance_requirements=["configurable"],
        supported_asset_classes=["configurable"],
    ),
}


class AcquirerAdapter:
    """
    Adapter for integrating QuantraCore with acquirer infrastructure.
    
    Provides abstraction layers for:
    - Data feed translation
    - Risk framework mapping
    - Compliance overlay configuration
    - Protocol configuration
    - Output format translation
    """
    
    def __init__(self, profile: Optional[AdaptationProfile] = None):
        self.profile = profile or AdaptationProfile.create_default()
        self._data_translators: Dict[str, Callable] = {}
        self._risk_mappings: Dict[str, Any] = {}
        self._compliance_hooks: List[Callable] = []
        
        logger.info(f"[AcquirerAdapter] Initialized for {self.profile.profile_name}")
    
    @classmethod
    def for_bloomberg(cls) -> "AcquirerAdapter":
        """Create adapter configured for Bloomberg infrastructure."""
        return cls(AdaptationProfile.create_bloomberg_compatible())
    
    @classmethod
    def for_refinitiv(cls) -> "AcquirerAdapter":
        """Create adapter configured for Refinitiv infrastructure."""
        return cls(AdaptationProfile.create_refinitiv_compatible())
    
    @classmethod
    def for_custom(
        cls,
        profile_name: str,
        data_mappings: Dict[str, str],
        risk_overrides: Optional[Dict[str, Any]] = None,
        compliance_requirements: Optional[List[str]] = None,
    ) -> "AcquirerAdapter":
        """Create adapter with custom configuration."""
        profile = AdaptationProfile(
            profile_id=f"custom_{profile_name.lower().replace(' ', '_')}",
            profile_name=profile_name,
            created_at=datetime.utcnow(),
            target_infrastructure="custom",
            data_feed_mappings=data_mappings,
            risk_framework_overrides=risk_overrides or {},
            compliance_overlays={
                "requirements": compliance_requirements or []
            },
            validation_status="pending",
        )
        return cls(profile)
    
    def translate_ohlcv(
        self,
        ohlcv_data: Dict[str, Any],
        target_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate OHLCV data to target infrastructure format.
        
        Supports Bloomberg, Refinitiv, and custom formats.
        """
        target = target_format or self.profile.target_infrastructure
        mappings = self.profile.data_feed_mappings
        
        if not mappings:
            return ohlcv_data
        
        translated = {}
        
        field_map = {
            "open": mappings.get("open", "open"),
            "high": mappings.get("high", "high"),
            "low": mappings.get("low", "low"),
            "close": mappings.get("close", "close"),
            "volume": mappings.get("volume", "volume"),
            "vwap": mappings.get("vwap", "vwap"),
        }
        
        for our_field, their_field in field_map.items():
            if our_field in ohlcv_data:
                translated[their_field] = ohlcv_data[our_field]
        
        for key, value in ohlcv_data.items():
            if key not in field_map:
                translated[key] = value
        
        return translated
    
    def translate_signal_output(
        self,
        apex_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate ApexEngine output to target format.
        
        Maps our fields to the acquirer's expected format.
        """
        base_output = {
            "signal_id": apex_result.get("signal_id", ""),
            "symbol": apex_result.get("symbol", ""),
            "timestamp": apex_result.get("timestamp", datetime.utcnow().isoformat()),
            "score": apex_result.get("quantrascore", 0),
            "direction": apex_result.get("direction", "NEUTRAL"),
            "quality_tier": apex_result.get("quality_tier", ""),
            "risk_tier": apex_result.get("risk_tier", ""),
            "compliance_status": ComplianceStatus.RESEARCH_ONLY.value,
        }
        
        if self.profile.target_infrastructure == "bloomberg":
            return self._format_for_bloomberg(base_output)
        elif self.profile.target_infrastructure == "refinitiv":
            return self._format_for_refinitiv(base_output)
        else:
            return base_output
    
    def _format_for_bloomberg(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format output for Bloomberg integration."""
        return {
            "TICKER": output.get("symbol", ""),
            "SIGNAL_SCORE": output.get("score", 0),
            "SIGNAL_DIRECTION": output.get("direction", "NEUTRAL"),
            "SIGNAL_QUALITY": output.get("quality_tier", ""),
            "RISK_RATING": output.get("risk_tier", ""),
            "TIMESTAMP": output.get("timestamp", ""),
            "SOURCE": "QUANTRACORE_APEX",
            "COMPLIANCE_NOTE": "Research Only - Not Investment Advice",
        }
    
    def _format_for_refinitiv(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format output for Refinitiv integration."""
        return {
            "RIC": output.get("symbol", ""),
            "SignalScore": output.get("score", 0),
            "SignalDirection": output.get("direction", "NEUTRAL"),
            "QualityTier": output.get("quality_tier", ""),
            "RiskRating": output.get("risk_tier", ""),
            "UpdateTime": output.get("timestamp", ""),
            "Source": "QuantraCoreApex",
            "Disclaimer": "Research Only",
        }
    
    def map_risk_tier(
        self,
        apex_risk_tier: str,
    ) -> Dict[str, Any]:
        """
        Map ApexEngine risk tier to target risk framework.
        
        Translates our risk tiers to the acquirer's risk terminology.
        """
        overrides = self.profile.risk_framework_overrides
        
        default_mapping = {
            "minimal": {"level": 1, "color": "green", "max_allocation": 0.20},
            "low": {"level": 2, "color": "green", "max_allocation": 0.15},
            "moderate": {"level": 3, "color": "yellow", "max_allocation": 0.10},
            "high": {"level": 4, "color": "orange", "max_allocation": 0.05},
            "extreme": {"level": 5, "color": "red", "max_allocation": 0.0},
        }
        
        if overrides and apex_risk_tier in overrides:
            return overrides[apex_risk_tier]
        
        return default_mapping.get(apex_risk_tier.lower(), default_mapping["moderate"])
    
    def apply_compliance_overlay(
        self,
        signal_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply compliance overlays for the target infrastructure.
        
        Adds required compliance fields and checks.
        """
        overlays = self.profile.compliance_overlays
        
        signal_data["compliance"] = {
            "status": ComplianceStatus.RESEARCH_ONLY.value,
            "classification": "research",
            "not_investment_advice": True,
            "regulatory_frameworks": overlays.get("requirements", ["research_only"]),
        }
        
        for hook in self._compliance_hooks:
            try:
                signal_data = hook(signal_data)
            except Exception as e:
                logger.warning(f"[AcquirerAdapter] Compliance hook error: {e}")
        
        return signal_data
    
    def register_compliance_hook(self, hook: Callable) -> None:
        """Register a custom compliance check hook."""
        self._compliance_hooks.append(hook)
        logger.info("[AcquirerAdapter] Registered compliance hook")
    
    def get_omega_mapping(self, directive_id: str) -> Optional[str]:
        """
        Map Omega directive to target infrastructure equivalent.
        
        Some infrastructures have their own override/safety systems.
        """
        mappings = self.profile.omega_directive_mappings
        return mappings.get(directive_id, directive_id)
    
    def configure_protocol_for_infrastructure(
        self,
        protocol_id: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Configure a protocol for the target infrastructure.
        
        Returns protocol configuration adapted for the infrastructure.
        """
        base_config = self.profile.protocol_configurations.get(protocol_id, {})
        
        if config_overrides:
            base_config = {**base_config, **config_overrides}
        
        base_config["infrastructure"] = self.profile.target_infrastructure
        base_config["compliance_level"] = "research_only"
        
        return base_config
    
    def generate_integration_spec(self) -> Dict[str, Any]:
        """
        Generate integration specification for acquirer.
        
        Documents how QuantraCore can integrate with their systems.
        """
        infrastructure = SUPPORTED_INFRASTRUCTURES.get(
            self.profile.target_infrastructure,
            SUPPORTED_INFRASTRUCTURES["custom"]
        )
        
        return {
            "integration_profile": self.profile.profile_name,
            "target_infrastructure": infrastructure.name,
            "created_at": self.profile.created_at.isoformat(),
            "data_integration": {
                "input_format": "OHLCV JSON (configurable)",
                "output_format": infrastructure.data_format,
                "field_mappings": self.profile.data_feed_mappings,
                "api_style": infrastructure.api_style,
            },
            "risk_integration": {
                "native_framework": infrastructure.risk_framework,
                "tier_mappings": {
                    tier: self.map_risk_tier(tier)
                    for tier in ["minimal", "low", "moderate", "high", "extreme"]
                },
            },
            "compliance_integration": {
                "supported_frameworks": infrastructure.compliance_requirements,
                "classification": "research_only",
                "required_disclaimers": [
                    "Not investment advice",
                    "Research and educational purposes only",
                    "Past performance does not guarantee future results",
                ],
            },
            "supported_features": {
                "protocols": 145,
                "asset_classes": self.profile.supported_asset_classes,
                "markets": self.profile.supported_markets,
                "real_time_streaming": True,
                "backtesting": True,
                "ml_integration": True,
            },
            "endpoints": self.profile.integration_endpoints,
            "validation_status": self.profile.validation_status,
        }
    
    def validate_profile(self) -> Dict[str, Any]:
        """
        Validate the adaptation profile configuration.
        
        Returns validation results and any issues found.
        """
        issues = []
        warnings = []
        
        if not self.profile.profile_name:
            issues.append("Profile name is required")
        
        if not self.profile.target_infrastructure:
            issues.append("Target infrastructure must be specified")
        
        if not self.profile.supported_asset_classes:
            warnings.append("No asset classes specified - using defaults")
        
        if not self.profile.data_feed_mappings:
            warnings.append("No data feed mappings - using native format")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            self.profile.validation_status = "validated"
        else:
            self.profile.validation_status = "invalid"
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }
