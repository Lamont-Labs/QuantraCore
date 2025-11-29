"""
Configuration for QuantraCore Apex Broker Layer.

Loads broker configuration from YAML and environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path
import yaml

from .enums import ExecutionMode


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_notional_exposure_usd: float = 50_000
    max_position_notional_per_symbol_usd: float = 5_000
    max_positions: int = 30
    max_daily_turnover_usd: float = 100_000
    max_order_notional_usd: float = 3_000
    max_leverage: float = 2.0
    block_short_selling: bool = True
    block_margin: bool = True
    require_positive_equity: bool = True
    per_trade_risk_fraction: float = 0.01  # 1% of equity per trade


@dataclass
class AlpacaConfig:
    """Alpaca broker configuration."""
    base_url: str = "https://paper-api.alpaca.markets"
    data_base_url: str = "https://data.alpaca.markets"
    api_key: str = ""
    api_secret: str = ""
    enabled: bool = True
    
    def load_from_env(self, key_env: str = "ALPACA_PAPER_API_KEY", 
                      secret_env: str = "ALPACA_PAPER_API_SECRET"):
        """Load API credentials from environment variables."""
        self.api_key = os.environ.get(key_env, "")
        self.api_secret = os.environ.get(secret_env, "")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    execution_log_dir: str = "logs/execution/"
    audit_log_dir: str = "logs/execution_audit/"
    broker_raw_dir: str = "logs/execution_broker_raw/"
    log_level: str = "INFO"


@dataclass
class AccountConfig:
    """Account configuration."""
    name: str = "primary"
    description: str = "Default account for paper/live"
    broker: str = "ALPACA"
    mode: str = "PAPER"
    risk_profile: str = "standard"


@dataclass
class BrokerConfig:
    """Complete broker layer configuration."""
    execution_mode: ExecutionMode = ExecutionMode.RESEARCH
    default_account: str = "primary"
    accounts: Dict[str, AccountConfig] = field(default_factory=dict)
    alpaca_paper: AlpacaConfig = field(default_factory=AlpacaConfig)
    alpaca_live: AlpacaConfig = field(default_factory=lambda: AlpacaConfig(
        base_url="https://api.alpaca.markets",
        enabled=False
    ))
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        if not self.accounts:
            self.accounts = {
                "primary": AccountConfig()
            }
    
    @property
    def is_paper_mode(self) -> bool:
        return self.execution_mode == ExecutionMode.PAPER
    
    @property
    def is_research_mode(self) -> bool:
        return self.execution_mode == ExecutionMode.RESEARCH
    
    @property
    def is_live_mode(self) -> bool:
        return self.execution_mode == ExecutionMode.LIVE
    
    def get_active_alpaca_config(self) -> AlpacaConfig:
        """Get the appropriate Alpaca config based on mode."""
        if self.is_live_mode and self.alpaca_live.enabled:
            return self.alpaca_live
        return self.alpaca_paper
    
    def to_dict(self) -> Dict:
        return {
            "execution_mode": self.execution_mode.value,
            "default_account": self.default_account,
            "alpaca_paper_configured": self.alpaca_paper.is_configured,
            "alpaca_live_enabled": self.alpaca_live.enabled,
            "risk": {
                "max_notional_exposure_usd": self.risk.max_notional_exposure_usd,
                "max_positions": self.risk.max_positions,
                "block_short_selling": self.risk.block_short_selling,
            }
        }


def load_broker_config(config_path: str = "config/broker.yaml") -> BrokerConfig:
    """
    Load broker configuration from YAML file.
    
    Falls back to defaults if file doesn't exist.
    """
    config = BrokerConfig()
    path = Path(config_path)
    
    if path.exists():
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Load execution settings
        execution = data.get("execution", {})
        mode_str = execution.get("mode", "RESEARCH").upper()
        try:
            config.execution_mode = ExecutionMode(mode_str)
        except ValueError:
            config.execution_mode = ExecutionMode.RESEARCH
        
        config.default_account = execution.get("default_account", "primary")
        
        # Load accounts
        accounts_data = data.get("accounts", {})
        for name, acc_data in accounts_data.items():
            config.accounts[name] = AccountConfig(
                name=name,
                description=acc_data.get("description", ""),
                broker=acc_data.get("broker", "ALPACA"),
                mode=acc_data.get("mode", "PAPER"),
                risk_profile=acc_data.get("risk_profile", "standard"),
            )
        
        # Load Alpaca settings
        brokers = data.get("brokers", {})
        alpaca = brokers.get("alpaca", {})
        
        paper = alpaca.get("paper", {})
        config.alpaca_paper.base_url = paper.get("base_url", config.alpaca_paper.base_url)
        config.alpaca_paper.data_base_url = paper.get("data_base_url", config.alpaca_paper.data_base_url)
        config.alpaca_paper.enabled = paper.get("enabled", True)
        
        # Load from environment variables
        key_env = paper.get("api_key_env", "ALPACA_PAPER_API_KEY")
        secret_env = paper.get("api_secret_env", "ALPACA_PAPER_API_SECRET")
        config.alpaca_paper.load_from_env(key_env, secret_env)
        
        live = alpaca.get("live", {})
        config.alpaca_live.base_url = live.get("base_url", config.alpaca_live.base_url)
        config.alpaca_live.enabled = live.get("enabled", False)
        
        if config.alpaca_live.enabled:
            live_key_env = live.get("api_key_env", "ALPACA_LIVE_API_KEY")
            live_secret_env = live.get("api_secret_env", "ALPACA_LIVE_API_SECRET")
            config.alpaca_live.load_from_env(live_key_env, live_secret_env)
        
        # Load risk settings
        risk = data.get("risk", {})
        config.risk.max_notional_exposure_usd = risk.get("max_notional_exposure_usd", 50_000)
        config.risk.max_position_notional_per_symbol_usd = risk.get("max_position_notional_per_symbol_usd", 5_000)
        config.risk.max_positions = risk.get("max_positions", 30)
        config.risk.max_daily_turnover_usd = risk.get("max_daily_turnover_usd", 100_000)
        config.risk.max_order_notional_usd = risk.get("max_order_notional_usd", 3_000)
        config.risk.max_leverage = risk.get("max_leverage", 2.0)
        config.risk.block_short_selling = risk.get("block_short_selling", True)
        config.risk.block_margin = risk.get("block_margin", True)
        config.risk.require_positive_equity = risk.get("require_positive_equity", True)
        
        # Load logging settings
        logging_cfg = data.get("logging", {})
        config.logging.execution_log_dir = logging_cfg.get("execution_log_dir", "logs/execution/")
        config.logging.audit_log_dir = logging_cfg.get("audit_log_dir", "logs/execution_audit/")
        config.logging.broker_raw_dir = logging_cfg.get("broker_raw_dir", "logs/execution_broker_raw/")
        config.logging.log_level = logging_cfg.get("log_level", "INFO")
    
    else:
        # Load Alpaca credentials from default env vars
        config.alpaca_paper.load_from_env()
    
    return config


def create_default_config_file(config_path: str = "config/broker.yaml"):
    """Create a default broker configuration file."""
    default_config = """
# QuantraCore Apex Broker Configuration
# SAFETY: Live trading is DISABLED by default. Paper trading only.

execution:
  mode: "RESEARCH"  # RESEARCH | PAPER | LIVE
  default_account: "primary"

accounts:
  primary:
    description: "Default account for paper/live"
    broker: "ALPACA"
    mode: "PAPER"  # LIVE not allowed until explicitly enabled
    risk_profile: "standard"

brokers:
  alpaca:
    paper:
      base_url: "https://paper-api.alpaca.markets"
      api_key_env: "ALPACA_PAPER_API_KEY"
      api_secret_env: "ALPACA_PAPER_API_SECRET"
      data_base_url: "https://data.alpaca.markets"
      enabled: true
    live:
      base_url: "https://api.alpaca.markets"
      api_key_env: "ALPACA_LIVE_API_KEY"
      api_secret_env: "ALPACA_LIVE_API_SECRET"
      enabled: false  # Must remain false until institution flips

risk:
  max_notional_exposure_usd: 50000
  max_position_notional_per_symbol_usd: 5000
  max_positions: 30
  max_daily_turnover_usd: 100000
  max_order_notional_usd: 3000
  max_leverage: 2.0
  block_short_selling: true
  block_margin: true
  require_positive_equity: true

logging:
  execution_log_dir: "logs/execution/"
  audit_log_dir: "logs/execution_audit/"
  log_level: "INFO"
"""
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(default_config.strip())
    
    return path
