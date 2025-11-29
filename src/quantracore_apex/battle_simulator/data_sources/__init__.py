"""
Data Sources for Battle Simulator.

All data sources are 100% compliant, using only publicly available
information from SEC EDGAR and other official sources.
"""

from .sec_edgar import SECEdgarClient

__all__ = ["SECEdgarClient"]
