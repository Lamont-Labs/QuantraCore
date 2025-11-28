"""
Stub protocol definitions for T21-T80.

These protocols are defined but not yet fully implemented.
They return structured no-op results for pipeline compatibility.
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


STUB_PROTOCOLS = {
    "T21": {"name": "Support Level Detection", "category": "Support/Resistance"},
    "T22": {"name": "Resistance Level Detection", "category": "Support/Resistance"},
    "T23": {"name": "Pivot Point Analysis", "category": "Support/Resistance"},
    "T24": {"name": "Fibonacci Retracement", "category": "Support/Resistance"},
    "T25": {"name": "Fibonacci Extension", "category": "Support/Resistance"},
    "T26": {"name": "Gap Analysis - Up", "category": "Gap Patterns"},
    "T27": {"name": "Gap Analysis - Down", "category": "Gap Patterns"},
    "T28": {"name": "Gap Fill Detection", "category": "Gap Patterns"},
    "T29": {"name": "Island Reversal", "category": "Gap Patterns"},
    "T30": {"name": "Exhaustion Gap", "category": "Gap Patterns"},
    "T31": {"name": "Breakout Detection", "category": "Breakout Patterns"},
    "T32": {"name": "Breakdown Detection", "category": "Breakout Patterns"},
    "T33": {"name": "False Breakout", "category": "Breakout Patterns"},
    "T34": {"name": "Range Breakout", "category": "Breakout Patterns"},
    "T35": {"name": "Consolidation Break", "category": "Breakout Patterns"},
    "T36": {"name": "RSI Divergence Bullish", "category": "Divergence"},
    "T37": {"name": "RSI Divergence Bearish", "category": "Divergence"},
    "T38": {"name": "MACD Divergence", "category": "Divergence"},
    "T39": {"name": "Volume Divergence", "category": "Divergence"},
    "T40": {"name": "Price-Momentum Divergence", "category": "Divergence"},
    "T41": {"name": "Bullish Engulfing", "category": "Candlestick Patterns"},
    "T42": {"name": "Bearish Engulfing", "category": "Candlestick Patterns"},
    "T43": {"name": "Doji Star", "category": "Candlestick Patterns"},
    "T44": {"name": "Hammer/Hanging Man", "category": "Candlestick Patterns"},
    "T45": {"name": "Morning/Evening Star", "category": "Candlestick Patterns"},
    "T46": {"name": "Accumulation Detection", "category": "Market Structure"},
    "T47": {"name": "Distribution Detection", "category": "Market Structure"},
    "T48": {"name": "Wyckoff Spring", "category": "Market Structure"},
    "T49": {"name": "Wyckoff Upthrust", "category": "Market Structure"},
    "T50": {"name": "Market Structure Break", "category": "Market Structure"},
    "T51": {"name": "Volume Profile Analysis", "category": "Advanced Volume"},
    "T52": {"name": "VWAP Deviation", "category": "Advanced Volume"},
    "T53": {"name": "OBV Trend", "category": "Advanced Volume"},
    "T54": {"name": "Money Flow Index", "category": "Advanced Volume"},
    "T55": {"name": "Volume Weighted Trend", "category": "Advanced Volume"},
    "T56": {"name": "Momentum Oscillator", "category": "Momentum Analysis"},
    "T57": {"name": "Rate of Change Extreme", "category": "Momentum Analysis"},
    "T58": {"name": "Stochastic Extreme", "category": "Momentum Analysis"},
    "T59": {"name": "Williams %R Signal", "category": "Momentum Analysis"},
    "T60": {"name": "CCI Signal", "category": "Momentum Analysis"},
    "T61": {"name": "Sector Rotation", "category": "Sector Analysis"},
    "T62": {"name": "Relative Strength vs Sector", "category": "Sector Analysis"},
    "T63": {"name": "Sector Leadership", "category": "Sector Analysis"},
    "T64": {"name": "Sector Divergence", "category": "Sector Analysis"},
    "T65": {"name": "Sector Momentum", "category": "Sector Analysis"},
    "T66": {"name": "Phase Compression Precursor", "category": "MonsterRunner Precursors"},
    "T67": {"name": "Volume Ignition Setup", "category": "MonsterRunner Precursors"},
    "T68": {"name": "Range Flip Potential", "category": "MonsterRunner Precursors"},
    "T69": {"name": "Entropy Collapse Signal", "category": "MonsterRunner Precursors"},
    "T70": {"name": "Sector Cascade Risk", "category": "MonsterRunner Precursors"},
    "T71": {"name": "Volatility Regime Shift", "category": "Regime Detection"},
    "T72": {"name": "Trend Regime Change", "category": "Regime Detection"},
    "T73": {"name": "Range to Trend Transition", "category": "Regime Detection"},
    "T74": {"name": "Trend to Range Transition", "category": "Regime Detection"},
    "T75": {"name": "Regime Uncertainty", "category": "Regime Detection"},
    "T76": {"name": "Multi-Timeframe Alignment", "category": "Multi-Timeframe"},
    "T77": {"name": "Higher Timeframe Trend", "category": "Multi-Timeframe"},
    "T78": {"name": "Lower Timeframe Entry", "category": "Multi-Timeframe"},
    "T79": {"name": "Timeframe Divergence", "category": "Multi-Timeframe"},
    "T80": {"name": "Composite Timeframe Score", "category": "Multi-Timeframe"},
}


def create_stub_run(protocol_id: str, info: dict):
    """Create a stub run function for a protocol."""
    def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
        return ProtocolResult(
            protocol_id=protocol_id,
            fired=False,
            confidence=0.0,
            signal_type=None,
            details={
                "status": "stub",
                "name": info["name"],
                "category": info["category"],
                "message": f"{protocol_id} ({info['name']}) - not yet implemented"
            }
        )
    return run
