"""
T45 - Money Flow Index Protocol

Analyzes MFI for overbought/oversold conditions with volume.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T45: Analyze Money Flow Index.
    
    Fires when MFI indicates extreme conditions.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T45",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    typical_prices = np.array([(b.high + b.low + b.close) / 3 for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    raw_money_flow = typical_prices * volumes
    
    positive_mf = 0
    negative_mf = 0
    
    period = 14
    for i in range(-period, 0):
        if i == -period:
            continue
        if typical_prices[i] > typical_prices[i-1]:
            positive_mf += raw_money_flow[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_mf += raw_money_flow[i]
    
    if negative_mf == 0:
        mfi = 100
    else:
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
    
    mfi_values = []
    for offset in range(1, 6):
        pm, nm = 0, 0
        start_idx = -period - offset
        end_idx = -offset
        for i in range(start_idx, end_idx):
            if i == start_idx:
                continue
            if len(typical_prices) > abs(i) and len(typical_prices) > abs(i-1):
                if typical_prices[i] > typical_prices[i-1]:
                    pm += raw_money_flow[i]
                elif typical_prices[i] < typical_prices[i-1]:
                    nm += raw_money_flow[i]
        if nm > 0:
            mfi_values.append(100 - (100 / (1 + pm/nm)))
    
    mfi_trend = mfi - np.mean(mfi_values) if mfi_values else 0
    
    if mfi > 80:
        signal_type = "overbought"
        confidence = min(0.9, 0.6 + (mfi - 80) * 0.015)
        fired = True
    elif mfi < 20:
        signal_type = "oversold"
        confidence = min(0.9, 0.6 + (20 - mfi) * 0.015)
        fired = True
    elif abs(mfi_trend) > 15:
        signal_type = "mfi_momentum_shift"
        confidence = 0.65
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T45",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "mfi": float(mfi),
            "mfi_trend": float(mfi_trend),
            "positive_mf": float(positive_mf),
            "negative_mf": float(negative_mf),
        }
    )
