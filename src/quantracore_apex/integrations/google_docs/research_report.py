"""
Research Report Generator for QuantraCore Apex.

Automatically generates comprehensive research reports from scan results,
backtest analyses, and signal outputs, saving them to Google Docs.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.quantracore_apex.integrations.google_docs.client import google_docs_client

logger = logging.getLogger(__name__)


class ReportSection(BaseModel):
    """A section of the research report."""
    title: str
    content: str
    level: int = 2


class ResearchReportConfig(BaseModel):
    """Configuration for research report generation."""
    include_protocols: bool = True
    include_omega_alerts: bool = True
    include_risk_assessment: bool = True
    include_monster_runner: bool = True
    include_chart_analysis: bool = True
    include_recommendations: bool = True
    include_compliance_note: bool = True


class ResearchReportGenerator:
    """
    Generates professional research reports from QuantraCore Apex analysis.
    
    Creates structured Google Docs with:
    - Executive summary
    - Symbol analysis breakdown
    - Protocol activations
    - Risk assessment
    - Omega directive alerts
    - MonsterRunner detection
    - Recommendations
    """
    
    def __init__(self, config: Optional[ResearchReportConfig] = None):
        self.client = google_docs_client
        self.config = config or ResearchReportConfig()
    
    def _format_header(self, symbol: str, timestamp: datetime) -> str:
        """Generate report header."""
        return f"""
QUANTRACORE APEX RESEARCH REPORT
================================

Symbol: {symbol}
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Engine Version: v9.0-A Institutional Hardening

COMPLIANCE NOTICE: This report is for research and educational purposes only.
It does not constitute trading advice or recommendations.

"""
    
    def _format_executive_summary(self, scan_result: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        return f"""
EXECUTIVE SUMMARY
-----------------

QuantraScore: {scan_result.get('quantrascore', 'N/A')}/100 ({scan_result.get('score_bucket', 'N/A')})
Market Regime: {scan_result.get('regime', 'N/A')}
Risk Tier: {scan_result.get('risk_tier', 'N/A')}
Verdict: {scan_result.get('verdict_action', 'N/A')} (Confidence: {scan_result.get('verdict_confidence', 0):.1%})

Key States:
  - Entropy: {scan_result.get('entropy_state', 'N/A')}
  - Suppression: {scan_result.get('suppression_state', 'N/A')}
  - Drift: {scan_result.get('drift_state', 'N/A')}

Protocols Fired: {scan_result.get('protocol_fired_count', 0)} of 145
Omega Alerts: {len(scan_result.get('omega_alerts', []))} active

"""
    
    def _format_protocol_analysis(self, scan_result: Dict[str, Any]) -> str:
        """Generate protocol analysis section."""
        protocols = scan_result.get('protocol_fired_count', 0)
        omega_alerts = scan_result.get('omega_alerts', [])
        
        content = f"""
PROTOCOL ANALYSIS
-----------------

Total Protocols Evaluated: 145
  - Tier Protocols (T01-T80): 80 available
  - Learning Protocols (LP01-LP25): 25 available
  - MonsterRunner Protocols (MR01-MR20): 20 available
  - Omega Directives (Omega01-Omega20): 20 available

Protocols Triggered: {protocols}

"""
        if omega_alerts:
            content += "Active Omega Directives:\n"
            for alert in omega_alerts:
                content += f"  - {alert}\n"
        else:
            content += "No Omega Directive alerts triggered.\n"
        
        return content
    
    def _format_risk_assessment(self, risk_data: Optional[Dict[str, Any]]) -> str:
        """Generate risk assessment section."""
        if not risk_data:
            return "\nRISK ASSESSMENT\n---------------\nNo risk assessment data available.\n\n"
        
        assessment = risk_data.get('risk_assessment', {})
        return f"""
RISK ASSESSMENT
---------------

Risk Tier: {assessment.get('risk_tier', 'N/A')}
Permission: {assessment.get('permission', 'N/A')}
Max Position Size: {assessment.get('max_position_pct', 0):.1%}
Volatility Adjusted: {assessment.get('volatility_adjusted', False)}

Risk Factors:
{self._format_list(assessment.get('risk_factors', []))}

Denial Reasons:
{self._format_list(assessment.get('denial_reasons', []) or ['None'])}

"""
    
    def _format_monster_runner(self, monster_data: Optional[Dict[str, Any]]) -> str:
        """Generate MonsterRunner analysis section."""
        if not monster_data:
            return "\nMONSTERRUNNER ANALYSIS\n----------------------\nNo MonsterRunner data available.\n\n"
        
        metrics = monster_data.get('metrics', {})
        return f"""
MONSTERRUNNER ANALYSIS
----------------------

Runner Probability: {monster_data.get('runner_probability', 0):.1%}
Runner State: {monster_data.get('runner_state', 'N/A')}
Rare Event Class: {monster_data.get('rare_event_class', 'N/A')}

Detection Metrics:
  - Compression Trace: {metrics.get('compression_trace', 'N/A')}
  - Entropy Floor: {metrics.get('entropy_floor', 'N/A')}
  - Volume Pulse: {metrics.get('volume_pulse', 'N/A')}
  - Range Contraction: {metrics.get('range_contraction', 'N/A')}
  - Primed Confidence: {metrics.get('primed_confidence', 0):.2f}

"""
    
    def _format_signal(self, signal_data: Optional[Dict[str, Any]]) -> str:
        """Generate signal analysis section."""
        if not signal_data:
            return "\nSIGNAL ANALYSIS\n---------------\nNo signal data available.\n\n"
        
        signal = signal_data.get('signal', {})
        return f"""
SIGNAL ANALYSIS
---------------

Signal Type: {signal.get('signal_type', 'N/A')}
Direction: {signal.get('direction', 'N/A')}
Strength: {signal.get('strength', 'N/A')}
Entry Price: {signal.get('entry_price', 'N/A')}
Stop Loss: {signal.get('stop_loss', 'N/A')}
Take Profit: {signal.get('take_profit', 'N/A')}

Risk-Reward Ratio: {signal.get('risk_reward_ratio', 'N/A')}
Time Horizon: {signal.get('time_horizon', 'N/A')}

"""
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items."""
        if not items:
            return "  - None\n"
        return "\n".join(f"  - {item}" for item in items) + "\n"
    
    def _format_footer(self) -> str:
        """Generate report footer."""
        return f"""
--------------------------------------------------------------------------------

DISCLAIMER
----------

This report was generated by QuantraCore Apex v9.0-A, an institutional-grade
deterministic AI trading intelligence engine. All analysis is based on
structural patterns and statistical probabilities.

THIS IS NOT TRADING ADVICE. Past performance does not guarantee future results.
All trading involves risk. Consult with a qualified financial advisor before
making any investment decisions.

QuantraCore Apex operates in RESEARCH ONLY mode. Live trading is disabled
by default per Omega Directive 4 (O4-ComplianceMode).

Generated by Lamont Labs | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Window Hash: {{window_hash}}

--------------------------------------------------------------------------------
"""
    
    async def generate_report(
        self,
        symbol: str,
        scan_result: Dict[str, Any],
        risk_data: Optional[Dict[str, Any]] = None,
        monster_data: Optional[Dict[str, Any]] = None,
        signal_data: Optional[Dict[str, Any]] = None,
        custom_sections: Optional[List[ReportSection]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive research report and save to Google Docs.
        
        Args:
            symbol: Trading symbol analyzed
            scan_result: Results from scan_symbol endpoint
            risk_data: Optional risk assessment data
            monster_data: Optional MonsterRunner analysis
            signal_data: Optional signal generation data
            custom_sections: Optional additional sections
            
        Returns:
            dict: Document metadata including documentId and URL
        """
        timestamp = datetime.utcnow()
        title = f"QuantraCore Research Report - {symbol} - {timestamp.strftime('%Y%m%d_%H%M')}"
        
        report_content = self._format_header(symbol, timestamp)
        report_content += self._format_executive_summary(scan_result)
        
        if self.config.include_protocols:
            report_content += self._format_protocol_analysis(scan_result)
        
        if self.config.include_risk_assessment:
            report_content += self._format_risk_assessment(risk_data)
        
        if self.config.include_monster_runner:
            report_content += self._format_monster_runner(monster_data)
        
        if signal_data:
            report_content += self._format_signal(signal_data)
        
        if custom_sections:
            for section in custom_sections:
                report_content += f"\n{section.title}\n"
                report_content += "-" * len(section.title) + "\n\n"
                report_content += section.content + "\n"
        
        footer = self._format_footer().replace(
            "{window_hash}",
            scan_result.get('window_hash', 'N/A')
        )
        report_content += footer
        
        document = await self.client.create_document(title)
        document_id = document.get("documentId")
        
        await self.client.insert_text(document_id, report_content)
        
        document_url = f"https://docs.google.com/document/d/{document_id}/edit"
        
        logger.info(f"Generated research report for {symbol}: {document_url}")
        
        return {
            "document_id": document_id,
            "title": title,
            "url": document_url,
            "symbol": symbol,
            "generated_at": timestamp.isoformat(),
            "sections_included": {
                "executive_summary": True,
                "protocol_analysis": self.config.include_protocols,
                "risk_assessment": self.config.include_risk_assessment,
                "monster_runner": self.config.include_monster_runner,
                "signal_analysis": signal_data is not None,
                "custom_sections": len(custom_sections or [])
            }
        }
    
    async def generate_batch_report(
        self,
        scan_results: List[Dict[str, Any]],
        title_prefix: str = "QuantraCore Universe Scan"
    ) -> Dict[str, Any]:
        """
        Generate a batch report for multiple symbols.
        
        Args:
            scan_results: List of scan results from scan_universe
            title_prefix: Prefix for document title
            
        Returns:
            dict: Document metadata
        """
        timestamp = datetime.utcnow()
        title = f"{title_prefix} - {timestamp.strftime('%Y%m%d_%H%M')}"
        
        content = f"""
QUANTRACORE APEX UNIVERSE SCAN REPORT
=====================================

Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Total Symbols Analyzed: {len(scan_results)}
Engine Version: v9.0-A Institutional Hardening

COMPLIANCE NOTICE: Research and educational purposes only.

"""
        
        sorted_results = sorted(
            scan_results,
            key=lambda x: x.get('quantrascore', 0),
            reverse=True
        )
        
        content += "\nTOP OPPORTUNITIES BY QUANTRASCORE\n"
        content += "-" * 40 + "\n\n"
        
        for i, result in enumerate(sorted_results[:10], 1):
            content += f"{i}. {result.get('symbol', 'N/A')}\n"
            content += f"   QuantraScore: {result.get('quantrascore', 0):.1f}/100\n"
            content += f"   Regime: {result.get('regime', 'N/A')}\n"
            content += f"   Risk Tier: {result.get('risk_tier', 'N/A')}\n"
            content += f"   Verdict: {result.get('verdict_action', 'N/A')}\n\n"
        
        content += "\nFULL SCAN SUMMARY\n"
        content += "-" * 40 + "\n\n"
        
        for result in sorted_results:
            alerts = result.get('omega_alerts', [])
            alert_str = f" [{len(alerts)} alerts]" if alerts else ""
            content += f"{result.get('symbol', 'N/A')}: "
            content += f"{result.get('quantrascore', 0):.1f} | "
            content += f"{result.get('regime', 'N/A')} | "
            content += f"{result.get('verdict_action', 'N/A')}{alert_str}\n"
        
        content += self._format_footer().replace("{window_hash}", "BATCH_SCAN")
        
        document = await self.client.create_document(title)
        document_id = document.get("documentId")
        await self.client.insert_text(document_id, content)
        
        return {
            "document_id": document_id,
            "title": title,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "symbols_analyzed": len(scan_results),
            "generated_at": timestamp.isoformat()
        }


research_report_generator = ResearchReportGenerator()
