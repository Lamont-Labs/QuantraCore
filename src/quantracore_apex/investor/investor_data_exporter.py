"""
Investor Data Export System.

Automated export of investor due diligence packages:
- Weekly snapshots
- Monthly performance packages
- Quarterly due diligence bundles
- On-demand data room exports

Provides everything an institutional investor needs in organized packages.
"""

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

EXPORT_DIR = Path("investor_logs/exports")
INVESTOR_DOCS_DIR = Path("docs/investor")
LEGAL_DOCS_DIR = Path("docs/investor/legal")


class InvestorDataExporter:
    """
    Comprehensive investor data export system.
    
    Creates organized packages containing all data and documents
    an institutional investor would need for due diligence.
    """
    
    def __init__(self):
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    def export_weekly_snapshot(self) -> Path:
        """
        Export weekly investor snapshot.
        
        Includes:
        - Week's trading activity
        - Performance metrics
        - Risk summary
        - Attestations
        """
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        export_name = f"weekly_snapshot_{now.strftime('%Y%m%d')}"
        export_path = EXPORT_DIR / export_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "export_type": "weekly_snapshot",
            "generated_at": now.isoformat(),
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": now.strftime("%Y-%m-%d"),
            "contents": [],
        }
        
        self._export_performance_data(export_path, manifest, days=7)
        self._export_trading_data(export_path, manifest, days=7)
        self._export_attestations(export_path, manifest, days=7)
        self._export_risk_summary(export_path, manifest)
        
        manifest_path = export_path / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        zip_path = self._create_zip(export_path, export_name)
        
        logger.info(f"Weekly snapshot exported to {zip_path}")
        return zip_path
    
    def export_monthly_package(self, year: int, month: int) -> Path:
        """
        Export monthly investor package.
        
        Includes:
        - Monthly performance report
        - All trading activity
        - Risk metrics
        - Model performance
        - Compliance attestations
        """
        now = datetime.now(timezone.utc)
        export_name = f"monthly_package_{year}{month:02d}"
        export_path = EXPORT_DIR / export_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "export_type": "monthly_package",
            "generated_at": now.isoformat(),
            "month": month,
            "year": year,
            "contents": [],
        }
        
        self._export_monthly_performance(export_path, manifest, year, month)
        self._export_trading_data(export_path, manifest, days=31)
        self._export_attestations(export_path, manifest, days=31)
        self._export_model_metrics(export_path, manifest)
        self._export_risk_summary(export_path, manifest)
        
        manifest_path = export_path / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        zip_path = self._create_zip(export_path, export_name)
        
        logger.info(f"Monthly package exported to {zip_path}")
        return zip_path
    
    def export_full_data_room(self) -> Path:
        """
        Export complete investor data room.
        
        Includes EVERYTHING an investor needs:
        - All legal documents
        - Complete trading history
        - Full performance history
        - All compliance records
        - Model documentation
        - Technical documentation
        """
        now = datetime.now(timezone.utc)
        export_name = f"full_data_room_{now.strftime('%Y%m%d_%H%M%S')}"
        export_path = EXPORT_DIR / export_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "export_type": "full_data_room",
            "generated_at": now.isoformat(),
            "sections": [],
            "document_count": 0,
        }
        
        self._export_section_legal(export_path, manifest)
        self._export_section_performance(export_path, manifest)
        self._export_section_trading(export_path, manifest)
        self._export_section_compliance(export_path, manifest)
        self._export_section_risk(export_path, manifest)
        self._export_section_models(export_path, manifest)
        self._export_section_technical(export_path, manifest)
        self._export_section_team(export_path, manifest)
        
        self._create_data_room_index(export_path, manifest)
        
        manifest_path = export_path / "MANIFEST.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        zip_path = self._create_zip(export_path, export_name)
        
        logger.info(f"Full data room exported to {zip_path}")
        return zip_path
    
    def _export_section_legal(self, export_path: Path, manifest: Dict):
        """Export legal documents section."""
        section_path = export_path / "01_Legal"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        if LEGAL_DOCS_DIR.exists():
            for doc in LEGAL_DOCS_DIR.glob("*.md"):
                shutil.copy(doc, section_path / doc.name)
                documents.append(doc.name)
        
        risk_disclosures = INVESTOR_DOCS_DIR / "33_LIMITATIONS_AND_HONEST_RISKS.md"
        if risk_disclosures.exists():
            shutil.copy(risk_disclosures, section_path / "RISK_LIMITATIONS.md")
            documents.append("RISK_LIMITATIONS.md")
        
        compliance_doc = INVESTOR_DOCS_DIR / "31_COMPLIANCE_AND_USAGE_POLICIES.md"
        if compliance_doc.exists():
            shutil.copy(compliance_doc, section_path / "COMPLIANCE_POLICIES.md")
            documents.append("COMPLIANCE_POLICIES.md")
        
        manifest["sections"].append({
            "name": "Legal",
            "path": "01_Legal",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_performance(self, export_path: Path, manifest: Dict):
        """Export performance data section."""
        section_path = export_path / "02_Performance"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        perf_logs = Path("investor_logs/performance")
        if perf_logs.exists():
            for log in perf_logs.glob("*.json*"):
                shutil.copy(log, section_path / log.name)
                documents.append(log.name)
        
        summaries = Path("investor_logs/summaries")
        if summaries.exists():
            for summary in summaries.glob("*.json"):
                shutil.copy(summary, section_path / summary.name)
                documents.append(summary.name)
        
        manifest["sections"].append({
            "name": "Performance",
            "path": "02_Performance",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_trading(self, export_path: Path, manifest: Dict):
        """Export trading history section."""
        section_path = export_path / "03_Trading"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        trades_dir = Path("investor_logs/trades")
        if trades_dir.exists():
            for log in trades_dir.glob("*"):
                if log.is_file():
                    shutil.copy(log, section_path / log.name)
                    documents.append(log.name)
        
        manifest["sections"].append({
            "name": "Trading",
            "path": "03_Trading",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_compliance(self, export_path: Path, manifest: Dict):
        """Export compliance records section."""
        section_path = export_path / "04_Compliance"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        compliance_dir = Path("investor_logs/compliance")
        if compliance_dir.exists():
            for subdir in compliance_dir.iterdir():
                if subdir.is_dir():
                    for log in subdir.glob("*"):
                        if log.is_file():
                            dest_name = f"{subdir.name}_{log.name}"
                            shutil.copy(log, section_path / dest_name)
                            documents.append(dest_name)
        
        manifest["sections"].append({
            "name": "Compliance",
            "path": "04_Compliance",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_risk(self, export_path: Path, manifest: Dict):
        """Export risk management section."""
        section_path = export_path / "05_Risk"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        risk_doc = INVESTOR_DOCS_DIR / "30_RISK_MANAGEMENT_AND_GUARDS.md"
        if risk_doc.exists():
            shutil.copy(risk_doc, section_path / "RISK_MANAGEMENT.md")
            documents.append("RISK_MANAGEMENT.md")
        
        audit_dir = Path("investor_logs/audit")
        if audit_dir.exists():
            for subdir in audit_dir.iterdir():
                if subdir.is_dir():
                    for log in subdir.glob("*"):
                        if log.is_file():
                            dest_name = f"{subdir.name}_{log.name}"
                            shutil.copy(log, section_path / dest_name)
                            documents.append(dest_name)
        
        manifest["sections"].append({
            "name": "Risk",
            "path": "05_Risk",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_models(self, export_path: Path, manifest: Dict):
        """Export ML model documentation section."""
        section_path = export_path / "06_Models"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        model_docs = [
            "12_APEXLAB_AND_APEXCORE_MODELS.md",
            "13_PREDICTIVE_LAYER_AND_SAFETY.md",
            "21_LABELING_METHODS_AND_LEAKAGE_GUARDS.md",
            "22_TRAINING_PROCESS_AND_HYPERPARAMS.md",
            "23_EVALUATION_AND_LIMITATIONS.md",
            "24_MONSTERRUNNER_EXPLAINED.md",
        ]
        
        for doc in model_docs:
            doc_path = INVESTOR_DOCS_DIR / doc
            if doc_path.exists():
                shutil.copy(doc_path, section_path / doc)
                documents.append(doc)
        
        model_logs = Path("investor_logs/models")
        if model_logs.exists():
            for log in model_logs.glob("*"):
                if log.is_file():
                    shutil.copy(log, section_path / log.name)
                    documents.append(log.name)
        
        manifest["sections"].append({
            "name": "Models",
            "path": "06_Models",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_technical(self, export_path: Path, manifest: Dict):
        """Export technical architecture section."""
        section_path = export_path / "07_Technical"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        tech_docs = [
            "10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md",
            "11_ENGINE_AND_PROTOCOLS.md",
            "14_BROKER_AND_EXECUTION_ENVELOPE.md",
            "15_APEXDESK_UI_AND_APIS.md",
            "20_DATA_SOURCES_AND_UNIVERSE.md",
            "32_SECURITY_AND_PROVENANCE.md",
            "50_ENGINEERING_OVERVIEW_AND_PRACTICES.md",
            "51_TESTING_AND_COVERAGE_SUMMARY.md",
            "52_OPERATIONS_AND_RUNBOOKS.md",
        ]
        
        for doc in tech_docs:
            doc_path = INVESTOR_DOCS_DIR / doc
            if doc_path.exists():
                shutil.copy(doc_path, section_path / doc)
                documents.append(doc)
        
        manifest["sections"].append({
            "name": "Technical",
            "path": "07_Technical",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_section_team(self, export_path: Path, manifest: Dict):
        """Export team/company section."""
        section_path = export_path / "08_Company"
        section_path.mkdir(parents=True, exist_ok=True)
        
        documents = []
        
        company_docs = [
            "00_EXEC_SUMMARY.md",
            "01_ONE_PAGER.md",
            "02_FOUNDER_PROFILE.md",
            "03_PLATFORM_OVERVIEW.md",
            "04_COMPETITIVE_POSITIONING.md",
            "05_DEAL_SUMMARY.md",
            "40_COMMERCIAL_MODELS_AND_PATHS.md",
            "41_TARGET_CUSTOMERS_AND_SEGMENTS.md",
            "42_ROADMAP_AND_CAPITAL_USE.md",
            "60_INVESTOR_FAQ.md",
        ]
        
        for doc in company_docs:
            doc_path = INVESTOR_DOCS_DIR / doc
            if doc_path.exists():
                shutil.copy(doc_path, section_path / doc)
                documents.append(doc)
        
        manifest["sections"].append({
            "name": "Company",
            "path": "08_Company",
            "documents": documents,
        })
        manifest["document_count"] += len(documents)
    
    def _export_performance_data(self, export_path: Path, manifest: Dict, days: int):
        """Export recent performance data."""
        perf_dir = export_path / "performance"
        perf_dir.mkdir(exist_ok=True)
        
        source = Path("investor_logs/performance")
        if source.exists():
            for f in source.glob("*.json*"):
                shutil.copy(f, perf_dir / f.name)
        
        manifest["contents"].append({"type": "performance", "path": "performance"})
    
    def _export_trading_data(self, export_path: Path, manifest: Dict, days: int):
        """Export recent trading data."""
        trades_dir = export_path / "trades"
        trades_dir.mkdir(exist_ok=True)
        
        source = Path("investor_logs/trades")
        if source.exists():
            for f in source.glob("*"):
                if f.is_file():
                    shutil.copy(f, trades_dir / f.name)
        
        manifest["contents"].append({"type": "trades", "path": "trades"})
    
    def _export_attestations(self, export_path: Path, manifest: Dict, days: int):
        """Export recent attestations."""
        attest_dir = export_path / "attestations"
        attest_dir.mkdir(exist_ok=True)
        
        source = Path("investor_logs/compliance/attestations")
        if source.exists():
            for f in source.glob("*"):
                if f.is_file():
                    shutil.copy(f, attest_dir / f.name)
        
        manifest["contents"].append({"type": "attestations", "path": "attestations"})
    
    def _export_risk_summary(self, export_path: Path, manifest: Dict):
        """Export risk summary."""
        risk_summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "risk_limits": {
                "max_exposure": 100000,
                "max_per_symbol": 10000,
                "max_positions": 50,
                "max_leverage": 4.0,
            },
            "current_status": "operational",
        }
        
        with open(export_path / "risk_summary.json", "w") as f:
            json.dump(risk_summary, f, indent=2)
        
        manifest["contents"].append({"type": "risk", "path": "risk_summary.json"})
    
    def _export_monthly_performance(self, export_path: Path, manifest: Dict, year: int, month: int):
        """Export monthly performance summary."""
        from src.quantracore_apex.investor.performance_logger import get_performance_logger
        
        perf_dir = export_path / "performance"
        perf_dir.mkdir(exist_ok=True)
        
        try:
            perf_logger = get_performance_logger()
            summary = perf_logger.generate_monthly_summary(year, month)
            if summary:
                with open(perf_dir / f"monthly_summary_{year}{month:02d}.json", "w") as f:
                    json.dump(summary.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not generate monthly summary: {e}")
        
        manifest["contents"].append({"type": "monthly_performance", "path": "performance"})
    
    def _export_model_metrics(self, export_path: Path, manifest: Dict):
        """Export model performance metrics."""
        models_dir = export_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        source = Path("investor_logs/models")
        if source.exists():
            for f in source.glob("*"):
                if f.is_file():
                    shutil.copy(f, models_dir / f.name)
        
        manifest["contents"].append({"type": "models", "path": "models"})
    
    def _create_data_room_index(self, export_path: Path, manifest: Dict):
        """Create data room index document."""
        index_content = f"""# QuantraCore Apex — Investor Data Room

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Documents:** {manifest['document_count']}

---

## Contents

"""
        
        for section in manifest["sections"]:
            index_content += f"### {section['name']}\n\n"
            index_content += f"Location: `{section['path']}/`\n\n"
            for doc in section["documents"]:
                index_content += f"- {doc}\n"
            index_content += "\n"
        
        index_content += """---

## About This Data Room

This data room contains all documentation and records necessary for 
institutional investor due diligence on QuantraCore Apex.

### Document Categories

1. **Legal** - Terms of service, risk disclosures, privacy policy
2. **Performance** - Historical returns, daily P&L, risk-adjusted metrics
3. **Trading** - Complete trade logs with execution details
4. **Compliance** - Attestations, incident records, policy acknowledgments
5. **Risk** - Risk management documentation, reconciliation records
6. **Models** - ML model documentation, training history, validation results
7. **Technical** - System architecture, security, operations
8. **Company** - Team profiles, business overview, roadmap

### Contact

For questions about this data room:
- Email: investors@quantracore.io

---

*This data room is provided for informational purposes only and does not constitute investment advice.*
"""
        
        with open(export_path / "INDEX.md", "w") as f:
            f.write(index_content)
    
    def _create_zip(self, source_path: Path, name: str) -> Path:
        """Create zip archive of export."""
        zip_path = EXPORT_DIR / f"{name}.zip"
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in source_path.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(source_path)
                    zipf.write(file, arcname)
        
        shutil.rmtree(source_path)
        
        return zip_path


_exporter = None


def get_investor_exporter() -> InvestorDataExporter:
    """Get singleton exporter instance."""
    global _exporter
    if _exporter is None:
        _exporter = InvestorDataExporter()
    return _exporter
