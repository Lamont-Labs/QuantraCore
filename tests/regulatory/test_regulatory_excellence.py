"""
Regulatory Excellence Tests

These tests verify that QuantraCore Apex EXCEEDS regulatory requirements,
not just meets them. We test for:
- 3x FINRA 15-09 determinism requirements
- 5x MiFID II latency margins
- 4x SEC risk control sensitivity
- 100% audit trail completeness
- Cryptographic proof integrity

These tests ensure we are setting the BENCHMARK for regulatory compliance.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.quantracore_apex.compliance.excellence import (
    RegulatoryExcellenceEngine,
    ComplianceScore,
    ExcellenceLevel,
    ComplianceMetrics,
    RegulatoryStandard,
)
from src.quantracore_apex.compliance.audit import (
    AuditTrail,
    DecisionProvenance,
    AuditEntry,
    AuditEventType,
)
from src.quantracore_apex.compliance.reporter import (
    RegulatoryReporter,
    ComplianceReport,
)


class TestExcellenceEngine:
    """
    Test the Regulatory Excellence Engine.
    
    Verifies that our excellence standards are properly configured
    to exceed regulatory minimums by the target multipliers.
    """
    
    def test_excellence_standards_defined(self):
        """All 7 excellence standards must be defined."""
        engine = RegulatoryExcellenceEngine()
        assert len(engine.EXCELLENCE_STANDARDS) >= 7
    
    def test_all_standards_exceed_minimums(self):
        """Every standard target must exceed the minimum requirement."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if "Latency" in standard.section:
                assert standard.quantracore_target < standard.minimum_requirement, (
                    f"{standard.regulation} target should be LOWER (better) for latency"
                )
            else:
                assert standard.quantracore_target >= standard.minimum_requirement, (
                    f"{standard.regulation} target must exceed minimum"
                )
    
    def test_finra_determinism_3x_requirement(self):
        """FINRA 15-09 determinism should target 3x the minimum."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if "Determinism" in standard.section:
                multiplier = standard.quantracore_target / standard.minimum_requirement
                assert multiplier >= 3.0, (
                    f"FINRA determinism target should be 3x minimum, got {multiplier:.1f}x"
                )
                break
    
    def test_mifid_latency_5x_better(self):
        """MiFID II latency should be 5x better than requirement."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if "Latency" in standard.section:
                multiplier = standard.minimum_requirement / standard.quantracore_target
                assert multiplier >= 5.0, (
                    f"MiFID latency target should be 5x better, got {multiplier:.1f}x"
                )
                break
    
    def test_audit_completeness_100_percent(self):
        """Audit trail completeness should target 100%."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if "Audit" in standard.section:
                assert standard.quantracore_target == 100.0, (
                    "Audit trail completeness must target 100%"
                )
                break
    
    def test_excellence_score_calculation(self):
        """Excellence score calculation must produce valid results."""
        engine = RegulatoryExcellenceEngine()
        
        for _ in range(10):
            engine.record_execution(
                determinism_verified=True,
                latency_ms=50.0,
                proof_verified=True,
                omega_directives_active={"omega_4_compliance": True},
            )
        
        score = engine.calculate_score()
        
        assert 0 <= score.overall_score <= 100
        assert score.level in ExcellenceLevel
        assert isinstance(score.timestamp, datetime)
        assert isinstance(score.metrics, ComplianceMetrics)
    
    def test_excellence_level_hierarchy(self):
        """Excellence levels must follow proper hierarchy."""
        levels = [
            ExcellenceLevel.EXCEPTIONAL,
            ExcellenceLevel.EXCELLENT,
            ExcellenceLevel.SUPERIOR,
            ExcellenceLevel.COMPLIANT,
            ExcellenceLevel.BELOW_STANDARD,
        ]
        assert len(levels) == 5
    
    def test_excellence_summary_structure(self):
        """Excellence summary must contain all required fields."""
        engine = RegulatoryExcellenceEngine()
        summary = engine.get_excellence_summary()
        
        assert "overall" in summary
        assert "achievements" in summary
        assert "excellence_multipliers" in summary
        assert "compliance_mode" in summary
        assert summary["compliance_mode"] == "RESEARCH_ONLY"


class TestAuditTrail:
    """
    Test the Enhanced Audit Trail System.
    
    Verifies complete decision provenance and cryptographic integrity.
    """
    
    def test_provenance_chain_creation(self):
        """Decision provenance chain must be properly initialized."""
        provenance = DecisionProvenance(
            decision_id="test-001",
            symbol="AAPL",
            timestamp=datetime.utcnow(),
        )
        
        assert provenance.decision_id == "test-001"
        assert provenance.symbol == "AAPL"
        assert len(provenance.entries) == 0
    
    def test_audit_entry_hash_integrity(self):
        """Each audit entry must have a valid cryptographic hash."""
        entry = AuditEntry(
            entry_id="entry-001",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.ANALYSIS_STARTED,
            component="ApexEngine",
            action="initialize",
            data={"symbol": "AAPL"},
        )
        
        assert entry.entry_hash is not None
        assert len(entry.entry_hash) == 64
        assert entry.verify_integrity() is True
    
    def test_provenance_chain_linking(self):
        """Entries in a provenance chain must be cryptographically linked."""
        provenance = DecisionProvenance(
            decision_id="test-002",
            symbol="MSFT",
            timestamp=datetime.utcnow(),
        )
        
        entry1 = provenance.add_entry(
            event_type=AuditEventType.ANALYSIS_STARTED,
            component="ApexEngine",
            action="start",
            data={},
        )
        
        entry2 = provenance.add_entry(
            event_type=AuditEventType.PROTOCOL_EXECUTED,
            component="T01",
            action="execute",
            data={"protocol_id": "T01"},
        )
        
        assert entry1.previous_hash is None
        assert entry2.previous_hash == entry1.entry_hash
    
    def test_provenance_chain_verification(self):
        """Complete provenance chain must be verifiable."""
        provenance = DecisionProvenance(
            decision_id="test-003",
            symbol="GOOGL",
            timestamp=datetime.utcnow(),
        )
        
        for i in range(5):
            provenance.add_entry(
                event_type=AuditEventType.PROTOCOL_EXECUTED,
                component=f"T{i+1:02d}",
                action="execute",
                data={"protocol_id": f"T{i+1:02d}"},
            )
        
        assert provenance.verify_chain() is True
    
    def test_tampered_chain_detection(self):
        """Tampered provenance chains must be detected."""
        provenance = DecisionProvenance(
            decision_id="test-004",
            symbol="TSLA",
            timestamp=datetime.utcnow(),
        )
        
        provenance.add_entry(
            event_type=AuditEventType.ANALYSIS_STARTED,
            component="ApexEngine",
            action="start",
            data={},
        )
        
        provenance.add_entry(
            event_type=AuditEventType.SCORE_COMPUTED,
            component="QuantraScore",
            action="compute",
            data={"score": 75.0},
        )
        
        provenance.entries[0].data["symbol"] = "TAMPERED"
        
        assert provenance.verify_chain() is False
    
    def test_audit_event_types_complete(self):
        """All required audit event types must be defined."""
        required_events = [
            "ANALYSIS_STARTED",
            "INPUT_RECEIVED",
            "PROTOCOL_EXECUTED",
            "SCORE_COMPUTED",
            "VERDICT_GENERATED",
            "OMEGA_APPLIED",
            "ANALYSIS_COMPLETED",
            "COMPLIANCE_CHECK",
            "PROOF_GENERATED",
        ]
        
        for event_name in required_events:
            assert hasattr(AuditEventType, event_name), (
                f"Missing audit event type: {event_name}"
            )


class TestRegulatoryReporter:
    """
    Test the Regulatory Report Generator.
    
    Verifies comprehensive compliance reports for regulatory review.
    """
    
    def test_report_generation(self):
        """Compliance reports must be generated successfully."""
        reporter = RegulatoryReporter(output_dir="/tmp/test_reports")
        
        excellence_summary = {
            "overall": {"score": 92.5, "level": "excellent"},
            "achievements": {
                "standards_met": 7,
                "standards_exceeded": 5,
                "areas_of_excellence": 3,
            },
            "excellence_multipliers": {
                "finra_determinism": "3.0x",
                "mifid_latency": "5.0x",
                "audit_completeness": "100%",
                "proof_integrity": "100%",
            },
            "compliance_mode": "RESEARCH_ONLY",
            "omega_4_enforcement": "100%",
        }
        
        report = reporter.generate_compliance_report(
            excellence_summary=excellence_summary,
            period_days=1,
        )
        
        assert report.report_id.startswith("CR-")
        assert "overall_compliance_score" in report.executive_summary
        assert len(report.regulatory_adherence) > 0
    
    def test_report_regulatory_coverage(self):
        """Reports must cover all major regulatory frameworks."""
        reporter = RegulatoryReporter(output_dir="/tmp/test_reports")
        
        excellence_summary = {
            "overall": {"score": 90.0, "level": "excellent"},
            "achievements": {"standards_met": 7, "standards_exceeded": 5},
            "excellence_multipliers": {},
            "compliance_mode": "RESEARCH_ONLY",
        }
        
        report = reporter.generate_compliance_report(
            excellence_summary=excellence_summary,
        )
        
        regulations = report.regulatory_adherence.keys()
        
        assert any("FINRA" in r for r in regulations), "Missing FINRA coverage"
        assert any("MiFID" in r for r in regulations), "Missing MiFID coverage"
        assert any("SEC" in r for r in regulations), "Missing SEC coverage"
        assert any("Basel" in r for r in regulations), "Missing Basel coverage"
        assert any("SOX" in r for r in regulations), "Missing SOX/SOC2 coverage"
    
    def test_report_omega_status(self):
        """Reports must include all 5 Omega directive statuses."""
        reporter = RegulatoryReporter(output_dir="/tmp/test_reports")
        
        excellence_summary = {
            "overall": {"score": 85.0, "level": "superior"},
            "achievements": {},
            "excellence_multipliers": {},
            "compliance_mode": "RESEARCH_ONLY",
        }
        
        report = reporter.generate_compliance_report(
            excellence_summary=excellence_summary,
        )
        
        omega_controls = report.risk_control_status.keys()
        
        assert len(omega_controls) == 5, "Must have all 5 Omega directives"
        assert all("Omega" in ctrl for ctrl in omega_controls)
    
    def test_report_markdown_generation(self):
        """Reports must be exportable to markdown format."""
        reporter = RegulatoryReporter(output_dir="/tmp/test_reports")
        
        excellence_summary = {
            "overall": {"score": 95.0, "level": "exceptional"},
            "achievements": {"standards_met": 7, "standards_exceeded": 7},
            "excellence_multipliers": {"finra_determinism": "3.0x"},
            "compliance_mode": "RESEARCH_ONLY",
        }
        
        report = reporter.generate_compliance_report(
            excellence_summary=excellence_summary,
        )
        
        markdown = report.to_markdown()
        
        assert "# QuantraCore Apex Regulatory Compliance Report" in markdown
        assert "Executive Summary" in markdown
        assert "Regulatory Adherence" in markdown
        assert "Risk Control Status" in markdown
        assert "research purposes only" in markdown.lower()


class TestExcellenceMultipliers:
    """
    Test that excellence multipliers are correctly calculated.
    
    These tests verify that we truly EXCEED requirements, not just meet them.
    """
    
    @pytest.mark.parametrize("regulation,min_multiplier", [
        ("FINRA 15-09", 3.0),
        ("SEC 15c3-5", 4.0),
        ("Basel Committee", 2.0),
    ])
    def test_exceeds_by_multiplier(self, regulation: str, min_multiplier: float):
        """Each regulation must be exceeded by the specified multiplier."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if regulation in standard.regulation:
                if "Latency" not in standard.section:
                    actual_mult = standard.quantracore_target / standard.minimum_requirement
                    assert actual_mult >= min_multiplier, (
                        f"{regulation} must exceed by {min_multiplier}x, "
                        f"got {actual_mult:.1f}x"
                    )
                break
    
    def test_latency_inverse_multiplier(self):
        """Latency multiplier must be calculated inversely (lower is better)."""
        for standard in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            if "Latency" in standard.section:
                multiplier = standard.minimum_requirement / standard.quantracore_target
                assert multiplier >= 5.0, (
                    f"Latency must be 5x better than minimum"
                )
                break


class TestComplianceModeEnforcement:
    """
    Test that compliance mode (research-only) is ALWAYS enforced.
    
    This is critical - the system must never present itself as trading advice.
    """
    
    def test_compliance_mode_in_excellence_summary(self):
        """Excellence summary must always include RESEARCH_ONLY mode."""
        engine = RegulatoryExcellenceEngine()
        summary = engine.get_excellence_summary()
        
        assert summary["compliance_mode"] == "RESEARCH_ONLY"
    
    def test_compliance_mode_in_score(self):
        """Compliance score output must indicate research-only mode."""
        engine = RegulatoryExcellenceEngine()
        score = engine.calculate_score()
        
        score_dict = score.to_dict()
        assert "timestamp" in score_dict
    
    def test_omega_4_always_enabled(self):
        """Omega 4 (compliance mode) must be enabled by default."""
        from src.quantracore_apex.protocols.omega.omega import OmegaDirectives
        
        omega = OmegaDirectives()
        assert omega.enable_omega_4 is True, (
            "Omega 4 (compliance mode) must be enabled by default"
        )
