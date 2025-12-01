"""
Investor Reporting Module.

Provides institutional-grade trade logging, reporting, and export capabilities
for due diligence, regulatory compliance, and investor updates.

Includes:
- Trade journal with complete execution details
- Due diligence logging (attestations, incidents, policies)
- Automated daily attestations
- Performance metrics tracking
- ML model training history
- Investor data export packages
"""

from .trade_journal import (
    InvestorTradeJournal,
    InvestorTradeEntry,
    AccountSnapshot,
    SignalQuality,
    MarketContext,
    RiskAssessment,
    ProtocolAnalysis,
    ExecutionDetails,
    DailySummary,
    MonthlyReport,
    get_trade_journal,
)

from .due_diligence_logger import (
    DueDiligenceLogger,
    ComplianceAttestation,
    IncidentLifecycle,
    PolicyManifestEntry,
    ReconciliationRecord,
    ConsentRecord,
    DocumentAccessLog,
    AttestationType,
    AttestationStatus,
    IncidentLifecycleStatus,
    ReconciliationStatus,
    ConsentType,
    DocumentAccessAction,
    get_due_diligence_logger,
)

from .automated_attestations import (
    AutomatedAttestationService,
    get_attestation_service,
    run_daily_attestations,
)

from .performance_logger import (
    PerformanceLogger,
    DailyPerformanceSnapshot,
    MonthlyPerformanceSummary,
    get_performance_logger,
)

from .model_training_logger import (
    ModelTrainingLogger,
    ModelTrainingRun,
    ModelValidationResult,
    ModelDeploymentEvent,
    DriftDetectionEvent,
    get_model_training_logger,
)

from .investor_data_exporter import (
    InvestorDataExporter,
    get_investor_exporter,
)

__all__ = [
    "InvestorTradeJournal",
    "InvestorTradeEntry",
    "AccountSnapshot",
    "SignalQuality",
    "MarketContext",
    "RiskAssessment",
    "ProtocolAnalysis",
    "ExecutionDetails",
    "DailySummary",
    "MonthlyReport",
    "get_trade_journal",
    "DueDiligenceLogger",
    "ComplianceAttestation",
    "IncidentLifecycle",
    "PolicyManifestEntry",
    "ReconciliationRecord",
    "ConsentRecord",
    "DocumentAccessLog",
    "AttestationType",
    "AttestationStatus",
    "IncidentLifecycleStatus",
    "ReconciliationStatus",
    "ConsentType",
    "DocumentAccessAction",
    "get_due_diligence_logger",
    "AutomatedAttestationService",
    "get_attestation_service",
    "run_daily_attestations",
    "PerformanceLogger",
    "DailyPerformanceSnapshot",
    "MonthlyPerformanceSummary",
    "get_performance_logger",
    "ModelTrainingLogger",
    "ModelTrainingRun",
    "ModelValidationResult",
    "ModelDeploymentEvent",
    "DriftDetectionEvent",
    "get_model_training_logger",
    "InvestorDataExporter",
    "get_investor_exporter",
]
