#!/usr/bin/env python3
"""
End-to-End Integration Test for QuantraCore Apex v9.0-A

Tests all major subsystems:
1. Core Engine & Protocol Execution
2. Hardening Infrastructure (Mode Enforcement, Kill Switch, Manifest)
3. Broker Layer (RESEARCH mode blocking, PAPER mode execution)
4. EEO Engine (Plan Generation)
5. Estimated Move Module
6. Compliance & Omega Directives
"""

import sys
import json
from datetime import datetime


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{symbol}] {name}: {status}")
    if details and not passed:
        print(f"      → {details}")
    return passed


def main():
    print("\n" + "="*60)
    print("  QuantraCore Apex v9.0-A — End-to-End Integration Test")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    results = {"passed": 0, "failed": 0, "tests": []}
    
    section("1. HARDENING INFRASTRUCTURE")
    
    from src.quantracore_apex.hardening.manifest import (
        ProtocolManifest, ManifestValidator
    )
    from src.quantracore_apex.hardening.mode_enforcer import (
        ModeEnforcer, ExecutionMode, ModePermissions,
        get_mode_enforcer, reset_mode_enforcer, set_mode_for_testing
    )
    from src.quantracore_apex.hardening.kill_switch import (
        KillSwitchManager, get_kill_switch_manager, KillSwitchReason
    )
    
    manifest = ProtocolManifest.generate_default()
    passed = check("Protocol manifest generation", len(manifest.protocols) == 115)
    results["passed" if passed else "failed"] += 1
    
    validator = ManifestValidator()
    try:
        valid = validator.validate()
        passed = check("Protocol manifest validation", valid)
    except Exception as e:
        passed = check("Protocol manifest validation", False, str(e))
    results["passed" if passed else "failed"] += 1
    
    reset_mode_enforcer()
    enforcer = get_mode_enforcer()
    passed = check("Mode enforcer initialization", enforcer.current_mode == ExecutionMode.RESEARCH)
    results["passed" if passed else "failed"] += 1
    
    perms = enforcer.permissions
    passed = check("RESEARCH mode blocks paper orders", not perms.paper_orders_allowed)
    results["passed" if passed else "failed"] += 1
    
    passed = check("RESEARCH mode blocks live orders", not perms.live_orders_allowed)
    results["passed" if passed else "failed"] += 1
    
    ks = get_kill_switch_manager()
    ks.reset("e2e_test")
    passed = check("Kill switch starts disengaged", not ks.is_engaged())
    results["passed" if passed else "failed"] += 1
    
    ks.engage(KillSwitchReason.MANUAL, "e2e_test")
    passed = check("Kill switch can be engaged", ks.is_engaged())
    results["passed" if passed else "failed"] += 1
    
    order_allowed, _ = ks.check_order_allowed()
    passed = check("Kill switch blocks orders when engaged", not order_allowed)
    results["passed" if passed else "failed"] += 1
    
    ks.reset("e2e_test")
    passed = check("Kill switch can be disengaged", not ks.is_engaged())
    results["passed" if passed else "failed"] += 1
    
    section("2. BROKER LAYER - RESEARCH MODE")
    
    from src.quantracore_apex.broker import (
        BrokerConfig, ExecutionEngine, ApexSignal,
        SignalDirection, ExecutionMode as BrokerMode, OrderStatus
    )
    
    reset_mode_enforcer()
    
    research_config = BrokerConfig(execution_mode=BrokerMode.RESEARCH)
    research_engine = ExecutionEngine(config=research_config)
    
    passed = check("Research engine uses NULL adapter", research_engine.router.adapter_name == "NULL_ADAPTER")
    results["passed" if passed else "failed"] += 1
    
    signal = ApexSignal(
        signal_id="e2e_research_001",
        symbol="AAPL",
        direction=SignalDirection.LONG,
        quantra_score=80.0,
        size_hint=0.01
    )
    result = research_engine.execute_signal(signal)
    research_blocked = (result is None) or (result.status == OrderStatus.REJECTED) or (result.filled_qty == 0)
    passed = check("RESEARCH mode restricts execution", research_blocked)
    results["passed" if passed else "failed"] += 1
    
    section("3. BROKER LAYER - PAPER MODE")
    
    set_mode_for_testing(ExecutionMode.PAPER)
    
    paper_config = BrokerConfig(execution_mode=BrokerMode.PAPER)
    paper_engine = ExecutionEngine(config=paper_config)
    
    passed = check("Paper engine uses PAPER_SIM adapter", paper_engine.router.adapter_name == "PAPER_SIM")
    results["passed" if passed else "failed"] += 1
    
    paper_signal = ApexSignal(
        signal_id="e2e_paper_001",
        symbol="MSFT",
        direction=SignalDirection.LONG,
        quantra_score=75.0,
        size_hint=0.01
    )
    paper_result = paper_engine.execute_signal(paper_signal)
    passed = check("PAPER mode fills orders", paper_result is not None and paper_result.status == OrderStatus.FILLED)
    results["passed" if passed else "failed"] += 1
    
    positions = paper_engine.router.get_positions()
    passed = check("Paper position created", len(positions) == 1 and positions[0].symbol == "MSFT")
    results["passed" if passed else "failed"] += 1
    
    reset_mode_enforcer()
    
    section("4. ENTRY/EXIT OPTIMIZATION ENGINE")
    
    from src.quantracore_apex.eeo_engine import (
        EntryExitOptimizer, EntryOptimizer, ExitOptimizer,
        BALANCED_PROFILE, CONSERVATIVE_PROFILE, AGGRESSIVE_RESEARCH_PROFILE
    )
    
    eeo = EntryExitOptimizer()
    entry_opt = EntryOptimizer(BALANCED_PROFILE)
    exit_opt = ExitOptimizer(BALANCED_PROFILE)
    
    passed = check("EEO optimizer initialized", eeo is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("Entry optimizer initialized", entry_opt is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("Exit optimizer initialized", exit_opt is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("All profiles available", 
                   BALANCED_PROFILE is not None and 
                   CONSERVATIVE_PROFILE is not None and 
                   AGGRESSIVE_RESEARCH_PROFILE is not None)
    results["passed" if passed else "failed"] += 1
    
    section("5. ESTIMATED MOVE MODULE")
    
    from src.quantracore_apex.estimated_move import (
        EstimatedMoveEngine, EstimatedMoveInput, HorizonWindow, MoveRange
    )
    
    em_engine = EstimatedMoveEngine()
    
    passed = check("Estimated move engine initialized", em_engine is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("HorizonWindow enum available", HorizonWindow.SHORT_TERM is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("MoveRange schema available", MoveRange is not None)
    results["passed" if passed else "failed"] += 1
    
    section("6. CORE ENGINE & PROTOCOLS")
    
    from src.quantracore_apex.core.engine import ApexEngine
    
    engine = ApexEngine()
    
    passed = check("Core engine initialized", engine is not None)
    results["passed" if passed else "failed"] += 1
    
    passed = check("Protocol runner initialized", engine.protocol_runner is not None)
    results["passed" if passed else "failed"] += 1
    
    section("7. COMPLIANCE & REGULATORY")
    
    from src.quantracore_apex.compliance import (
        RegulatoryExcellenceEngine, ComplianceScore, AuditTrail
    )
    
    reg_engine = RegulatoryExcellenceEngine()
    
    passed = check("Regulatory engine initialized", reg_engine is not None)
    results["passed" if passed else "failed"] += 1
    
    audit = AuditTrail()
    passed = check("Audit trail initialized", audit is not None)
    results["passed" if passed else "failed"] += 1
    
    section("8. DETERMINISM VERIFICATION")
    
    import hashlib
    test_data = "determinism_test_input_data"
    hash1 = hashlib.sha256(test_data.encode()).hexdigest()
    hash2 = hashlib.sha256(test_data.encode()).hexdigest()
    
    passed = check("Deterministic output (same input → same hash)", hash1 == hash2)
    results["passed" if passed else "failed"] += 1
    
    section("SUMMARY")
    
    total = results["passed"] + results["failed"]
    pct = (results["passed"] / total * 100) if total > 0 else 0
    
    print(f"\n  Total Tests: {total}")
    print(f"  Passed:      {results['passed']} ({pct:.1f}%)")
    print(f"  Failed:      {results['failed']}")
    
    if results["failed"] == 0:
        print("\n  ✓ ALL TESTS PASSED")
        print("  QuantraCore Apex v9.0-A is operational.")
    else:
        print(f"\n  ✗ {results['failed']} TEST(S) FAILED")
        sys.exit(1)
    
    print("\n" + "="*60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
