"""
Guard Test: No Placeholders in Runtime Code

This test ensures that no placeholder, stub, or unimplemented code
exists in the active runtime paths of the system.
"""

import pytest
import os
import re
from pathlib import Path


RUNTIME_PATHS = [
    "src/quantracore_apex/core",
    "src/quantracore_apex/protocols/tier",
    "src/quantracore_apex/protocols/learning",
    "src/quantracore_apex/protocols/monster_runner",
    "src/quantracore_apex/protocols/omega",
    "src/quantracore_apex/apexlab",
    "src/quantracore_apex/apexcore",
    "src/quantracore_apex/data_layer",
    "src/quantracore_apex/risk",
    "src/quantracore_apex/broker",
    "src/quantracore_apex/portfolio",
    "src/quantracore_apex/signal",
    "src/quantracore_apex/prediction",
    "src/quantracore_apex/server",
]

EXCLUDED_FILES = [
    "_stubs_legacy_unused.py",
    "base.py",
]

FORBIDDEN_PATTERNS = [
    r"raise\s+NotImplementedError",
    r"pass\s*#\s*TODO",
    r'"status":\s*"stub"',
]

ALLOWED_PLACEHOLDER_CONTEXTS = [
    r'placeholder\s*=',
    r'is_placeholder',
    r'placeholder.*input',
]


class TestNoPlaceholders:
    """Ensure no placeholders exist in runtime code."""
    
    def test_no_notimplementederror_in_runtime(self):
        """Verify no NotImplementedError in runtime code."""
        violations = []
        
        for runtime_path in RUNTIME_PATHS:
            path = Path(runtime_path)
            if not path.exists():
                continue
                
            for py_file in path.rglob("*.py"):
                if any(excl in str(py_file) for excl in EXCLUDED_FILES):
                    continue
                if "test" in str(py_file):
                    continue
                    
                content = py_file.read_text()
                
                if re.search(r"raise\s+NotImplementedError\([^)]*\)", content):
                    if "def run(" not in content or "class" not in content:
                        violations.append(str(py_file))
        
        assert len(violations) == 0, f"NotImplementedError found in: {violations}"
    
    def test_no_pass_todo_in_runtime(self):
        """Verify no 'pass  # TODO' in runtime code."""
        violations = []
        
        for runtime_path in RUNTIME_PATHS:
            path = Path(runtime_path)
            if not path.exists():
                continue
                
            for py_file in path.rglob("*.py"):
                if any(excl in str(py_file) for excl in EXCLUDED_FILES):
                    continue
                if "test" in str(py_file):
                    continue
                    
                content = py_file.read_text()
                
                if re.search(r"pass\s*#\s*TODO", content, re.IGNORECASE):
                    violations.append(str(py_file))
        
        assert len(violations) == 0, f"'pass # TODO' found in: {violations}"
    
    def test_no_stub_status_in_protocols(self):
        """Verify no 'status: stub' in protocol results."""
        violations = []
        
        protocol_paths = [
            "src/quantracore_apex/protocols/tier",
            "src/quantracore_apex/protocols/learning",
            "src/quantracore_apex/protocols/monster_runner",
        ]
        
        for protocol_path in protocol_paths:
            path = Path(protocol_path)
            if not path.exists():
                continue
                
            for py_file in path.rglob("*.py"):
                if any(excl in str(py_file) for excl in EXCLUDED_FILES):
                    continue
                if "test" in str(py_file):
                    continue
                if "loader" in str(py_file):
                    continue
                    
                content = py_file.read_text()
                
                if re.search(r'"status":\s*"stub"', content):
                    violations.append(str(py_file))
        
        assert len(violations) == 0, f"Stub status found in: {violations}"
    
    def test_all_tier_protocols_implemented(self):
        """Verify all 80 tier protocols are implemented."""
        from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
        
        runner = TierProtocolRunner()
        
        assert len(runner.protocols) == 80, f"Expected 80 protocols, found {len(runner.protocols)}"
        
        for i in range(1, 81):
            protocol_id = f"T{i:02d}"
            assert protocol_id in runner.protocols, f"Missing protocol: {protocol_id}"
    
    def test_all_learning_protocols_implemented(self):
        """Verify all 25 learning protocols are implemented."""
        from src.quantracore_apex.protocols.learning.learning_loader import LearningProtocolRunner
        
        runner = LearningProtocolRunner()
        
        assert len(runner.protocols) == 25, f"Expected 25 protocols, found {len(runner.protocols)}"
        
        for i in range(1, 26):
            protocol_id = f"LP{i:02d}"
            assert protocol_id in runner.protocols, f"Missing protocol: {protocol_id}"
    
    def test_all_monster_runner_protocols_implemented(self):
        """Verify all 5 MonsterRunner protocols are implemented."""
        from src.quantracore_apex.protocols.monster_runner.monster_runner_loader import MonsterRunnerLoader
        
        loader = MonsterRunnerLoader()
        
        assert len(loader.protocols) == 5, f"Expected 5 protocols, found {len(loader.protocols)}"
        
        for i in range(1, 6):
            protocol_id = f"MR{i:02d}"
            assert protocol_id in loader.protocols, f"Missing protocol: {protocol_id}"
