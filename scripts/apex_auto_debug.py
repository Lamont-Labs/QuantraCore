#!/usr/bin/env python3
"""
QuantraCore Apex v9.0-A — Auto Debug Script
Runs linting (ruff) and typechecking (mypy) on the codebase.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          QUANTRACORE APEX v9.0-A — AUTO DEBUG                ║
║                    Lamont Labs                               ║
╚══════════════════════════════════════════════════════════════╝
"""

SOURCE_DIR = Path("src/quantracore_apex")
TESTS_DIR = Path("tests")


def run_command(cmd: list[str], description: str) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    if output.strip():
        print(output)
    
    if result.returncode == 0:
        print(f"  ✓ {description} passed")
    else:
        print(f"  ✗ {description} failed (exit code: {result.returncode})")
    
    return result.returncode, output


def run_lint() -> int:
    """Run ruff linter."""
    cmd = ["ruff", "check", str(SOURCE_DIR), str(TESTS_DIR), "--output-format=concise"]
    exit_code, _ = run_command(cmd, "RUFF LINTING")
    return exit_code


def run_typecheck() -> int:
    """Run mypy type checker."""
    cmd = [
        "mypy", 
        str(SOURCE_DIR),
        "--ignore-missing-imports",
        "--no-error-summary",
        "--show-error-codes",
        "--pretty"
    ]
    exit_code, _ = run_command(cmd, "MYPY TYPE CHECKING")
    return exit_code


def run_security() -> int:
    """Run bandit security scanner."""
    cmd = ["bandit", "-r", str(SOURCE_DIR), "-ll", "-q"]
    exit_code, _ = run_command(cmd, "BANDIT SECURITY SCAN")
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="QuantraCore Apex Auto Debug - Lint, Typecheck, and Security Scan"
    )
    parser.add_argument("--lint", action="store_true", help="Run ruff linter")
    parser.add_argument("--typecheck", action="store_true", help="Run mypy type checker")
    parser.add_argument("--security", action="store_true", help="Run bandit security scan")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    
    args = parser.parse_args()
    
    if not any([args.lint, args.typecheck, args.security, args.all]):
        parser.print_help()
        return 0
    
    print(BANNER)
    
    results = {}
    
    if args.lint or args.all:
        results["lint"] = run_lint()
    
    if args.typecheck or args.all:
        results["typecheck"] = run_typecheck()
    
    if args.security or args.all:
        results["security"] = run_security()
    
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    
    total_issues = 0
    for check, code in results.items():
        status = "✓ PASS" if code == 0 else f"✗ FAIL ({code})"
        print(f"  {check.upper():15} {status}")
        if code != 0:
            total_issues += 1
    
    print(f"{'='*60}")
    
    if total_issues == 0:
        print("  All checks passed!")
        return 0
    else:
        print(f"  {total_issues} check(s) reported issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
