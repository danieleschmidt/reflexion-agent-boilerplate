#!/usr/bin/env python3
"""
Security scanning script for Reflexion Agent.
Performs various security checks and generates reports.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
import tempfile


def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, 1, "", f"Command not found: {cmd[0]}")


def scan_dependencies() -> Dict[str, Any]:
    """Scan dependencies for known vulnerabilities."""
    print("Scanning dependencies for vulnerabilities...")
    
    result = run_command(["safety", "check", "--json"])
    
    if result.returncode == 0:
        return {
            "status": "clean",
            "vulnerabilities": [],
            "message": "No known vulnerabilities found"
        }
    else:
        try:
            vulnerabilities = json.loads(result.stdout) if result.stdout else []
            return {
                "status": "vulnerabilities_found",
                "vulnerabilities": vulnerabilities,
                "count": len(vulnerabilities)
            }
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": result.stderr or "Failed to parse safety output"
            }


def scan_code_security() -> Dict[str, Any]:
    """Scan code for security issues using bandit."""
    print("Scanning code for security issues...")
    
    result = run_command([
        "bandit", 
        "-r", "reflexion/", 
        "-f", "json",
        "-ll"  # Low level and above
    ])
    
    try:
        if result.stdout:
            bandit_output = json.loads(result.stdout)
            issues = bandit_output.get("results", [])
            
            # Categorize issues by severity
            severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for issue in issues:
                severity = issue.get("issue_severity", "LOW")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "status": "completed",
                "total_issues": len(issues),
                "severity_breakdown": severity_counts,
                "issues": issues[:10]  # Top 10 issues for summary
            }
        else:
            return {
                "status": "clean",
                "total_issues": 0,
                "message": "No security issues found"
            }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Failed to parse bandit output"
        }


def scan_secrets() -> Dict[str, Any]:
    """Scan for accidentally committed secrets."""
    print("Scanning for secrets...")
    
    # Simple regex-based secret detection
    secret_patterns = [
        r'(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
        r'(?i)bearer\s+[a-zA-Z0-9_-]{20,}',
        r'(?i)(aws|sk)_[a-zA-Z0-9_]{20,}',
        r'(?i)ghp_[a-zA-Z0-9_]{36}',  # GitHub personal access token
        r'(?i)gho_[a-zA-Z0-9_]{36}',  # GitHub OAuth token
    ]
    
    issues = []
    
    # Scan Python files
    for py_file in Path("reflexion").rglob("*.py"):
        try:
            content = py_file.read_text()
            for line_num, line in enumerate(content.splitlines(), 1):
                for pattern in secret_patterns:
                    import re
                    if re.search(pattern, line):
                        issues.append({
                            "file": str(py_file),
                            "line": line_num,
                            "pattern": "potential_secret",
                            "content": line.strip()[:100]  # Truncate for safety
                        })
        except Exception as e:
            print(f"Error scanning {py_file}: {e}")
    
    return {
        "status": "completed",
        "issues_found": len(issues),
        "issues": issues
    }


def check_file_permissions() -> Dict[str, Any]:
    """Check for files with overly permissive permissions."""
    print("Checking file permissions...")
    
    issues = []
    
    # Check for world-writable files
    for file_path in Path(".").rglob("*"):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                # Check if world-writable (mode & 0o002)
                if stat.st_mode & 0o002:
                    issues.append({
                        "file": str(file_path),
                        "issue": "world_writable",
                        "permissions": oct(stat.st_mode)[-3:]
                    })
                # Check for executable config files
                elif file_path.suffix in ['.json', '.yml', '.yaml', '.toml'] and stat.st_mode & 0o111:
                    issues.append({
                        "file": str(file_path),
                        "issue": "executable_config",
                        "permissions": oct(stat.st_mode)[-3:]
                    })
            except Exception:
                pass  # Skip files we can't stat
    
    return {
        "status": "completed",
        "issues_found": len(issues),
        "issues": issues
    }


def check_docker_security() -> Dict[str, Any]:
    """Check Docker configuration for security issues."""
    print("Checking Docker security...")
    
    issues = []
    
    # Check if Dockerfile exists
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        return {
            "status": "skipped",
            "message": "No Dockerfile found"
        }
    
    try:
        dockerfile_content = dockerfile_path.read_text()
        lines = dockerfile_content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for running as root
            if line.startswith("USER root"):
                issues.append({
                    "line": line_num,
                    "issue": "running_as_root",
                    "content": line,
                    "severity": "HIGH"
                })
            
            # Check for using latest tag
            if "FROM" in line and ":latest" in line:
                issues.append({
                    "line": line_num,
                    "issue": "using_latest_tag",
                    "content": line,
                    "severity": "MEDIUM"
                })
            
            # Check for ADD instead of COPY
            if line.startswith("ADD ") and not line.startswith("ADD --"):
                issues.append({
                    "line": line_num,
                    "issue": "using_add_instead_of_copy",
                    "content": line,
                    "severity": "LOW"
                })
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading Dockerfile: {e}"
        }
    
    return {
        "status": "completed",
        "issues_found": len(issues),
        "issues": issues
    }


def generate_report(results: Dict[str, Any], output_file: str = None) -> str:
    """Generate a security scan report."""
    report_lines = [
        "# Security Scan Report",
        f"Generated: {subprocess.check_output(['date'], text=True).strip()}",
        "",
        "## Summary",
        ""
    ]
    
    total_issues = 0
    for scan_type, result in results.items():
        if isinstance(result, dict):
            issues = result.get("issues_found", result.get("total_issues", 0))
            total_issues += issues
            status = result.get("status", "unknown")
            report_lines.append(f"- **{scan_type.replace('_', ' ').title()}**: {status} ({issues} issues)")
    
    report_lines.extend([
        "",
        f"**Total Issues Found**: {total_issues}",
        ""
    ])
    
    # Detailed results
    for scan_type, result in results.items():
        if isinstance(result, dict) and result.get("issues"):
            report_lines.extend([
                f"## {scan_type.replace('_', ' ').title()}",
                ""
            ])
            
            for issue in result["issues"][:5]:  # Top 5 issues per category
                if isinstance(issue, dict):
                    if "file" in issue:
                        report_lines.append(f"- **{issue.get('file')}**: {issue.get('issue', 'Unknown issue')}")
                    else:
                        report_lines.append(f"- {issue.get('issue_text', str(issue))}")
            
            if len(result["issues"]) > 5:
                report_lines.append(f"- ... and {len(result['issues']) - 5} more issues")
            
            report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    if output_file:
        Path(output_file).write_text(report_content)
        print(f"Report saved to {output_file}")
    
    return report_content


def main():
    parser = argparse.ArgumentParser(description="Reflexion Agent Security Scanner")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    parser.add_argument("--scan", action="append", choices=[
        "dependencies", "code", "secrets", "permissions", "docker"
    ], help="Specific scans to run (default: all)")
    
    args = parser.parse_args()
    
    # If no specific scans specified, run all
    scans_to_run = args.scan or ["dependencies", "code", "secrets", "permissions", "docker"]
    
    results = {}
    
    for scan_type in scans_to_run:
        try:
            if scan_type == "dependencies":
                results["dependency_vulnerabilities"] = scan_dependencies()
            elif scan_type == "code":
                results["code_security"] = scan_code_security()
            elif scan_type == "secrets":
                results["secret_detection"] = scan_secrets()
            elif scan_type == "permissions":
                results["file_permissions"] = check_file_permissions()
            elif scan_type == "docker":
                results["docker_security"] = check_docker_security()
        except Exception as e:
            results[f"{scan_type}_error"] = {
                "status": "error",
                "message": str(e)
            }
    
    # Generate output
    if args.format == "json":
        output_content = json.dumps(results, indent=2)
        if args.output:
            Path(args.output).write_text(output_content)
        else:
            print(output_content)
    else:
        report = generate_report(results, args.output)
        if not args.output:
            print(report)
    
    # Exit with error code if security issues found
    total_issues = sum(
        result.get("issues_found", result.get("total_issues", 0))
        for result in results.values()
        if isinstance(result, dict)
    )
    
    if total_issues > 0:
        print(f"\n⚠️  Security scan completed with {total_issues} issues found")
        sys.exit(1)
    else:
        print("\n✅ Security scan completed - no issues found")
        sys.exit(0)


if __name__ == "__main__":
    main()