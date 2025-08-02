#!/usr/bin/env python3
"""
Automated metrics collection script for the reflexion agent boilerplate.
Collects various metrics about code quality, performance, and repository health.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class MetricsCollector:
    """Collects and aggregates project metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def collect_coverage_metrics(self) -> Dict[str, Any]:
        """Collect test coverage metrics."""
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    return {
                        "coverage_current": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                        "lines_total": coverage_data.get("totals", {}).get("num_statements", 0)
                    }
        except Exception as e:
            print(f"Error collecting coverage metrics: {e}")
        
        return {"coverage_current": 0, "lines_covered": 0, "lines_total": 0}
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics using bandit."""
        try:
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                return {
                    "vulnerability_count": len(bandit_data.get("results", [])),
                    "security_hotspots": len([r for r in bandit_data.get("results", []) 
                                            if r.get("issue_severity") == "HIGH"]),
                    "last_security_scan": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error collecting security metrics: {e}")
        
        return {"vulnerability_count": 0, "security_hotspots": 0, "last_security_scan": None}
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect git repository metrics."""
        try:
            # Get commit count for last week
            result = subprocess.run(
                ["git", "rev-list", "--count", "--since='1 week ago'", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            commits_per_week = int(result.stdout.strip()) if result.stdout.strip() else 0
            
            # Get contributor count
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            contributors = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            return {
                "commits_per_week": commits_per_week,
                "contributors": contributors
            }
        except Exception as e:
            print(f"Error collecting git metrics: {e}")
            return {"commits_per_week": 0, "contributors": 0}
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics using ruff."""
        try:
            # Run ruff check
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "src/", "--output-format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            issues = []
            if result.stdout:
                issues = json.loads(result.stdout)
            
            # Calculate maintainability score (simple heuristic)
            issue_count = len(issues)
            maintainability_index = max(0, 100 - (issue_count * 2))
            
            return {
                "maintainability_index": maintainability_index,
                "code_issues": issue_count,
                "last_quality_check": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error collecting code quality metrics: {e}")
            return {"maintainability_index": 100, "code_issues": 0}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        current_metrics = self.load_current_metrics()
        
        # Update project metadata
        current_metrics.setdefault("project", {})["updated"] = datetime.now().isoformat()
        
        # Collect new metrics
        coverage_metrics = self.collect_coverage_metrics()
        security_metrics = self.collect_security_metrics()
        git_metrics = self.collect_git_metrics()
        quality_metrics = self.collect_code_quality_metrics()
        
        # Update metrics
        current_metrics.setdefault("metrics", {})
        current_metrics["metrics"].setdefault("code_quality", {}).update(coverage_metrics)
        current_metrics["metrics"].setdefault("code_quality", {}).update(quality_metrics)
        current_metrics["metrics"].setdefault("security", {}).update(security_metrics)
        current_metrics["metrics"].setdefault("repository", {}).update(git_metrics)
        
        return current_metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# Project Metrics Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Code Quality
        code_quality = metrics.get("metrics", {}).get("code_quality", {})
        report.append("## Code Quality")
        report.append(f"- Coverage: {code_quality.get('coverage_current', 0):.1f}%")
        report.append(f"- Maintainability Index: {code_quality.get('maintainability_index', 0)}")
        report.append(f"- Code Issues: {code_quality.get('code_issues', 0)}")
        report.append("")
        
        # Security
        security = metrics.get("metrics", {}).get("security", {})
        report.append("## Security")
        report.append(f"- Vulnerabilities: {security.get('vulnerability_count', 0)}")
        report.append(f"- Security Hotspots: {security.get('security_hotspots', 0)}")
        report.append("")
        
        # Repository
        repository = metrics.get("metrics", {}).get("repository", {})
        report.append("## Repository Activity")
        report.append(f"- Contributors: {repository.get('contributors', 0)}")
        report.append(f"- Commits (last week): {repository.get('commits_per_week', 0)}")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    collector = MetricsCollector(project_root)
    
    print("Collecting project metrics...")
    metrics = collector.collect_all_metrics()
    
    print("Saving metrics...")
    collector.save_metrics(metrics)
    
    print("Generating report...")
    report = collector.generate_report(metrics)
    
    report_file = project_root / "metrics-report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Metrics collected and saved to {collector.metrics_file}")
    print(f"Report generated at {report_file}")
    
    # Print summary
    print("\n" + "="*50)
    print(report)


if __name__ == "__main__":
    main()