#!/usr/bin/env python3
"""
Repository health monitoring script.
Checks various aspects of repository health and provides recommendations.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class RepositoryHealthChecker:
    """Monitors and reports on repository health metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "checks": {},
            "recommendations": []
        }
    
    def check_documentation_health(self) -> Dict[str, Any]:
        """Check documentation completeness and quality."""
        score = 0
        details = []
        
        # Check for essential documentation files
        essential_docs = [
            "README.md", "CONTRIBUTING.md", "LICENSE", "SECURITY.md",
            "CODE_OF_CONDUCT.md", "CHANGELOG.md"
        ]
        
        existing_docs = []
        for doc in essential_docs:
            if (self.project_root / doc).exists():
                existing_docs.append(doc)
                score += 15
        
        details.append(f"Essential documentation: {len(existing_docs)}/{len(essential_docs)}")
        
        # Check for project-specific documentation
        if (self.project_root / "docs").exists():
            docs_dir = self.project_root / "docs"
            doc_files = list(docs_dir.rglob("*.md"))
            if len(doc_files) >= 5:
                score += 10
                details.append(f"Extended documentation: {len(doc_files)} files")
            else:
                details.append(f"Limited extended documentation: {len(doc_files)} files")
        
        # Check README quality
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            if len(readme_content) > 1000:
                score += 10
                details.append("README is comprehensive")
            else:
                details.append("README could be more detailed")
        
        return {
            "score": min(score, 100),
            "details": details,
            "status": "excellent" if score >= 80 else "good" if score >= 60 else "needs_improvement"
        }
    
    def check_code_organization(self) -> Dict[str, Any]:
        """Check code organization and structure."""
        score = 0
        details = []
        
        # Check for proper project structure
        expected_dirs = ["src", "tests", "docs", "scripts"]
        existing_dirs = []
        
        for directory in expected_dirs:
            if (self.project_root / directory).exists():
                existing_dirs.append(directory)
                score += 20
        
        details.append(f"Project structure: {len(existing_dirs)}/{len(expected_dirs)} directories")
        
        # Check for configuration files
        config_files = [
            "pyproject.toml", ".gitignore", ".editorconfig",
            ".pre-commit-config.yaml", "pytest.ini"
        ]
        
        existing_configs = []
        for config in config_files:
            if (self.project_root / config).exists():
                existing_configs.append(config)
                score += 4
        
        details.append(f"Configuration files: {len(existing_configs)}/{len(config_files)}")
        
        return {
            "score": min(score, 100),
            "details": details,
            "status": "excellent" if score >= 80 else "good" if score >= 60 else "needs_improvement"
        }
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage and quality."""
        score = 0
        details = []
        
        # Check if tests directory exists and has tests
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("test_*.py"))
            if test_files:
                score += 30
                details.append(f"Test files: {len(test_files)}")
                
                # Check for different types of tests
                test_types = ["unit", "integration", "performance"]
                existing_test_types = []
                
                for test_type in test_types:
                    if (tests_dir / test_type).exists():
                        existing_test_types.append(test_type)
                        score += 20
                
                details.append(f"Test types: {existing_test_types}")
            else:
                details.append("Tests directory exists but no test files found")
        else:
            details.append("No tests directory found")
        
        # Try to get actual coverage if available
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=term-missing", "--quiet"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if "TOTAL" in result.stdout:
                # Extract coverage percentage
                lines = result.stdout.split('\n')
                for line in lines:
                    if "TOTAL" in line:
                        parts = line.split()
                        if len(parts) >= 4 and parts[-1].endswith('%'):
                            coverage = int(parts[-1].rstrip('%'))
                            score = max(score, coverage)
                            details.append(f"Coverage: {coverage}%")
                            break
        except Exception as e:
            details.append(f"Could not run coverage analysis: {str(e)[:50]}")
        
        return {
            "score": min(score, 100),
            "details": details,
            "status": "excellent" if score >= 80 else "good" if score >= 60 else "needs_improvement"
        }
    
    def check_git_health(self) -> Dict[str, Any]:
        """Check git repository health."""
        score = 0
        details = []
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                score += 20
                details.append("Git repository initialized")
                
                # Check for clean working directory
                if not result.stdout.strip():
                    score += 20
                    details.append("Working directory is clean")
                else:
                    details.append("Working directory has uncommitted changes")
                
                # Check for recent commits
                result = subprocess.run(
                    ["git", "log", "--oneline", "-10"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    commit_count = len(result.stdout.strip().split('\n'))
                    if commit_count >= 5:
                        score += 20
                        details.append(f"Recent activity: {commit_count} recent commits")
                    else:
                        details.append(f"Limited activity: {commit_count} recent commits")
                
                # Check for branches
                result = subprocess.run(
                    ["git", "branch", "-a"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    branches = result.stdout.strip().split('\n')
                    if len(branches) > 1:
                        score += 20
                        details.append(f"Branch strategy: {len(branches)} branches")
                    else:
                        details.append("Single branch (consider branching strategy)")
                
                # Check for .gitignore
                if (self.project_root / ".gitignore").exists():
                    score += 20
                    details.append(".gitignore file present")
                else:
                    details.append(".gitignore file missing")
                    
        except Exception as e:
            details.append(f"Git check failed: {str(e)[:50]}")
        
        return {
            "score": min(score, 100),
            "details": details,
            "status": "excellent" if score >= 80 else "good" if score >= 60 else "needs_improvement"
        }
    
    def check_security_posture(self) -> Dict[str, Any]:
        """Check security configuration and practices."""
        score = 0
        details = []
        
        # Check for security-related files
        security_files = ["SECURITY.md", ".bandit", "bandit.toml"]
        existing_security_files = []
        
        for file in security_files:
            if (self.project_root / file).exists():
                existing_security_files.append(file)
                score += 25
        
        details.append(f"Security files: {existing_security_files}")
        
        # Check for pre-commit hooks
        if (self.project_root / ".pre-commit-config.yaml").exists():
            score += 25
            details.append("Pre-commit hooks configured")
        else:
            details.append("Pre-commit hooks not configured")
        
        # Check for dependency scanning configuration
        if (self.project_root / "pyproject.toml").exists():
            score += 25
            details.append("Dependency management configured")
        
        return {
            "score": min(score, 100),
            "details": details,
            "status": "excellent" if score >= 80 else "good" if score >= 60 else "needs_improvement"
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on health checks."""
        recommendations = []
        
        for check_name, check_result in self.health_report["checks"].items():
            if check_result["status"] == "needs_improvement":
                if check_name == "documentation":
                    recommendations.append("📚 Improve documentation: Add missing essential files and expand README")
                elif check_name == "code_organization":
                    recommendations.append("🏗️  Improve code organization: Add missing directories and configuration files")
                elif check_name == "test_coverage":
                    recommendations.append("🧪 Improve test coverage: Add more tests and aim for 80%+ coverage")
                elif check_name == "git_health":
                    recommendations.append("🔧 Improve git practices: Ensure clean commits and proper branching")
                elif check_name == "security":
                    recommendations.append("🔒 Improve security: Add security documentation and scanning tools")
        
        # General recommendations based on overall score
        overall_score = self.health_report["overall_score"]
        if overall_score < 70:
            recommendations.append("⚠️  Overall repository health needs attention - focus on the areas marked as 'needs_improvement'")
        elif overall_score < 85:
            recommendations.append("✨ Good repository health - consider addressing remaining improvement areas")
        else:
            recommendations.append("🎉 Excellent repository health - maintain current practices!")
        
        return recommendations
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run all health checks and generate report."""
        print("Running repository health checks...")
        
        # Run individual checks
        checks = {
            "documentation": self.check_documentation_health,
            "code_organization": self.check_code_organization,
            "test_coverage": self.check_test_coverage,
            "git_health": self.check_git_health,
            "security": self.check_security_posture
        }
        
        total_score = 0
        for check_name, check_function in checks.items():
            print(f"  Checking {check_name}...")
            result = check_function()
            self.health_report["checks"][check_name] = result
            total_score += result["score"]
        
        # Calculate overall score
        self.health_report["overall_score"] = total_score / len(checks)
        self.health_report["recommendations"] = self.generate_recommendations()
        
        return self.health_report
    
    def save_report(self, report: Dict[str, Any]) -> None:
        """Save health report to file."""
        report_file = self.project_root / "repository-health-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create a markdown report
        md_report = self.generate_markdown_report(report)
        md_file = self.project_root / "repository-health-report.md"
        with open(md_file, 'w') as f:
            f.write(md_report)
        
        print(f"Health report saved to {report_file} and {md_file}")
    
    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a markdown version of the health report."""
        lines = []
        lines.append("# Repository Health Report")
        lines.append(f"Generated: {report['timestamp']}")
        lines.append(f"Overall Score: **{report['overall_score']:.1f}/100**")
        lines.append("")
        
        # Overall status
        overall_score = report['overall_score']
        if overall_score >= 85:
            lines.append("🎉 **Status: Excellent** - Repository is in great shape!")
        elif overall_score >= 70:
            lines.append("✅ **Status: Good** - Repository is well-maintained with room for improvement.")
        else:
            lines.append("⚠️  **Status: Needs Improvement** - Several areas require attention.")
        
        lines.append("")
        lines.append("## Detailed Results")
        lines.append("")
        
        # Individual check results
        for check_name, check_result in report["checks"].items():
            status_emoji = {
                "excellent": "🟢",
                "good": "🟡", 
                "needs_improvement": "🔴"
            }.get(check_result["status"], "⚪")
            
            lines.append(f"### {check_name.title().replace('_', ' ')} {status_emoji}")
            lines.append(f"Score: {check_result['score']}/100")
            lines.append("")
            
            for detail in check_result["details"]:
                lines.append(f"- {detail}")
            lines.append("")
        
        # Recommendations
        if report["recommendations"]:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")
        
        lines.append("---")
        lines.append("*This report was generated automatically by the repository health checker.*")
        
        return "\n".join(lines)


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    checker = RepositoryHealthChecker(project_root)
    
    report = checker.run_health_check()
    checker.save_report(report)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Repository Health Score: {report['overall_score']:.1f}/100")
    
    if report['overall_score'] >= 85:
        print("🎉 Excellent! Repository is in great shape!")
    elif report['overall_score'] >= 70:
        print("✅ Good! Repository is well-maintained.")
    else:
        print("⚠️  Needs improvement. Check the report for details.")
    
    print("="*60)


if __name__ == "__main__":
    main()