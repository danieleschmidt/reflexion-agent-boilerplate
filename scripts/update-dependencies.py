#!/usr/bin/env python3
"""
Automated dependency update script.
Checks for outdated dependencies and provides update recommendations.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class DependencyUpdater:
    """Manages dependency updates for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_toml = project_root / "pyproject.toml"
        
    def check_outdated_packages(self) -> List[Dict]:
        """Check for outdated Python packages."""
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "list", "--outdated", "--format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"Error checking outdated packages: {e}")
        
        return []
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety."""
        try:
            result = subprocess.run(
                ["python", "-m", "safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"Error checking security vulnerabilities: {e}")
        
        return []
    
    def update_pre_commit_hooks(self) -> bool:
        """Update pre-commit hooks to latest versions."""
        try:
            result = subprocess.run(
                ["pre-commit", "autoupdate"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Pre-commit hooks updated successfully")
                return True
            else:
                print(f"Error updating pre-commit hooks: {result.stderr}")
        except Exception as e:
            print(f"Error updating pre-commit hooks: {e}")
        
        return False
    
    def generate_update_report(self, outdated: List[Dict], vulnerabilities: List[Dict]) -> str:
        """Generate a dependency update report."""
        report = []
        report.append("# Dependency Update Report")
        report.append(f"Generated: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
        report.append("")
        
        # Outdated packages
        if outdated:
            report.append("## Outdated Packages")
            report.append("")
            for package in outdated:
                name = package.get("name", "unknown")
                current = package.get("version", "unknown")
                latest = package.get("latest_version", "unknown")
                report.append(f"- **{name}**: {current} → {latest}")
            report.append("")
        else:
            report.append("## Outdated Packages")
            report.append("All packages are up to date! 🎉")
            report.append("")
        
        # Security vulnerabilities
        if vulnerabilities:
            report.append("## Security Vulnerabilities")
            report.append("")
            for vuln in vulnerabilities:
                package = vuln.get("package", "unknown")
                installed = vuln.get("installed_version", "unknown")
                advisory = vuln.get("advisory", "No details available")
                report.append(f"- **{package}** ({installed}): {advisory}")
            report.append("")
        else:
            report.append("## Security Vulnerabilities")
            report.append("No known security vulnerabilities found! 🔒")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if outdated or vulnerabilities:
            report.append("1. Review the outdated packages and security vulnerabilities above")
            report.append("2. Test updates in a development environment first")
            report.append("3. Update dependencies incrementally to avoid conflicts")
            report.append("4. Run the full test suite after each update")
            report.append("5. Update documentation if any breaking changes are introduced")
        else:
            report.append("No actions needed - all dependencies are current and secure!")
        
        report.append("")
        report.append("## Update Commands")
        if outdated:
            report.append("```bash")
            report.append("# Update specific packages")
            for package in outdated[:5]:  # Show first 5 packages
                name = package.get("name", "unknown")
                report.append(f"pip install --upgrade {name}")
            report.append("")
            report.append("# Or update all packages (use with caution)")
            report.append("pip list --outdated --format=freeze | grep -v '^\\-e' | cut -d = -f 1 | xargs -n1 pip install -U")
            report.append("```")
        
        return "\n".join(report)
    
    def run_update_check(self) -> None:
        """Run the complete dependency update check."""
        print("Checking for outdated packages...")
        outdated = self.check_outdated_packages()
        
        print("Checking for security vulnerabilities...")
        vulnerabilities = self.check_security_vulnerabilities()
        
        print("Updating pre-commit hooks...")
        self.update_pre_commit_hooks()
        
        print("Generating update report...")
        report = self.generate_update_report(outdated, vulnerabilities)
        
        # Save report
        report_file = self.project_root / "dependency-update-report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Dependency update report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"Found {len(outdated)} outdated packages")
        print(f"Found {len(vulnerabilities)} security vulnerabilities")
        print("="*60)
        
        if outdated or vulnerabilities:
            print("\nIMMEDIATE ACTION REQUIRED!")
            if vulnerabilities:
                print("⚠️  Security vulnerabilities found!")
            if outdated:
                print(f"📦 {len(outdated)} packages need updates")
        else:
            print("\n✅ All dependencies are up to date and secure!")


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    updater = DependencyUpdater(project_root)
    
    updater.run_update_check()


if __name__ == "__main__":
    main()