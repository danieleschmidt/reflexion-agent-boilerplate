#!/usr/bin/env python3
"""
Technical Debt Tracking and Analysis System
Automated detection, classification, and prioritization of technical debt
"""

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml


@dataclass
class DebtItem:
    """Represents a technical debt item."""
    id: str
    type: str  # code_complexity, security, performance, maintainability, documentation
    severity: str  # critical, high, medium, low
    title: str
    description: str
    file_path: str
    line_number: Optional[int]
    detected_at: str
    estimated_hours: float
    interest_rate: float  # How much debt grows over time
    hot_spot_multiplier: float  # Based on file change frequency
    debt_score: float
    remediation_strategy: str
    dependencies: List[str]
    tags: List[str]


@dataclass
class DebtMetrics:
    """Technical debt metrics and analytics."""
    total_debt_hours: float
    debt_by_category: Dict[str, float]
    debt_by_severity: Dict[str, int]
    hot_spots: List[str]
    debt_trend: str  # increasing, stable, decreasing
    interest_accrued: float
    remediation_backlog: int


class TechnicalDebtTracker:
    """Comprehensive technical debt tracking and analysis."""
    
    def __init__(self, repo_root: Path = None):
        """Initialize debt tracker."""
        self.repo_root = repo_root or Path.cwd()
        self.debt_items: List[DebtItem] = []
        self.file_change_frequency = {}
        self.complexity_cache = {}
        
    def analyze_technical_debt(self) -> DebtMetrics:
        """Perform comprehensive technical debt analysis."""
        print("🔍 Analyzing technical debt...")
        
        # Clear previous analysis
        self.debt_items = []
        
        # Analyze file change frequency for hot-spot detection
        self._analyze_file_churn()
        
        # Different types of debt analysis
        self._analyze_code_complexity()
        self._analyze_security_debt()
        self._analyze_performance_debt()
        self._analyze_maintainability_debt()
        self._analyze_documentation_debt()
        self._analyze_dependency_debt()
        
        # Calculate debt scores
        for item in self.debt_items:
            item.debt_score = self._calculate_debt_score(item)
        
        # Sort by debt score (highest first)
        self.debt_items.sort(key=lambda x: x.debt_score, reverse=True)
        
        # Generate metrics
        metrics = self._generate_metrics()
        
        print(f"✅ Found {len(self.debt_items)} debt items")
        print(f"📊 Total debt: {metrics.total_debt_hours:.1f} hours")
        
        return metrics
    
    def _analyze_file_churn(self):
        """Analyze file change frequency to identify hot spots."""
        try:
            # Get file change counts from git history
            result = subprocess.run([
                'git', 'log', '--name-only', '--pretty=format:', '--since=6.months.ago'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\\n')
                for file_path in files:
                    if file_path and file_path.endswith('.py'):
                        self.file_change_frequency[file_path] = self.file_change_frequency.get(file_path, 0) + 1
            
            # Normalize to multipliers (1.0 to 3.0)
            if self.file_change_frequency:
                max_changes = max(self.file_change_frequency.values())
                for file_path in self.file_change_frequency:
                    normalized = self.file_change_frequency[file_path] / max_changes
                    self.file_change_frequency[file_path] = 1.0 + (normalized * 2.0)
                    
        except Exception as e:
            print(f"⚠️ Failed to analyze file churn: {e}")
    
    def _analyze_code_complexity(self):
        """Analyze code complexity and identify complex functions/classes."""
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Analyze functions and methods
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        
                        if complexity > 10:  # High complexity threshold
                            severity = "critical" if complexity > 20 else "high" if complexity > 15 else "medium"
                            
                            self.debt_items.append(DebtItem(
                                id=f"complexity-{py_file.stem}-{node.name}-{node.lineno}",
                                type="code_complexity",
                                severity=severity,
                                title=f"High complexity function: {node.name}",
                                description=f"Function has cyclomatic complexity of {complexity} (threshold: 10)",
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=node.lineno,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=max(2.0, complexity * 0.2),
                                interest_rate=0.1,  # 10% growth rate
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,  # Will be calculated
                                remediation_strategy="Break down into smaller functions, reduce nesting",
                                dependencies=[],
                                tags=["complexity", "refactoring"]
                            ))
                    
                    elif isinstance(node, ast.ClassDef):
                        # Analyze class complexity (number of methods, inheritance depth)
                        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        
                        if len(methods) > 15:  # Large class
                            self.debt_items.append(DebtItem(
                                id=f"class-size-{py_file.stem}-{node.name}-{node.lineno}",
                                type="code_complexity",
                                severity="medium",
                                title=f"Large class: {node.name}",
                                description=f"Class has {len(methods)} methods (threshold: 15)",
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=node.lineno,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=4.0,
                                interest_rate=0.08,
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,
                                remediation_strategy="Split class responsibilities, extract sub-classes",
                                dependencies=[],
                                tags=["class-design", "single-responsibility"]
                            ))
                            
            except Exception as e:
                print(f"⚠️ Failed to analyze {py_file}: {e}")
    
    def _analyze_security_debt(self):
        """Analyze security-related technical debt."""
        # Look for common security anti-patterns
        security_patterns = {
            r'eval\\s*\\(': ("Use of eval() function", "critical", 4.0, "Replace with safer alternatives"),
            r'exec\\s*\\(': ("Use of exec() function", "critical", 4.0, "Replace with safer alternatives"),
            r'pickle\\.loads?\\s*\\(': ("Unsafe pickle usage", "high", 2.0, "Use safer serialization"),
            r'subprocess\\.call\\s*\\([^)]*shell\\s*=\\s*True': ("Shell injection risk", "critical", 3.0, "Use shell=False"),
            r'password\\s*=\\s*["\'][^"\']*["\']': ("Hardcoded password", "critical", 1.0, "Use environment variables"),
            r'api_key\\s*=\\s*["\'][^"\']*["\']': ("Hardcoded API key", "critical", 1.0, "Use environment variables"),
        }
        
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\\n')
                
                for pattern, (title, severity, hours, strategy) in security_patterns.items():
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            self.debt_items.append(DebtItem(
                                id=f"security-{py_file.stem}-{i}-{hash(line) % 10000}",
                                type="security",
                                severity=severity,
                                title=title,
                                description=f"Security issue detected: {line.strip()[:100]}",
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=i,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=hours,
                                interest_rate=0.15,  # Security debt grows fast
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,
                                remediation_strategy=strategy,
                                dependencies=[],
                                tags=["security", "vulnerability"]
                            ))
                            
            except Exception as e:
                print(f"⚠️ Failed to analyze security in {py_file}: {e}")
    
    def _analyze_performance_debt(self):
        """Analyze performance-related technical debt."""
        performance_patterns = {
            r'for\\s+\\w+\\s+in\\s+range\\s*\\(\\s*len\\s*\\(': ("Use enumerate instead of range(len())", "low", 0.5, "Replace with enumerate()"),
            r'\\.+\\s*\\+=\\s*\\w+': ("String concatenation in loop", "medium", 1.0, "Use join() or f-strings"),
            r'\\blist\\s*\\(\\s*filter\\s*\\(': ("Use list comprehension", "low", 0.5, "Replace with list comprehension"),
            r'\\blist\\s*\\(\\s*map\\s*\\(': ("Use list comprehension", "low", 0.5, "Replace with list comprehension"),
        }
        
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\\n')
                
                for pattern, (title, severity, hours, strategy) in performance_patterns.items():
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            self.debt_items.append(DebtItem(
                                id=f"performance-{py_file.stem}-{i}-{hash(line) % 10000}",
                                type="performance",
                                severity=severity,
                                title=title,
                                description=f"Performance issue: {line.strip()[:100]}",
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=i,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=hours,
                                interest_rate=0.05,  # Performance debt grows slowly
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,
                                remediation_strategy=strategy,
                                dependencies=[],
                                tags=["performance", "optimization"]
                            ))
                            
            except Exception as e:
                print(f"⚠️ Failed to analyze performance in {py_file}: {e}")
    
    def _analyze_maintainability_debt(self):
        """Analyze maintainability issues."""
        # Look for code smells and maintainability issues
        maintainability_patterns = {
            r'#\\s*(TODO|FIXME|HACK|XXX)\\s*:?\\s*(.*)': ("Technical debt comment", "medium", 1.0, "Address the noted issue"),
            r'pylint:\\s*disable': ("Pylint suppression", "low", 0.5, "Fix underlying issue instead of suppressing"),
            r'type:\\s*ignore': ("Type checking suppression", "low", 0.5, "Fix type annotations"),
            r'except\\s*:\\s*pass': ("Bare except clause", "medium", 1.0, "Handle specific exceptions"),
        }
        
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\\n')
                
                for pattern, (title, severity, hours, strategy) in maintainability_patterns.items():
                    for i, line in enumerate(lines, 1):
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            description = f"Maintainability issue: {line.strip()[:100]}"
                            if match.groups() and len(match.groups()) > 1:
                                description = f"Technical debt: {match.group(2).strip()}"
                            
                            self.debt_items.append(DebtItem(
                                id=f"maintainability-{py_file.stem}-{i}-{hash(line) % 10000}",
                                type="maintainability",
                                severity=severity,
                                title=title,
                                description=description,
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=i,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=hours,
                                interest_rate=0.08,
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,
                                remediation_strategy=strategy,
                                dependencies=[],
                                tags=["maintainability", "code-smell"]
                            ))
                            
            except Exception as e:
                print(f"⚠️ Failed to analyze maintainability in {py_file}: {e}")
    
    def _analyze_documentation_debt(self):
        """Analyze documentation-related debt."""
        for py_file in self.repo_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            severity = "medium" if isinstance(node, ast.ClassDef) else "low"
                            
                            self.debt_items.append(DebtItem(
                                id=f"docs-{py_file.stem}-{node.name}-{node.lineno}",
                                type="documentation",
                                severity=severity,
                                title=f"Missing docstring: {node.name}",
                                description=f"{'Class' if isinstance(node, ast.ClassDef) else 'Function'} lacks documentation",
                                file_path=str(py_file.relative_to(self.repo_root)),
                                line_number=node.lineno,
                                detected_at=datetime.now().isoformat(),
                                estimated_hours=0.25,
                                interest_rate=0.02,  # Documentation debt grows very slowly
                                hot_spot_multiplier=self._get_hot_spot_multiplier(str(py_file.relative_to(self.repo_root))),
                                debt_score=0.0,
                                remediation_strategy="Add comprehensive docstring with parameters and return values",
                                dependencies=[],
                                tags=["documentation", "docstring"]
                            ))
                            
            except Exception as e:
                print(f"⚠️ Failed to analyze documentation in {py_file}: {e}")
    
    def _analyze_dependency_debt(self):
        """Analyze dependency-related debt."""
        pyproject_path = self.repo_root / "pyproject.toml"
        requirements_path = self.repo_root / "requirements.txt"
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                if outdated:
                    total_outdated = len(outdated)
                    severity = "high" if total_outdated > 10 else "medium" if total_outdated > 5 else "low"
                    
                    self.debt_items.append(DebtItem(
                        id="dependency-outdated",
                        type="dependency",
                        severity=severity,
                        title=f"{total_outdated} outdated dependencies",
                        description=f"Found {total_outdated} packages that need updates",
                        file_path=str(pyproject_path.relative_to(self.repo_root)) if pyproject_path.exists() else "requirements.txt",
                        line_number=None,
                        detected_at=datetime.now().isoformat(),
                        estimated_hours=max(1.0, total_outdated * 0.2),
                        interest_rate=0.12,  # Dependencies grow stale quickly
                        hot_spot_multiplier=1.5,  # Dependencies affect whole project
                        debt_score=0.0,
                        remediation_strategy="Update dependencies systematically, test for breaking changes",
                        dependencies=outdated[:5],  # List first 5 as examples
                        tags=["dependencies", "updates"]
                    ))
                    
        except Exception as e:
            print(f"⚠️ Failed to analyze dependencies: {e}")
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.ExceptHandler,)):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, (ast.BoolOp,)):
                # Add complexity for and/or operations
                complexity += len(child.values) - 1
        
        return complexity
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            "build",
            "dist",
            ".terragon"
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _get_hot_spot_multiplier(self, file_path: str) -> float:
        """Get hot spot multiplier for a file based on change frequency."""
        return self.file_change_frequency.get(file_path, 1.0)
    
    def _calculate_debt_score(self, item: DebtItem) -> float:
        """Calculate comprehensive debt score for prioritization."""
        # Base score from severity
        severity_scores = {"critical": 100, "high": 75, "medium": 50, "low": 25}
        base_score = severity_scores.get(item.severity, 25)
        
        # Factor in estimated effort (higher effort = lower priority)
        effort_factor = max(0.2, 1.0 / (1.0 + item.estimated_hours * 0.1))
        
        # Factor in interest rate (how fast debt grows)
        interest_factor = 1.0 + item.interest_rate
        
        # Apply hot spot multiplier
        hot_spot_factor = item.hot_spot_multiplier
        
        # Calculate final score
        debt_score = base_score * effort_factor * interest_factor * hot_spot_factor
        
        return round(debt_score, 2)
    
    def _generate_metrics(self) -> DebtMetrics:
        """Generate comprehensive debt metrics."""
        if not self.debt_items:
            return DebtMetrics(0, {}, {}, [], "stable", 0, 0)
        
        total_hours = sum(item.estimated_hours for item in self.debt_items)
        
        # Debt by category
        debt_by_category = {}
        for item in self.debt_items:
            debt_by_category[item.type] = debt_by_category.get(item.type, 0) + item.estimated_hours
        
        # Debt by severity
        debt_by_severity = {}
        for item in self.debt_items:
            debt_by_severity[item.severity] = debt_by_severity.get(item.severity, 0) + 1
        
        # Hot spots (files with most debt)
        file_debt = {}
        for item in self.debt_items:
            file_debt[item.file_path] = file_debt.get(item.file_path, 0) + item.debt_score
        
        hot_spots = sorted(file_debt.items(), key=lambda x: x[1], reverse=True)[:5]
        hot_spots = [f"{path} ({score:.1f})" for path, score in hot_spots]
        
        # Calculate interest accrued (simplified)
        interest_accrued = sum(item.estimated_hours * item.interest_rate for item in self.debt_items)
        
        return DebtMetrics(
            total_debt_hours=total_hours,
            debt_by_category=debt_by_category,
            debt_by_severity=debt_by_severity,
            hot_spots=hot_spots,
            debt_trend="stable",  # Would need historical data
            interest_accrued=interest_accrued,
            remediation_backlog=len(self.debt_items)
        )
    
    def save_debt_report(self, output_path: str = ".terragon/DEBT_REPORT.md"):
        """Save comprehensive debt report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(exist_ok=True)
        
        metrics = self._generate_metrics()
        
        with open(report_path, 'w') as f:
            f.write("# 📊 Technical Debt Analysis Report\\n\\n")
            f.write(f"Generated: {datetime.now().isoformat()}\\n")
            f.write(f"Repository: {self.repo_root.name}\\n\\n")
            
            # Executive Summary
            f.write("## 📈 Executive Summary\\n\\n")
            f.write(f"- **Total Debt**: {metrics.total_debt_hours:.1f} hours\\n")
            f.write(f"- **Total Items**: {metrics.remediation_backlog}\\n")
            f.write(f"- **Interest Accrued**: {metrics.interest_accrued:.1f} hours/month\\n")
            f.write(f"- **Trend**: {metrics.debt_trend}\\n\\n")
            
            # Debt by Category
            f.write("## 🏷️ Debt by Category\\n\\n")
            for category, hours in sorted(metrics.debt_by_category.items()):
                percentage = (hours / metrics.total_debt_hours) * 100
                f.write(f"- **{category.replace('_', ' ').title()}**: {hours:.1f}h ({percentage:.1f}%)\\n")
            f.write("\\n")
            
            # Debt by Severity
            f.write("## ⚠️ Debt by Severity\\n\\n")
            for severity, count in sorted(metrics.debt_by_severity.items()):
                f.write(f"- **{severity.title()}**: {count} items\\n")
            f.write("\\n")
            
            # Hot Spots
            f.write("## 🔥 Hot Spots\\n\\n")
            f.write("Files with highest debt concentration:\\n\\n")
            for i, hot_spot in enumerate(metrics.hot_spots, 1):
                f.write(f"{i}. {hot_spot}\\n")
            f.write("\\n")
            
            # Top Priority Items
            f.write("## 🎯 Top Priority Items\\n\\n")
            f.write("| Priority | ID | Type | Severity | File | Effort | Score |\\n")
            f.write("|----------|-----|------|----------|------|--------|-------|\\n")
            
            for i, item in enumerate(self.debt_items[:15], 1):
                f.write(f"| {i} | {item.id[:20]}... | {item.type} | {item.severity} | {Path(item.file_path).name} | {item.estimated_hours}h | {item.debt_score} |\\n")
            
            f.write("\\n## 📋 Detailed Items\\n\\n")
            
            # Group by type
            by_type = {}
            for item in self.debt_items:
                if item.type not in by_type:
                    by_type[item.type] = []
                by_type[item.type].append(item)
            
            for debt_type, items in sorted(by_type.items()):
                f.write(f"### {debt_type.replace('_', ' ').title()} ({len(items)} items)\\n\\n")
                
                for item in items[:10]:  # Limit to top 10 per category
                    f.write(f"**{item.title}**\\n")
                    f.write(f"- File: `{item.file_path}`")
                    if item.line_number:
                        f.write(f":{item.line_number}")
                    f.write("\\n")
                    f.write(f"- Severity: {item.severity}\\n")
                    f.write(f"- Effort: {item.estimated_hours}h\\n")
                    f.write(f"- Score: {item.debt_score}\\n")
                    f.write(f"- Strategy: {item.remediation_strategy}\\n")
                    f.write(f"- Description: {item.description}\\n\\n")
                
                if len(items) > 10:
                    f.write(f"... and {len(items) - 10} more items\\n\\n")
        
        print(f"📝 Debt report saved to {report_path}")
    
    def export_debt_data(self, output_path: str = ".terragon/debt-data.json"):
        """Export debt data as JSON for integration with other tools."""
        data_path = Path(output_path)
        data_path.parent.mkdir(exist_ok=True)
        
        metrics = self._generate_metrics()
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.repo_root.name),
            "metrics": asdict(metrics),
            "debt_items": [asdict(item) for item in self.debt_items],
            "file_change_frequency": self.file_change_frequency
        }
        
        with open(data_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Debt data exported to {data_path}")


def main():
    """Main entry point for technical debt analysis."""
    print("🔍 Technical Debt Tracker")
    print("=" * 40)
    
    tracker = TechnicalDebtTracker()
    metrics = tracker.analyze_technical_debt()
    
    print(f"\\n📊 Analysis Results:")
    print(f"   Total Debt: {metrics.total_debt_hours:.1f} hours")
    print(f"   Critical Items: {metrics.debt_by_severity.get('critical', 0)}")
    print(f"   High Priority: {metrics.debt_by_severity.get('high', 0)}")
    print(f"   Interest Rate: {metrics.interest_accrued:.1f} hours/month")
    
    # Save reports
    tracker.save_debt_report()
    tracker.export_debt_data()
    
    print("\\n🏁 Technical debt analysis complete!")


if __name__ == "__main__":
    main()