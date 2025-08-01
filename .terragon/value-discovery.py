#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuous discovery and prioritization of highest-value SDLC improvements
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import re
from dataclasses import dataclass, asdict


@dataclass
class ValueItem:
    """Represents a discovered value item for potential execution."""
    id: str
    title: str
    description: str
    category: str
    estimated_effort_hours: float
    impact_score: float
    confidence_score: float
    ease_score: float
    technical_debt_impact: float
    security_priority: float
    wsjf_score: float
    ice_score: float
    composite_score: float
    discovered_at: str
    source: str
    files_affected: List[str]
    dependencies: List[str]
    risk_level: str


class AutonomousValueDiscovery:
    """Autonomous value discovery and prioritization engine."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        """Initialize the value discovery engine."""
        self.config_path = Path(config_path)
        self.repo_root = Path.cwd()
        self.config = self._load_config()
        self.discovered_items: List[ValueItem] = []
        self.execution_history: List[Dict] = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration if config file not found."""
        return {
            'scoring': {
                'weights': {'wsjf': 0.5, 'ice': 0.2, 'technical_debt': 0.2, 'security': 0.1},
                'thresholds': {'min_score': 15, 'max_risk': 0.7}
            },
            'discovery': {
                'sources': ['git_history', 'static_analysis', 'dependency_scan']
            }
        }
    
    def discover_value_items(self) -> List[ValueItem]:
        """Execute comprehensive value discovery across all sources."""
        print("🔍 Starting autonomous value discovery...")
        
        discovered_items = []
        
        # Source 1: Git history analysis
        discovered_items.extend(self._discover_from_git_history())
        
        # Source 2: Static analysis
        discovered_items.extend(self._discover_from_static_analysis())
        
        # Source 3: Dependency scanning
        discovered_items.extend(self._discover_from_dependencies())
        
        # Source 4: Performance analysis
        discovered_items.extend(self._discover_from_performance())
        
        # Source 5: Documentation gaps
        discovered_items.extend(self._discover_from_documentation())
        
        # Calculate composite scores for all items
        for item in discovered_items:
            item.composite_score = self._calculate_composite_score(item)
        
        # Sort by composite score (highest first)
        discovered_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.discovered_items = discovered_items
        print(f"✅ Discovered {len(discovered_items)} value items")
        
        return discovered_items
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover value items from git history analysis."""
        items = []
        
        try:
            # Find TODO, FIXME, HACK comments
            result = subprocess.run([
                'git', 'grep', '-n', '-i', 
                '-E', r'(TODO|FIXME|HACK|XXX|BUG|OPTIMIZE)'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\\n')[:10]:  # Limit to first 10
                    if ':' in line:
                        file_path, content = line.split(':', 1)
                        items.append(ValueItem(
                            id=f"git-todo-{len(items)}",
                            title=f"Address TODO/FIXME in {Path(file_path).name}",
                            description=f"Code comment indicates needed improvement: {content.strip()}",
                            category="technical_debt",
                            estimated_effort_hours=1.0,
                            impact_score=6.0,
                            confidence_score=8.0,
                            ease_score=7.0,
                            technical_debt_impact=15.0,
                            security_priority=3.0,
                            wsjf_score=0.0,  # Will be calculated
                            ice_score=0.0,   # Will be calculated
                            composite_score=0.0,  # Will be calculated
                            discovered_at=datetime.now().isoformat(),
                            source="git_history",
                            files_affected=[file_path],
                            dependencies=[],
                            risk_level="low"
                        ))
            
            # Analyze commit messages for patterns
            recent_commits = subprocess.run([
                'git', 'log', '--oneline', '-20', 
                '--grep=fix', '--grep=bug', '--grep=hack', '--grep=temp', '--grep=quick'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if recent_commits.returncode == 0:
                commit_lines = recent_commits.stdout.strip().split('\\n')
                if len(commit_lines) > 2:  # Multiple quick fixes suggest debt
                    items.append(ValueItem(
                        id="git-pattern-quick-fixes",
                        title="Multiple quick fixes detected - refactoring needed",
                        description=f"Found {len(commit_lines)} recent commits with quick fix patterns",
                        category="technical_debt",
                        estimated_effort_hours=4.0,
                        impact_score=7.0,
                        confidence_score=6.0,
                        ease_score=5.0,
                        technical_debt_impact=25.0,
                        security_priority=2.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        discovered_at=datetime.now().isoformat(),
                        source="git_history",
                        files_affected=[],
                        dependencies=[],
                        risk_level="medium"
                    ))
                    
        except Exception as e:
            print(f"⚠️ Git history analysis failed: {e}")
        
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover value items from static code analysis."""
        items = []
        
        try:
            # Run ruff to find code quality issues
            result = subprocess.run([
                'ruff', 'check', 'src/', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                try:
                    ruff_issues = json.loads(result.stdout)
                    if len(ruff_issues) > 0:
                        items.append(ValueItem(
                            id="static-ruff-issues",
                            title=f"Fix {len(ruff_issues)} code quality issues",
                            description=f"Ruff detected {len(ruff_issues)} code quality improvements",
                            category="code_quality",
                            estimated_effort_hours=max(0.5, len(ruff_issues) * 0.1),
                            impact_score=5.0,
                            confidence_score=9.0,
                            ease_score=8.0,
                            technical_debt_impact=10.0,
                            security_priority=2.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            discovered_at=datetime.now().isoformat(),
                            source="static_analysis",
                            files_affected=[issue.get('filename', '') for issue in ruff_issues[:5]],
                            dependencies=[],
                            risk_level="low"
                        ))
                except json.JSONDecodeError:
                    pass
            
            # Check test coverage
            try:
                coverage_result = subprocess.run([
                    'python', '-m', 'pytest', '--cov=src/reflexion', 
                    '--cov-report=json', '--cov-report=term'
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                coverage_json_path = self.repo_root / 'coverage.json'
                if coverage_json_path.exists():
                    with open(coverage_json_path, 'r') as f:
                        coverage_data = json.load(f)
                        total_coverage = coverage_data.get('totals', {}).get('percent_covered', 100)
                        
                        if total_coverage < 85:
                            items.append(ValueItem(
                                id="static-test-coverage",
                                title=f"Improve test coverage from {total_coverage:.1f}% to 90%",
                                description=f"Current test coverage is {total_coverage:.1f}%, target is 90%",
                                category="testing",
                                estimated_effort_hours=3.0,
                                impact_score=8.0,
                                confidence_score=7.0,
                                ease_score=6.0,
                                technical_debt_impact=20.0,
                                security_priority=3.0,
                                wsjf_score=0.0,
                                ice_score=0.0,
                                composite_score=0.0,
                                discovered_at=datetime.now().isoformat(),
                                source="static_analysis",
                                files_affected=["tests/"],
                                dependencies=[],
                                risk_level="low"
                            ))
            except Exception:
                pass
                
        except Exception as e:
            print(f"⚠️ Static analysis failed: {e}")
        
        return items
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover value items from dependency analysis."""
        items = []
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and result.stdout:
                try:
                    outdated = json.loads(result.stdout)
                    if len(outdated) > 0:
                        items.append(ValueItem(
                            id="deps-outdated-packages",
                            title=f"Update {len(outdated)} outdated dependencies",
                            description=f"Found {len(outdated)} packages that can be updated",
                            category="dependency_management",
                            estimated_effort_hours=1.5,
                            impact_score=6.0,
                            confidence_score=8.0,
                            ease_score=7.0,
                            technical_debt_impact=12.0,
                            security_priority=5.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            discovered_at=datetime.now().isoformat(),
                            source="dependency_scan",
                            files_affected=["pyproject.toml"],
                            dependencies=[],
                            risk_level="low"
                        ))
                except json.JSONDecodeError:
                    pass
            
            # Run safety check for vulnerabilities
            safety_result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if safety_result.stdout:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    if len(safety_data) > 0:
                        items.append(ValueItem(
                            id="deps-security-vulnerabilities",
                            title=f"Fix {len(safety_data)} security vulnerabilities",
                            description=f"Safety check found {len(safety_data)} known vulnerabilities",
                            category="security",
                            estimated_effort_hours=2.0,
                            impact_score=9.0,
                            confidence_score=9.0,
                            ease_score=6.0,
                            technical_debt_impact=30.0,
                            security_priority=10.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            discovered_at=datetime.now().isoformat(),
                            source="dependency_scan",
                            files_affected=["pyproject.toml"],
                            dependencies=[],
                            risk_level="high"
                        ))
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"⚠️ Dependency analysis failed: {e}")
        
        return items
    
    def _discover_from_performance(self) -> List[ValueItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Check for large files that might need optimization
        try:
            result = subprocess.run([
                'find', 'src/', '-name', '*.py', '-size', '+1000c'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                large_files = result.stdout.strip().split('\\n')
                large_files = [f for f in large_files if f]  # Remove empty strings
                
                if len(large_files) > 5:
                    items.append(ValueItem(
                        id="perf-large-files",
                        title=f"Review {len(large_files)} large Python files for optimization",
                        description=f"Found {len(large_files)} files >1KB that may benefit from refactoring",
                        category="performance",
                        estimated_effort_hours=2.5,
                        impact_score=5.0,
                        confidence_score=5.0,
                        ease_score=6.0,
                        technical_debt_impact=15.0,
                        security_priority=1.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        discovered_at=datetime.now().isoformat(),
                        source="performance_analysis",
                        files_affected=large_files[:5],
                        dependencies=[],
                        risk_level="low"
                    ))
        except Exception as e:
            print(f"⚠️ Performance analysis failed: {e}")
        
        return items
    
    def _discover_from_documentation(self) -> List[ValueItem]:
        """Discover documentation gaps and improvements."""
        items = []
        
        # Check for missing docstrings
        try:
            result = subprocess.run([
                'python', '-c', 
                '''
import ast
import os
missing_docs = 0
for root, dirs, files in os.walk("src/"):
    for file in files:
        if file.endswith(".py"):
            try:
                with open(os.path.join(root, file), "r") as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if not ast.get_docstring(node):
                                missing_docs += 1
            except: pass
print(missing_docs)
                '''
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                missing_docs = int(result.stdout.strip() or 0)
                if missing_docs > 0:
                    items.append(ValueItem(
                        id="docs-missing-docstrings",
                        title=f"Add docstrings to {missing_docs} functions/classes",
                        description=f"Found {missing_docs} functions or classes without docstrings",
                        category="documentation",
                        estimated_effort_hours=max(0.5, missing_docs * 0.1),
                        impact_score=4.0,
                        confidence_score=8.0,
                        ease_score=9.0,
                        technical_debt_impact=8.0,
                        security_priority=1.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        discovered_at=datetime.now().isoformat(),
                        source="documentation_analysis",
                        files_affected=["src/"],
                        dependencies=[],
                        risk_level="low"
                    ))
        except Exception as e:
            print(f"⚠️ Documentation analysis failed: {e}")
        
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate composite score using WSJF, ICE, and technical debt weighting."""
        
        # WSJF (Weighted Shortest Job First)
        cost_of_delay = (
            item.impact_score * 0.4 +          # User/Business Value
            item.security_priority * 0.3 +     # Time Criticality  
            item.technical_debt_impact * 0.2 + # Risk Reduction
            item.impact_score * 0.1             # Opportunity Enablement
        )
        
        job_size = max(item.estimated_effort_hours, 0.5)  # Avoid division by zero
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE (Impact, Confidence, Ease)
        item.ice_score = item.impact_score * item.confidence_score * item.ease_score
        
        # Get weights from config
        weights = self.config.get('scoring', {}).get('weights', {})
        w_wsjf = weights.get('wsjf', 0.5)
        w_ice = weights.get('ice', 0.2) 
        w_debt = weights.get('technical_debt', 0.2)
        w_security = weights.get('security', 0.1)
        
        # Normalize scores to 0-100 scale
        normalized_wsjf = min(item.wsjf_score * 10, 100)
        normalized_ice = min(item.ice_score / 10, 100) 
        normalized_debt = min(item.technical_debt_impact, 100)
        normalized_security = min(item.security_priority * 10, 100)
        
        # Calculate composite score
        composite = (
            w_wsjf * normalized_wsjf +
            w_ice * normalized_ice +
            w_debt * normalized_debt +
            w_security * normalized_security
        )
        
        # Apply risk penalty
        if item.risk_level == "high":
            composite *= 0.8
        elif item.risk_level == "medium":
            composite *= 0.9
        
        return round(composite, 2)
    
    def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the next highest-value item for execution."""
        if not self.discovered_items:
            self.discover_value_items()
        
        # Filter items above minimum threshold
        min_score = self.config.get('scoring', {}).get('thresholds', {}).get('min_score', 15)
        candidate_items = [item for item in self.discovered_items if item.composite_score >= min_score]
        
        if not candidate_items:
            return None
        
        # Return highest scored item
        return candidate_items[0]
    
    def save_backlog(self, output_path: str = ".terragon/BACKLOG.md"):
        """Save discovered items to a backlog file."""
        backlog_path = Path(output_path)
        backlog_path.parent.mkdir(exist_ok=True)
        
        with open(backlog_path, 'w') as f:
            f.write("# 📊 Autonomous Value Backlog\\n\\n")
            f.write(f"Last Updated: {datetime.now().isoformat()}\\n")
            f.write(f"Total Items Discovered: {len(self.discovered_items)}\\n\\n")
            
            if self.discovered_items:
                next_item = self.get_next_best_value_item()
                if next_item:
                    f.write("## 🎯 Next Best Value Item\\n")
                    f.write(f"**[{next_item.id}] {next_item.title}**\\n")
                    f.write(f"- **Composite Score**: {next_item.composite_score}\\n")
                    f.write(f"- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.1f} | **Tech Debt**: {next_item.technical_debt_impact}\\n")
                    f.write(f"- **Estimated Effort**: {next_item.estimated_effort_hours} hours\\n")
                    f.write(f"- **Expected Impact**: {next_item.description}\\n\\n")
                
                f.write("## 📋 Top 10 Backlog Items\\n\\n")
                f.write("| Rank | ID | Title | Score | Category | Est. Hours |\\n")
                f.write("|------|-----|--------|---------|----------|------------|\\n")
                
                for i, item in enumerate(self.discovered_items[:10], 1):
                    f.write(f"| {i} | {item.id} | {item.title[:50]}{'...' if len(item.title) > 50 else ''} | {item.composite_score} | {item.category} | {item.estimated_effort_hours} |\\n")
                
                f.write("\\n## 📈 Value Metrics\\n")
                f.write(f"- **Items by Category**:\\n")
                
                categories = {}
                for item in self.discovered_items:
                    categories[item.category] = categories.get(item.category, 0) + 1
                
                for category, count in sorted(categories.items()):
                    f.write(f"  - {category}: {count}\\n")
                
                f.write(f"\\n- **Average Composite Score**: {sum(item.composite_score for item in self.discovered_items) / len(self.discovered_items):.2f}\\n")
                f.write(f"- **Total Estimated Effort**: {sum(item.estimated_effort_hours for item in self.discovered_items):.1f} hours\\n")
        
        print(f"📝 Backlog saved to {backlog_path}")
    
    def export_metrics(self, output_path: str = ".terragon/value-metrics.json"):
        """Export value metrics to JSON for tracking and analysis."""
        metrics_path = Path(output_path)
        metrics_path.parent.mkdir(exist_ok=True)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "repository": self.config.get("metadata", {}).get("repository", "unknown"),
            "maturity_level": self.config.get("metadata", {}).get("maturity_level", "unknown"),
            "discovery_summary": {
                "total_items": len(self.discovered_items),
                "avg_composite_score": sum(item.composite_score for item in self.discovered_items) / len(self.discovered_items) if self.discovered_items else 0,
                "total_estimated_effort": sum(item.estimated_effort_hours for item in self.discovered_items),
                "categories": {}
            },
            "top_items": [asdict(item) for item in self.discovered_items[:5]],
            "execution_history": self.execution_history
        }
        
        # Category breakdown
        for item in self.discovered_items:
            cat = item.category
            if cat not in metrics["discovery_summary"]["categories"]:
                metrics["discovery_summary"]["categories"][cat] = {"count": 0, "total_score": 0}
            metrics["discovery_summary"]["categories"][cat]["count"] += 1
            metrics["discovery_summary"]["categories"][cat]["total_score"] += item.composite_score
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📊 Metrics exported to {metrics_path}")


def main():
    """Main entry point for autonomous value discovery."""
    print("🚀 Terragon Autonomous Value Discovery Engine")
    print("=" * 50)
    
    discovery = AutonomousValueDiscovery()
    
    # Discover value items
    items = discovery.discover_value_items()
    
    # Get next best item
    next_item = discovery.get_next_best_value_item()
    
    if next_item:
        print(f"\\n🎯 Next Best Value Item:")
        print(f"   ID: {next_item.id}")
        print(f"   Title: {next_item.title}")
        print(f"   Score: {next_item.composite_score}")
        print(f"   Effort: {next_item.estimated_effort_hours}h")
        print(f"   Risk: {next_item.risk_level}")
    else:
        print("\\n✅ No high-value items found above threshold")
    
    # Save backlog and metrics
    discovery.save_backlog()
    discovery.export_metrics()
    
    print("\\n🏁 Value discovery complete!")


if __name__ == "__main__":
    main()