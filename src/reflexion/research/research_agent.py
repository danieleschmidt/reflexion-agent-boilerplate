"""Research-oriented reflexion agent with hypothesis testing capabilities."""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

from ..core.agent import ReflexionAgent
from ..core.types import ReflectionType, ReflexionResult
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentCondition, ExperimentType


class ResearchObjectiveType(Enum):
    """Types of research objectives."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PARAMETER_TUNING = "parameter_tuning"
    FAILURE_ANALYSIS = "failure_analysis"
    SCALABILITY_STUDY = "scalability_study"
    NOVELTY_DISCOVERY = "novelty_discovery"


@dataclass
class ResearchObjective:
    """Defines a research objective with hypotheses."""
    name: str
    description: str
    objective_type: ResearchObjectiveType
    hypotheses: List[str]
    success_metrics: List[str]
    expected_outcomes: List[str]
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


@dataclass
class ResearchFinding:
    """Represents a research finding with evidence."""
    objective: ResearchObjective
    hypothesis: str
    supported: bool
    confidence: float
    evidence: Dict[str, Any]
    statistical_significance: Optional[float]
    effect_size: Optional[float]
    practical_significance: str
    limitations: List[str]
    future_work: List[str]
    timestamp: str


class ResearchAgent:
    """Advanced reflexion agent specialized for research and experimentation."""
    
    def __init__(
        self,
        base_llm: str = "gpt-4",
        experiment_runner: Optional[ExperimentRunner] = None,
        output_dir: str = "./research_output"
    ):
        """Initialize research agent.
        
        Args:
            base_llm: Base language model for reflexion
            experiment_runner: Experimental framework
            output_dir: Directory for research outputs
        """
        self.base_llm = base_llm
        self.experiment_runner = experiment_runner or ExperimentRunner(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Research state
        self.active_objectives: List[ResearchObjective] = []
        self.completed_studies: List[Dict[str, Any]] = []
        self.findings_database: List[ResearchFinding] = []
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    async def conduct_research_study(
        self,
        objective: ResearchObjective,
        test_scenarios: List[str],
        num_trials: int = 50
    ) -> List[ResearchFinding]:
        """Conduct a comprehensive research study."""
        self.logger.info(f"Starting research study: {objective.name}")
        
        # Add to active objectives
        self.active_objectives.append(objective)
        
        findings = []
        
        try:
            # Generate experimental conditions based on hypotheses
            conditions = await self._generate_experimental_conditions(objective)
            
            # Create experiment configuration
            experiment_config = ExperimentConfig(
                name=f"research_{objective.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Research study: {objective.description}",
                experiment_type=self._map_objective_to_experiment_type(objective.objective_type),
                conditions=conditions,
                test_tasks=test_scenarios,
                success_criteria=objective.success_metrics,
                metrics=self._get_research_metrics(objective),
                num_trials=num_trials
            )
            
            # Run experiment
            experiment_result = await self.experiment_runner.run_experiment(experiment_config)
            
            # Analyze results for each hypothesis
            for hypothesis in objective.hypotheses:
                finding = await self._analyze_hypothesis(
                    objective, hypothesis, experiment_result
                )
                findings.append(finding)
                self.findings_database.append(finding)
            
            # Generate meta-analysis
            meta_findings = await self._generate_meta_analysis(objective, findings)
            findings.extend(meta_findings)
            
            # Save research report
            await self._generate_research_report(objective, findings, experiment_result)
            
            # Mark study as completed
            study_summary = {
                "objective": asdict(objective),
                "findings_count": len(findings),
                "experiment_id": experiment_config.name,
                "completed_at": datetime.now().isoformat()
            }
            self.completed_studies.append(study_summary)
            
            self.logger.info(
                f"Research study completed: {objective.name} "
                f"with {len(findings)} findings"
            )
            
        except Exception as e:
            self.logger.error(f"Research study failed: {e}")
            raise
        finally:
            # Remove from active objectives
            if objective in self.active_objectives:
                self.active_objectives.remove(objective)
        
        return findings
    
    async def _generate_experimental_conditions(
        self,
        objective: ResearchObjective
    ) -> List[ExperimentCondition]:
        """Generate experimental conditions based on research objective."""
        conditions = []
        
        # Baseline condition
        baseline_config = {
            "llm": self.base_llm,
            "max_iterations": 3,
            "reflection_type": ReflectionType.BINARY,
            "success_threshold": 0.8
        }
        
        conditions.append(ExperimentCondition(
            name="baseline",
            description="Standard reflexion configuration",
            config=baseline_config,
            parameters={}
        ))
        
        # Generate hypothesis-specific conditions
        if objective.objective_type == ResearchObjectiveType.PERFORMANCE_OPTIMIZATION:
            # Test different iteration counts
            for iterations in [1, 2, 3, 5, 7]:
                config = baseline_config.copy()
                config["max_iterations"] = iterations
                conditions.append(ExperimentCondition(
                    name=f"iterations_{iterations}",
                    description=f"Reflexion with {iterations} max iterations",
                    config=config,
                    parameters={}
                ))
            
            # Test different thresholds
            for threshold in [0.6, 0.7, 0.8, 0.9]:
                config = baseline_config.copy()
                config["success_threshold"] = threshold
                conditions.append(ExperimentCondition(
                    name=f"threshold_{threshold}",
                    description=f"Success threshold {threshold}",
                    config=config,
                    parameters={}
                ))
        
        elif objective.objective_type == ResearchObjectiveType.ALGORITHM_COMPARISON:
            # Test different reflection types
            for reflection_type in [ReflectionType.BINARY, ReflectionType.SCALAR, ReflectionType.STRUCTURED]:
                config = baseline_config.copy()
                config["reflection_type"] = reflection_type
                conditions.append(ExperimentCondition(
                    name=f"reflection_{reflection_type.value}",
                    description=f"Reflection type: {reflection_type.value}",
                    config=config,
                    parameters={}
                ))
        
        elif objective.objective_type == ResearchObjectiveType.SCALABILITY_STUDY:
            # Test different complexity levels
            complexity_configs = [
                {"max_iterations": 1, "success_threshold": 0.9},
                {"max_iterations": 3, "success_threshold": 0.8},
                {"max_iterations": 5, "success_threshold": 0.7},
                {"max_iterations": 7, "success_threshold": 0.6}
            ]
            
            for i, complexity_config in enumerate(complexity_configs):
                config = baseline_config.copy()
                config.update(complexity_config)
                conditions.append(ExperimentCondition(
                    name=f"complexity_level_{i+1}",
                    description=f"Complexity level {i+1}",
                    config=config,
                    parameters={}
                ))
        
        # Add custom conditions from objective constraints
        if "custom_conditions" in objective.constraints:
            for custom_condition in objective.constraints["custom_conditions"]:
                conditions.append(custom_condition)
        
        return conditions
    
    def _map_objective_to_experiment_type(self, objective_type: ResearchObjectiveType) -> ExperimentType:
        """Map research objective type to experiment type."""
        mapping = {
            ResearchObjectiveType.PERFORMANCE_OPTIMIZATION: ExperimentType.COMPARATIVE,
            ResearchObjectiveType.ALGORITHM_COMPARISON: ExperimentType.COMPARATIVE,
            ResearchObjectiveType.PARAMETER_TUNING: ExperimentType.ABLATION,
            ResearchObjectiveType.FAILURE_ANALYSIS: ExperimentType.HYPOTHESIS_TEST,
            ResearchObjectiveType.SCALABILITY_STUDY: ExperimentType.SCALING,
            ResearchObjectiveType.NOVELTY_DISCOVERY: ExperimentType.HYPOTHESIS_TEST
        }
        return mapping.get(objective_type, ExperimentType.COMPARATIVE)
    
    def _get_research_metrics(self, objective: ResearchObjective) -> List[str]:
        """Get relevant metrics for research objective."""
        base_metrics = ["success", "iterations", "reflections", "total_time"]
        
        if objective.objective_type == ResearchObjectiveType.PERFORMANCE_OPTIMIZATION:
            base_metrics.extend(["avg_confidence", "reflections_per_iteration"])
        elif objective.objective_type == ResearchObjectiveType.SCALABILITY_STUDY:
            base_metrics.extend(["memory_usage", "cpu_utilization"])
        elif objective.objective_type == ResearchObjectiveType.FAILURE_ANALYSIS:
            base_metrics.extend(["error_rate", "recovery_success"])
        
        # Add custom metrics from objective
        if "custom_metrics" in objective.constraints:
            base_metrics.extend(objective.constraints["custom_metrics"])
        
        return base_metrics
    
    async def _analyze_hypothesis(
        self,
        objective: ResearchObjective,
        hypothesis: str,
        experiment_result
    ) -> ResearchFinding:
        """Analyze experimental results to evaluate a hypothesis."""
        
        # Extract relevant data
        evidence = {
            "total_trials": len(experiment_result.trials),
            "conditions_tested": len(experiment_result.config.conditions),
            "summary_statistics": experiment_result.summary_statistics,
            "statistical_tests": experiment_result.statistical_tests
        }
        
        # Determine hypothesis support
        supported, confidence = await self._evaluate_hypothesis_support(
            hypothesis, experiment_result
        )
        
        # Extract statistical significance
        statistical_significance = self._extract_statistical_significance(
            experiment_result.statistical_tests
        )
        
        # Calculate effect size
        effect_size = self._calculate_overall_effect_size(
            experiment_result.statistical_tests
        )
        
        # Assess practical significance
        practical_significance = self._assess_practical_significance(
            effect_size, experiment_result.summary_statistics
        )
        
        # Identify limitations
        limitations = self._identify_study_limitations(objective, experiment_result)
        
        # Suggest future work
        future_work = self._suggest_future_work(hypothesis, experiment_result)
        
        return ResearchFinding(
            objective=objective,
            hypothesis=hypothesis,
            supported=supported,
            confidence=confidence,
            evidence=evidence,
            statistical_significance=statistical_significance,
            effect_size=effect_size,
            practical_significance=practical_significance,
            limitations=limitations,
            future_work=future_work,
            timestamp=datetime.now().isoformat()
        )
    
    async def _evaluate_hypothesis_support(
        self,
        hypothesis: str,
        experiment_result
    ) -> Tuple[bool, float]:
        """Evaluate whether experimental results support a hypothesis."""
        
        # Simple heuristic-based evaluation
        # In production, this would use more sophisticated analysis
        
        support_indicators = 0
        total_indicators = 0
        
        # Check for significant differences
        for comparison, tests in experiment_result.statistical_tests.items():
            total_indicators += 1
            
            # Look for significant p-values
            significant_tests = [
                key for key, value in tests.items()
                if "p_value" in key and value < 0.05
            ]
            
            if significant_tests:
                support_indicators += 1
        
        # Check performance improvements
        success_rates = {}
        for condition, stats in experiment_result.summary_statistics.items():
            success_rates[condition] = stats.get("success_mean", 0.0)
        
        if len(success_rates) > 1:
            total_indicators += 1
            max_success = max(success_rates.values())
            min_success = min(success_rates.values())
            
            # If there's substantial improvement, support hypothesis
            if (max_success - min_success) > 0.1:  # 10% improvement threshold
                support_indicators += 1
        
        # Calculate confidence
        if total_indicators > 0:
            confidence = support_indicators / total_indicators
        else:
            confidence = 0.5  # Neutral confidence when no indicators
        
        # Determine support (threshold: 0.6)
        supported = confidence > 0.6
        
        return supported, confidence
    
    def _extract_statistical_significance(
        self,
        statistical_tests: Dict[str, Dict[str, float]]
    ) -> Optional[float]:
        """Extract overall statistical significance."""
        p_values = []
        
        for comparison, tests in statistical_tests.items():
            for test_name, value in tests.items():
                if "p_value" in test_name:
                    p_values.append(value)
        
        if p_values:
            # Return minimum p-value as overall significance
            return min(p_values)
        
        return None
    
    def _calculate_overall_effect_size(
        self,
        statistical_tests: Dict[str, Dict[str, float]]
    ) -> Optional[float]:
        """Calculate overall effect size across comparisons."""
        effect_sizes = []
        
        for comparison, tests in statistical_tests.items():
            for test_name, value in tests.items():
                if "effect_size" in test_name or "cohens_d" in test_name:
                    effect_sizes.append(abs(value))
        
        if effect_sizes:
            # Return average effect size
            import statistics
            return statistics.mean(effect_sizes)
        
        return None
    
    def _assess_practical_significance(
        self,
        effect_size: Optional[float],
        summary_stats: Dict[str, Dict[str, float]]
    ) -> str:
        """Assess practical significance of findings."""
        if effect_size is None:
            return "unknown"
        
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _identify_study_limitations(
        self,
        objective: ResearchObjective,
        experiment_result
    ) -> List[str]:
        """Identify limitations of the study."""
        limitations = []
        
        # Sample size limitations
        total_trials = len(experiment_result.trials)
        if total_trials < 30:
            limitations.append("Small sample size may limit statistical power")
        
        # Task diversity limitations
        num_tasks = len(experiment_result.config.test_tasks)
        if num_tasks < 5:
            limitations.append("Limited task diversity may affect generalizability")
        
        # Single LLM limitation
        limitations.append("Results limited to single LLM model")
        
        # Simulated environment
        limitations.append("Results from simulated rather than production environment")
        
        # Objective-specific limitations
        if objective.objective_type == ResearchObjectiveType.SCALABILITY_STUDY:
            limitations.append("Scalability tested in controlled conditions only")
        
        return limitations
    
    def _suggest_future_work(
        self,
        hypothesis: str,
        experiment_result
    ) -> List[str]:
        """Suggest future research directions."""
        suggestions = []
        
        # General suggestions
        suggestions.extend([
            "Replicate study with larger sample sizes",
            "Test with different LLM models",
            "Validate in production environment",
            "Investigate interaction effects between parameters"
        ])
        
        # Specific suggestions based on results
        significant_differences = any(
            test_result < 0.05
            for comparison in experiment_result.statistical_tests.values()
            for test_name, test_result in comparison.items()
            if "p_value" in test_name
        )
        
        if not significant_differences:
            suggestions.append("Explore alternative experimental conditions")
            suggestions.append("Consider different success metrics")
        
        if len(experiment_result.config.conditions) < 5:
            suggestions.append("Test additional experimental conditions")
        
        return suggestions
    
    async def _generate_meta_analysis(
        self,
        objective: ResearchObjective,
        findings: List[ResearchFinding]
    ) -> List[ResearchFinding]:
        """Generate meta-analysis findings across hypotheses."""
        if len(findings) < 2:
            return []
        
        # Overall research objective assessment
        supported_hypotheses = [f for f in findings if f.supported]
        support_rate = len(supported_hypotheses) / len(findings)
        
        overall_confidence = sum(f.confidence for f in findings) / len(findings)
        
        # Create meta-finding
        meta_finding = ResearchFinding(
            objective=objective,
            hypothesis=f"Overall research objective: {objective.description}",
            supported=support_rate > 0.5,
            confidence=overall_confidence,
            evidence={
                "hypotheses_tested": len(findings),
                "hypotheses_supported": len(supported_hypotheses),
                "support_rate": support_rate,
                "average_confidence": overall_confidence
            },
            statistical_significance=None,
            effect_size=None,
            practical_significance="meta_analysis",
            limitations=["Meta-analysis based on single study"],
            future_work=["Conduct systematic review of multiple studies"],
            timestamp=datetime.now().isoformat()
        )
        
        return [meta_finding]
    
    async def _generate_research_report(
        self,
        objective: ResearchObjective,
        findings: List[ResearchFinding],
        experiment_result
    ) -> str:
        """Generate comprehensive research report."""
        
        report = {
            "title": f"Research Study: {objective.name}",
            "abstract": {
                "objective": objective.description,
                "methods": f"Experimental study with {len(experiment_result.config.conditions)} conditions",
                "results": f"{len(findings)} hypotheses tested, "
                         f"{sum(1 for f in findings if f.supported)} supported",
                "conclusions": experiment_result.conclusions
            },
            "introduction": {
                "research_objective": asdict(objective),
                "hypotheses": objective.hypotheses
            },
            "methods": {
                "experiment_design": asdict(experiment_result.config),
                "data_collection": f"{len(experiment_result.trials)} trials conducted",
                "analysis_approach": "Comparative statistical analysis"
            },
            "results": {
                "summary_statistics": experiment_result.summary_statistics,
                "statistical_tests": experiment_result.statistical_tests,
                "findings": [asdict(finding) for finding in findings]
            },
            "discussion": {
                "implications": experiment_result.recommendations,
                "limitations": findings[0].limitations if findings else [],
                "future_work": findings[0].future_work if findings else []
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{objective.name}_{timestamp}.json"
        filepath = f"{self.output_dir}/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Research report saved: {filepath}")
        return filepath
    
    def create_performance_optimization_objective(
        self,
        name: str,
        target_metric: str = "success_rate",
        improvement_threshold: float = 0.1
    ) -> ResearchObjective:
        """Create a performance optimization research objective."""
        return ResearchObjective(
            name=name,
            description=f"Optimize reflexion agent performance for {target_metric}",
            objective_type=ResearchObjectiveType.PERFORMANCE_OPTIMIZATION,
            hypotheses=[
                f"Increasing max iterations improves {target_metric}",
                f"Different success thresholds affect {target_metric}",
                f"Reflection type impacts {target_metric}"
            ],
            success_metrics=[target_metric, "efficiency", "consistency"],
            expected_outcomes=[
                f"Identify optimal configuration for {target_metric}",
                "Quantify performance trade-offs",
                "Establish best practices"
            ]
        )
    
    def create_algorithm_comparison_objective(
        self,
        name: str,
        algorithms: List[str]
    ) -> ResearchObjective:
        """Create an algorithm comparison research objective."""
        return ResearchObjective(
            name=name,
            description=f"Compare reflexion algorithms: {', '.join(algorithms)}",
            objective_type=ResearchObjectiveType.ALGORITHM_COMPARISON,
            hypotheses=[
                f"{algo} outperforms other algorithms in specific contexts"
                for algo in algorithms
            ],
            success_metrics=["accuracy", "efficiency", "robustness"],
            expected_outcomes=[
                "Rank algorithms by performance",
                "Identify optimal use cases for each algorithm",
                "Quantify performance differences"
            ]
        )
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of all research activities."""
        return {
            "active_objectives": len(self.active_objectives),
            "completed_studies": len(self.completed_studies),
            "total_findings": len(self.findings_database),
            "supported_findings": sum(1 for f in self.findings_database if f.supported),
            "recent_studies": self.completed_studies[-5:] if self.completed_studies else [],
            "research_areas": list(set(
                f.objective.objective_type.value for f in self.findings_database
            ))
        }