"""Experimental framework for hypothesis-driven reflexion research."""

import asyncio
import json
import time
import statistics
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import logging

from ..core.types import ReflexionResult, ReflectionType
from ..core.agent import ReflexionAgent


class ExperimentType(Enum):
    """Types of experiments for reflexion research."""
    COMPARATIVE = "comparative"  # Compare different approaches
    ABLATION = "ablation"       # Remove/modify components
    SCALING = "scaling"         # Test at different scales
    LONGITUDINAL = "longitudinal"  # Track changes over time
    HYPOTHESIS_TEST = "hypothesis_test"  # Test specific hypotheses


@dataclass
class ExperimentCondition:
    """Defines a single experimental condition."""
    name: str
    description: str
    config: Dict[str, Any]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExperimentConfig:
    """Configuration for an experimental study."""
    name: str
    description: str
    experiment_type: ExperimentType
    conditions: List[ExperimentCondition]
    test_tasks: List[str]
    success_criteria: List[str]
    metrics: List[str]
    num_trials: int = 10
    randomization_seed: int = 42
    timeout_seconds: float = 300.0
    statistical_tests: List[str] = None

    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ["t_test", "wilcoxon", "effect_size"]


@dataclass  
class ExperimentTrial:
    """Results from a single experimental trial."""
    condition_name: str
    task: str
    trial_number: int
    result: ReflexionResult
    metrics: Dict[str, float]
    timestamp: str
    duration_seconds: float


@dataclass
class ExperimentResult:
    """Complete results from an experimental study."""
    config: ExperimentConfig
    trials: List[ExperimentTrial]
    summary_statistics: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, float]]
    conclusions: List[str]
    recommendations: List[str]
    raw_data_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": asdict(self.config),
            "trials": [asdict(trial) for trial in self.trials],
            "summary_statistics": self.summary_statistics,
            "statistical_tests": self.statistical_tests,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations,
            "raw_data_path": self.raw_data_path,
            "generated_at": datetime.now().isoformat()
        }


class ExperimentRunner:
    """Runs controlled experiments with reflexion agents."""
    
    def __init__(self, output_dir: str = "./experiments"):
        """Initialize experiment runner.
        
        Args:
            output_dir: Directory to save experimental results
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete experimental study."""
        self.logger.info(f"Starting experiment: {config.name}")
        start_time = time.time()
        
        # Initialize trial storage
        trials = []
        
        # Randomize trial order if needed
        import random
        random.seed(config.randomization_seed)
        
        # Generate all trial combinations
        trial_combinations = []
        for condition in config.conditions:
            for task in config.test_tasks:
                for trial_num in range(config.num_trials):
                    trial_combinations.append((condition, task, trial_num))
        
        # Randomize order
        random.shuffle(trial_combinations)
        
        # Run trials
        total_trials = len(trial_combinations)
        completed_trials = 0
        
        for condition, task, trial_num in trial_combinations:
            try:
                trial = await self._run_single_trial(
                    condition, task, trial_num, config.timeout_seconds
                )
                trials.append(trial)
                completed_trials += 1
                
                if completed_trials % 10 == 0:
                    self.logger.info(
                        f"Completed {completed_trials}/{total_trials} trials"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Trial failed - Condition: {condition.name}, "
                    f"Task: {task}, Trial: {trial_num}, Error: {e}"
                )
                continue
        
        # Analyze results
        summary_stats = self._calculate_summary_statistics(trials, config.metrics)
        statistical_tests = self._run_statistical_tests(trials, config)
        conclusions = self._generate_conclusions(summary_stats, statistical_tests)
        recommendations = self._generate_recommendations(conclusions, config)
        
        # Save raw data
        raw_data_path = await self._save_raw_data(trials, config)
        
        experiment_result = ExperimentResult(
            config=config,
            trials=trials,
            summary_statistics=summary_stats,
            statistical_tests=statistical_tests,
            conclusions=conclusions,
            recommendations=recommendations,
            raw_data_path=raw_data_path
        )
        
        # Save experiment results
        await self._save_experiment_results(experiment_result)
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Experiment {config.name} completed in {total_time:.2f}s "
            f"with {len(trials)} successful trials"
        )
        
        return experiment_result
    
    async def _run_single_trial(
        self, 
        condition: ExperimentCondition,
        task: str,
        trial_number: int,
        timeout: float
    ) -> ExperimentTrial:
        """Run a single experimental trial."""
        trial_start = time.time()
        
        # Create agent with condition parameters
        agent = ReflexionAgent(**condition.config)
        
        # Execute task with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_task_async(agent, task, condition.parameters),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Create timeout result
            result = ReflexionResult(
                task=task,
                output="TIMEOUT: Task execution exceeded time limit",
                success=False,
                iterations=0,
                reflections=[],
                total_time=timeout,
                metadata={"timeout": True, "condition": condition.name}
            )
        
        # Calculate metrics
        metrics = self._calculate_trial_metrics(result, condition)
        
        trial_duration = time.time() - trial_start
        
        return ExperimentTrial(
            condition_name=condition.name,
            task=task,
            trial_number=trial_number,
            result=result,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            duration_seconds=trial_duration
        )
    
    async def _execute_task_async(
        self, 
        agent: ReflexionAgent,
        task: str, 
        parameters: Dict[str, Any]
    ) -> ReflexionResult:
        """Execute agent task asynchronously."""
        # Use the agent's execution method
        return agent.run(task, **parameters)
    
    def _calculate_trial_metrics(
        self, 
        result: ReflexionResult,
        condition: ExperimentCondition
    ) -> Dict[str, float]:
        """Calculate metrics for a single trial."""
        metrics = {
            "success": 1.0 if result.success else 0.0,
            "iterations": float(result.iterations),
            "reflections": float(len(result.reflections)),
            "total_time": result.total_time,
            "reflections_per_iteration": (
                len(result.reflections) / max(1, result.iterations)
            ),
        }
        
        # Calculate reflection quality metrics
        if result.reflections:
            confidences = [r.confidence for r in result.reflections]
            metrics.update({
                "avg_confidence": statistics.mean(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            })
        else:
            metrics.update({
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "confidence_std": 0.0
            })
        
        # Add condition-specific metrics if available
        if "custom_metrics" in condition.metadata:
            for metric_name, metric_func in condition.metadata["custom_metrics"].items():
                try:
                    metrics[metric_name] = metric_func(result)
                except Exception as e:
                    self.logger.warning(f"Custom metric {metric_name} failed: {e}")
                    metrics[metric_name] = 0.0
        
        return metrics
    
    def _calculate_summary_statistics(
        self, 
        trials: List[ExperimentTrial],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for all conditions."""
        # Group trials by condition
        condition_data = {}
        for trial in trials:
            if trial.condition_name not in condition_data:
                condition_data[trial.condition_name] = []
            condition_data[trial.condition_name].append(trial)
        
        summary = {}
        
        for condition_name, condition_trials in condition_data.items():
            condition_stats = {}
            
            for metric in metrics:
                values = []
                for trial in condition_trials:
                    if metric in trial.metrics:
                        values.append(trial.metrics[metric])
                
                if values:
                    condition_stats[f"{metric}_mean"] = statistics.mean(values)
                    condition_stats[f"{metric}_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                    condition_stats[f"{metric}_min"] = min(values)
                    condition_stats[f"{metric}_max"] = max(values)
                    condition_stats[f"{metric}_median"] = statistics.median(values)
                    condition_stats[f"{metric}_count"] = len(values)
                else:
                    for stat in ["mean", "std", "min", "max", "median"]:
                        condition_stats[f"{metric}_{stat}"] = 0.0
                    condition_stats[f"{metric}_count"] = 0
            
            summary[condition_name] = condition_stats
        
        return summary
    
    def _run_statistical_tests(
        self,
        trials: List[ExperimentTrial],
        config: ExperimentConfig
    ) -> Dict[str, Dict[str, float]]:
        """Run statistical tests comparing conditions."""
        if len(config.conditions) < 2:
            return {}
        
        # Group data by condition
        condition_data = {}
        for trial in trials:
            if trial.condition_name not in condition_data:
                condition_data[trial.condition_name] = {}
            
            for metric, value in trial.metrics.items():
                if metric not in condition_data[trial.condition_name]:
                    condition_data[trial.condition_name][metric] = []
                condition_data[trial.condition_name][metric].append(value)
        
        statistical_results = {}
        
        # Compare all pairs of conditions
        condition_names = list(condition_data.keys())
        for i in range(len(condition_names)):
            for j in range(i + 1, len(condition_names)):
                cond1, cond2 = condition_names[i], condition_names[j]
                comparison_key = f"{cond1}_vs_{cond2}"
                
                statistical_results[comparison_key] = {}
                
                for metric in config.metrics:
                    if (metric in condition_data[cond1] and 
                        metric in condition_data[cond2]):
                        
                        data1 = condition_data[cond1][metric]
                        data2 = condition_data[cond2][metric]
                        
                        # Run statistical tests
                        test_results = self._perform_statistical_tests(data1, data2)
                        
                        for test_name, result in test_results.items():
                            statistical_results[comparison_key][f"{metric}_{test_name}"] = result
        
        return statistical_results
    
    def _perform_statistical_tests(
        self, 
        data1: List[float], 
        data2: List[float]
    ) -> Dict[str, float]:
        """Perform statistical tests between two datasets."""
        results = {}
        
        if len(data1) < 2 or len(data2) < 2:
            return {"insufficient_data": 1.0}
        
        try:
            # T-test (assuming scipy is available in production)
            try:
                from scipy import stats
                
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                results["t_test_statistic"] = float(t_stat)
                results["t_test_p_value"] = float(p_value)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                results["mannwhitney_statistic"] = float(u_stat)
                results["mannwhitney_p_value"] = float(u_p_value)
                
                # Effect size (Cohen's d)
                pooled_std = ((statistics.stdev(data1) ** 2 + statistics.stdev(data2) ** 2) / 2) ** 0.5
                if pooled_std > 0:
                    cohens_d = (statistics.mean(data1) - statistics.mean(data2)) / pooled_std
                    results["cohens_d"] = float(cohens_d)
                else:
                    results["cohens_d"] = 0.0
                    
            except ImportError:
                # Fallback to basic statistical comparison
                mean1, mean2 = statistics.mean(data1), statistics.mean(data2)
                std1, std2 = statistics.stdev(data1), statistics.stdev(data2)
                
                # Basic effect size
                pooled_std = ((std1 ** 2 + std2 ** 2) / 2) ** 0.5
                if pooled_std > 0:
                    effect_size = (mean1 - mean2) / pooled_std
                else:
                    effect_size = 0.0
                
                results["mean_difference"] = mean1 - mean2
                results["effect_size"] = effect_size
                results["practical_significance"] = 1.0 if abs(effect_size) > 0.5 else 0.0
        
        except Exception as e:
            self.logger.warning(f"Statistical test failed: {e}")
            results["error"] = 1.0
        
        return results
    
    def _generate_conclusions(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        statistical_tests: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate research conclusions from experimental results."""
        conclusions = []
        
        # Overall performance comparison
        if len(summary_stats) > 1:
            success_rates = {
                cond: stats.get("success_mean", 0.0) 
                for cond, stats in summary_stats.items()
            }
            
            best_condition = max(success_rates.items(), key=lambda x: x[1])
            conclusions.append(
                f"Best performing condition: {best_condition[0]} "
                f"(success rate: {best_condition[1]:.2%})"
            )
            
            # Statistical significance
            significant_differences = []
            for comparison, tests in statistical_tests.items():
                p_value = tests.get("success_t_test_p_value", 1.0)
                if p_value < 0.05:
                    effect_size = tests.get("success_cohens_d", 0.0)
                    significant_differences.append(
                        f"{comparison}: p={p_value:.4f}, effect size={effect_size:.3f}"
                    )
            
            if significant_differences:
                conclusions.append(
                    f"Statistically significant differences found: {significant_differences}"
                )
            else:
                conclusions.append("No statistically significant differences detected")
        
        # Reflection patterns
        reflection_efficiency = {}
        for cond, stats in summary_stats.items():
            avg_reflections = stats.get("reflections_mean", 0.0)
            success_rate = stats.get("success_mean", 0.0)
            if avg_reflections > 0:
                efficiency = success_rate / avg_reflections
                reflection_efficiency[cond] = efficiency
        
        if reflection_efficiency:
            most_efficient = max(reflection_efficiency.items(), key=lambda x: x[1])
            conclusions.append(
                f"Most reflection-efficient condition: {most_efficient[0]} "
                f"(success per reflection: {most_efficient[1]:.3f})"
            )
        
        return conclusions
    
    def _generate_recommendations(
        self,
        conclusions: List[str],
        config: ExperimentConfig
    ) -> List[str]:
        """Generate actionable recommendations from conclusions."""
        recommendations = []
        
        # Based on experiment type
        if config.experiment_type == ExperimentType.COMPARATIVE:
            recommendations.append(
                "Use best-performing condition as baseline for future experiments"
            )
            recommendations.append(
                "Investigate why top conditions outperformed others"
            )
        
        elif config.experiment_type == ExperimentType.ABLATION:
            recommendations.append(
                "Remove components that don't improve performance"
            )
            recommendations.append(
                "Focus resources on most impactful components"
            )
        
        elif config.experiment_type == ExperimentType.SCALING:
            recommendations.append(
                "Identify optimal scale parameters for deployment"
            )
            recommendations.append(
                "Monitor performance degradation at higher scales"
            )
        
        # General recommendations
        recommendations.extend([
            "Replicate findings with larger sample sizes",
            "Test with different task types for generalizability",
            "Consider cost-benefit trade-offs in production deployment"
        ])
        
        return recommendations
    
    async def _save_raw_data(
        self,
        trials: List[ExperimentTrial],
        config: ExperimentConfig
    ) -> str:
        """Save raw trial data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.name}_{timestamp}_raw_data.json"
        filepath = f"{self.output_dir}/{filename}"
        
        # Convert trials to serializable format
        trial_data = []
        for trial in trials:
            trial_dict = asdict(trial)
            # Convert ReflexionResult to dict
            trial_dict["result"] = asdict(trial.result)
            trial_data.append(trial_dict)
        
        with open(filepath, 'w') as f:
            json.dump(trial_data, f, indent=2, default=str)
        
        return filepath
    
    async def _save_experiment_results(self, result: ExperimentResult) -> str:
        """Save complete experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}_results.json"
        filepath = f"{self.output_dir}/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Experiment results saved to: {filepath}")
        return filepath
    
    def create_comparative_experiment(
        self,
        name: str,
        baseline_config: Dict[str, Any],
        experimental_configs: List[Dict[str, Any]],
        test_tasks: List[str],
        num_trials: int = 20
    ) -> ExperimentConfig:
        """Create a comparative experiment configuration."""
        conditions = []
        
        # Add baseline condition
        conditions.append(ExperimentCondition(
            name="baseline",
            description="Baseline configuration",
            config=baseline_config,
            parameters={}
        ))
        
        # Add experimental conditions
        for i, config in enumerate(experimental_configs):
            conditions.append(ExperimentCondition(
                name=f"experimental_{i+1}",
                description=f"Experimental configuration {i+1}",
                config=config,
                parameters={}
            ))
        
        return ExperimentConfig(
            name=name,
            description=f"Comparative study with {len(conditions)} conditions",
            experiment_type=ExperimentType.COMPARATIVE,
            conditions=conditions,
            test_tasks=test_tasks,
            success_criteria=["success", "efficiency"],
            metrics=["success", "iterations", "reflections", "total_time"],
            num_trials=num_trials
        )