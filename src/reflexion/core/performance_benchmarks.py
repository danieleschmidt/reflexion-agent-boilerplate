"""
Comprehensive performance benchmarking suite for reflexion agents.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .quantum_reflexion_agent import QuantumReflexionAgent
from .types import ReflectionType, ReflexionResult
from ..research.novel_algorithms import research_comparator
from .intelligent_monitoring import intelligent_monitor
from .logging_config import logger


@dataclass
class BenchmarkResult:
    """Individual benchmark test result."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    output_quality_score: float
    iterations_count: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "output_quality_score": self.output_quality_score,
            "iterations_count": self.iterations_count,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    suite_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_execution_time: float
    avg_execution_time: float
    avg_memory_usage: float
    avg_cpu_usage: float
    avg_quality_score: float
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]
    timestamp: datetime
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_tests": self.total_tests,
            "success_rate": self.successful_tests / max(1, self.total_tests),
            "avg_execution_time": self.avg_execution_time,
            "avg_memory_usage": self.avg_memory_usage,
            "avg_cpu_usage": self.avg_cpu_usage,
            "avg_quality_score": self.avg_quality_score,
            "timestamp": self.timestamp.isoformat()
        }


class SystemProfiler:
    """System performance profiling utilities."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        self.baseline_cpu = self.get_cpu_usage()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self, interval: float = 0.1) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=interval)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            "disk_usage": psutil.disk_usage('/').percent,
            "python_version": os.sys.version,
            "platform": os.sys.platform
        }
    
    def monitor_performance(self, duration: float = 1.0, interval: float = 0.1) -> Dict[str, List[float]]:
        """Monitor performance metrics over time."""
        metrics = {
            "memory_usage": [],
            "cpu_usage": [],
            "timestamps": []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics["memory_usage"].append(self.get_memory_usage())
            metrics["cpu_usage"].append(self.get_cpu_usage(interval=0))
            metrics["timestamps"].append(time.time() - start_time)
            time.sleep(interval)
        
        return metrics


class TaskGenerator:
    """Generate benchmark tasks of varying complexity."""
    
    @staticmethod
    def get_simple_tasks() -> List[str]:
        """Get simple benchmark tasks."""
        return [
            "Write a hello world function in Python",
            "Calculate the sum of numbers 1 to 10",
            "Create a simple variable assignment",
            "Write a basic if-else statement",
            "Define a simple data structure"
        ]
    
    @staticmethod
    def get_medium_tasks() -> List[str]:
        """Get medium complexity benchmark tasks."""
        return [
            "Implement a binary search algorithm",
            "Write a function to reverse a linked list",
            "Create a simple sorting algorithm",
            "Implement a stack data structure with push/pop operations",
            "Write a function to find the factorial of a number using recursion"
        ]
    
    @staticmethod
    def get_complex_tasks() -> List[str]:
        """Get complex benchmark tasks."""
        return [
            "Implement a complete binary search tree with insertion, deletion, and traversal",
            "Write a dynamic programming solution for the knapsack problem",
            "Create a graph representation with BFS and DFS traversal algorithms",
            "Implement a complete LRU cache with O(1) operations",
            "Write a thread-safe producer-consumer pattern implementation"
        ]
    
    @staticmethod
    def get_research_tasks() -> List[str]:
        """Get research-oriented benchmark tasks."""
        return [
            "Design and implement a novel machine learning algorithm for classification",
            "Create an innovative approach to distributed system consensus",
            "Develop an optimization algorithm for resource allocation in cloud computing",
            "Implement a new data structure for efficient range queries",
            "Design a fault-tolerant distributed caching system"
        ]


class QualityAssessor:
    """Assess output quality of reflexion results."""
    
    def __init__(self):
        self.quality_metrics = {
            "completeness": self._assess_completeness,
            "correctness": self._assess_correctness,
            "efficiency": self._assess_efficiency,
            "readability": self._assess_readability,
            "best_practices": self._assess_best_practices
        }
    
    def assess_quality(self, task: str, output: str, metadata: Dict[str, Any] = None) -> float:
        """Assess overall quality score (0.0 to 1.0)."""
        
        if not output or len(output.strip()) < 10:
            return 0.0
        
        scores = []
        
        for metric_name, assessor in self.quality_metrics.items():
            try:
                score = assessor(task, output, metadata or {})
                scores.append(score)
            except Exception as e:
                logger.warning(f"Quality assessment {metric_name} failed: {e}")
                scores.append(0.5)  # Default neutral score
        
        return statistics.mean(scores)
    
    def _assess_completeness(self, task: str, output: str, metadata: Dict[str, Any]) -> float:
        """Assess completeness of the output."""
        
        # Basic completeness indicators
        has_function_def = "def " in output
        has_logic = any(keyword in output for keyword in ["if", "for", "while", "try"])
        has_return = "return" in output
        reasonable_length = len(output) > 50
        
        completeness_factors = [has_function_def, has_logic, has_return, reasonable_length]
        
        # Task-specific completeness
        if "algorithm" in task.lower():
            has_implementation = len([line for line in output.split('\n') if line.strip()]) > 5
            completeness_factors.append(has_implementation)
        
        if "test" in task.lower():
            has_assertions = any(keyword in output for keyword in ["assert", "test", "check"])
            completeness_factors.append(has_assertions)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _assess_correctness(self, task: str, output: str, metadata: Dict[str, Any]) -> float:
        """Assess correctness indicators."""
        
        # Syntax correctness indicators
        balanced_parens = output.count('(') == output.count(')')
        balanced_brackets = output.count('[') == output.count(']')
        balanced_braces = output.count('{') == output.count('}')
        proper_indentation = not output.startswith(' ') and '\t' not in output[:10]
        
        syntax_factors = [balanced_parens, balanced_brackets, balanced_braces, proper_indentation]
        
        # Semantic correctness indicators
        no_obvious_errors = "error" not in output.lower()
        has_meaningful_names = len([word for word in output.split() if len(word) > 2]) > 3
        
        semantic_factors = [no_obvious_errors, has_meaningful_names]
        
        all_factors = syntax_factors + semantic_factors
        return sum(all_factors) / len(all_factors)
    
    def _assess_efficiency(self, task: str, output: str, metadata: Dict[str, Any]) -> float:
        """Assess efficiency indicators."""
        
        efficiency_score = 0.5  # Default neutral
        
        # Positive efficiency indicators
        if "O(" in output:  # Time/space complexity mentioned
            efficiency_score += 0.2
        
        if any(pattern in output for pattern in ["memoiz", "cache", "optimize"]):
            efficiency_score += 0.2
        
        # Negative efficiency indicators
        if "nested loop" in output or output.count("for") > 3:
            efficiency_score -= 0.1
        
        if "recursion" in output and "memoiz" not in output:
            efficiency_score -= 0.1
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _assess_readability(self, task: str, output: str, metadata: Dict[str, Any]) -> float:
        """Assess code readability."""
        
        # Readability indicators
        has_comments = "#" in output or '"""' in output
        reasonable_line_length = all(len(line) < 120 for line in output.split('\n'))
        proper_spacing = " = " in output and " + " in output
        descriptive_names = len([word for word in output.split() if len(word) > 4]) > 2
        
        readability_factors = [has_comments, reasonable_line_length, proper_spacing, descriptive_names]
        
        return sum(readability_factors) / len(readability_factors)
    
    def _assess_best_practices(self, task: str, output: str, metadata: Dict[str, Any]) -> float:
        """Assess adherence to best practices."""
        
        practices_score = 0.5  # Default
        
        # Positive practices
        if 'docstring' in output or '"""' in output:
            practices_score += 0.2
        
        if "type hint" in output or "->" in output:
            practices_score += 0.1
        
        if "exception" in output or "try:" in output:
            practices_score += 0.1
        
        # Negative practices
        if "global " in output:
            practices_score -= 0.1
        
        if output.count("print(") > 3:  # Too many print statements
            practices_score -= 0.1
        
        return max(0.0, min(1.0, practices_score))


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for reflexion agents.
    
    Features:
    - Multi-tier task complexity testing
    - Resource usage monitoring
    - Quality assessment
    - Comparative analysis
    - Statistical significance testing
    - Production load simulation
    """
    
    def __init__(self):
        self.system_profiler = SystemProfiler()
        self.task_generator = TaskGenerator()
        self.quality_assessor = QualityAssessor()
        self.benchmark_history: List[BenchmarkSuite] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_benchmark(
        self,
        agent: QuantumReflexionAgent,
        include_simple: bool = True,
        include_medium: bool = True,
        include_complex: bool = True,
        include_research: bool = False,
        iterations_per_task: int = 3,
        concurrent_tasks: int = 1
    ) -> BenchmarkSuite:
        """Run comprehensive benchmark suite."""
        
        self.logger.info("Starting comprehensive benchmark suite")
        
        # Collect all tasks
        all_tasks = []
        
        if include_simple:
            all_tasks.extend([(task, "simple") for task in self.task_generator.get_simple_tasks()])
        
        if include_medium:
            all_tasks.extend([(task, "medium") for task in self.task_generator.get_medium_tasks()])
        
        if include_complex:
            all_tasks.extend([(task, "complex") for task in self.task_generator.get_complex_tasks()])
        
        if include_research:
            all_tasks.extend([(task, "research") for task in self.task_generator.get_research_tasks()])
        
        # Expand tasks with iterations
        expanded_tasks = []
        for task, complexity in all_tasks:
            for iteration in range(iterations_per_task):
                expanded_tasks.append((f"{task} (iter {iteration+1})", complexity, task))
        
        # Run benchmarks
        start_time = time.time()
        results = []
        
        if concurrent_tasks == 1:
            # Sequential execution
            for task_name, complexity, original_task in expanded_tasks:
                result = await self._run_single_benchmark(agent, task_name, complexity, original_task)
                results.append(result)
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(concurrent_tasks)
            
            async def run_with_semaphore(task_name, complexity, original_task):
                async with semaphore:
                    return await self._run_single_benchmark(agent, task_name, complexity, original_task)
            
            tasks = [
                run_with_semaphore(task_name, complexity, original_task)
                for task_name, complexity, original_task in expanded_tasks
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, BenchmarkResult)]
            exception_count = len(results) - len(valid_results)
            
            if exception_count > 0:
                self.logger.warning(f"{exception_count} benchmark tasks failed with exceptions")
            
            results = valid_results
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_results = [r for r in results if r.success]
        successful_count = len(successful_results)
        failed_count = len(results) - successful_count
        
        avg_execution_time = statistics.mean([r.execution_time for r in successful_results]) if successful_results else 0
        avg_memory_usage = statistics.mean([r.memory_usage_mb for r in successful_results]) if successful_results else 0
        avg_cpu_usage = statistics.mean([r.cpu_usage_percent for r in successful_results]) if successful_results else 0
        avg_quality_score = statistics.mean([r.output_quality_score for r in successful_results]) if successful_results else 0
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name="Comprehensive Performance Benchmark",
            total_tests=len(results),
            successful_tests=successful_count,
            failed_tests=failed_count,
            total_execution_time=total_time,
            avg_execution_time=avg_execution_time,
            avg_memory_usage=avg_memory_usage,
            avg_cpu_usage=avg_cpu_usage,
            avg_quality_score=avg_quality_score,
            results=results,
            system_info=self.system_profiler.get_system_info(),
            timestamp=datetime.now()
        )
        
        self.benchmark_history.append(suite)
        
        self.logger.info(f"Benchmark suite completed: {successful_count}/{len(results)} successful")
        return suite
    
    async def _run_single_benchmark(
        self,
        agent: QuantumReflexionAgent,
        task_name: str,
        complexity: str,
        original_task: str
    ) -> BenchmarkResult:
        """Run single benchmark test."""
        
        # Monitor system resources before
        initial_memory = self.system_profiler.get_memory_usage()
        
        # Start performance monitoring
        performance_monitor = threading.Thread(
            target=self._monitor_resources_thread,
            args=(task_name,),
            daemon=True
        )
        
        self.resource_metrics = {"memory": [], "cpu": [], "running": True}
        performance_monitor.start()
        
        try:
            start_time = time.time()
            
            # Execute task
            result = await agent.quantum_run(
                task=original_task,
                algorithm_ensemble=complexity in ["complex", "research"]
            )
            
            execution_time = time.time() - start_time
            
            # Stop monitoring
            self.resource_metrics["running"] = False
            performance_monitor.join(timeout=1.0)
            
            # Calculate resource usage
            final_memory = self.system_profiler.get_memory_usage()
            memory_delta = max(0, final_memory - initial_memory)
            
            avg_cpu = statistics.mean(self.resource_metrics["cpu"]) if self.resource_metrics["cpu"] else 0
            peak_memory = max(self.resource_metrics["memory"]) if self.resource_metrics["memory"] else final_memory
            
            # Assess quality
            quality_score = self.quality_assessor.assess_quality(
                task=original_task,
                output=result.output,
                metadata=result.metadata
            )
            
            # Record metrics
            intelligent_monitor.record_metric(f"benchmark_execution_time_{complexity}", execution_time)
            intelligent_monitor.record_metric(f"benchmark_quality_score_{complexity}", quality_score)
            intelligent_monitor.record_metric(f"benchmark_memory_usage_{complexity}", memory_delta)
            
            return BenchmarkResult(
                test_name=task_name,
                execution_time=execution_time,
                memory_usage_mb=memory_delta,
                cpu_usage_percent=avg_cpu,
                success=result.success,
                output_quality_score=quality_score,
                iterations_count=result.iterations,
                metadata={
                    "complexity": complexity,
                    "output_length": len(result.output),
                    "peak_memory_mb": peak_memory,
                    "quantum_enhanced": result.metadata.get("quantum_enhanced", False)
                }
            )
            
        except Exception as e:
            # Stop monitoring
            self.resource_metrics["running"] = False
            if performance_monitor.is_alive():
                performance_monitor.join(timeout=1.0)
            
            execution_time = time.time() - start_time
            
            self.logger.error(f"Benchmark task {task_name} failed: {e}")
            
            return BenchmarkResult(
                test_name=task_name,
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                success=False,
                output_quality_score=0.0,
                iterations_count=0,
                error_message=str(e),
                metadata={"complexity": complexity}
            )
    
    def _monitor_resources_thread(self, task_name: str):
        """Monitor system resources in background thread."""
        
        while self.resource_metrics.get("running", False):
            try:
                cpu_usage = self.system_profiler.get_cpu_usage(interval=0)
                memory_usage = self.system_profiler.get_memory_usage()
                
                self.resource_metrics["cpu"].append(cpu_usage)
                self.resource_metrics["memory"].append(memory_usage)
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break
    
    async def run_stress_test(
        self,
        agent: QuantumReflexionAgent,
        concurrent_tasks: int = 5,
        duration_minutes: int = 10,
        task_complexity: str = "medium"
    ) -> Dict[str, Any]:
        """Run stress test with high concurrent load."""
        
        self.logger.info(f"Starting stress test: {concurrent_tasks} concurrent tasks for {duration_minutes} minutes")
        
        # Get tasks for stress test
        if task_complexity == "simple":
            tasks = self.task_generator.get_simple_tasks()
        elif task_complexity == "complex":
            tasks = self.task_generator.get_complex_tasks()
        else:
            tasks = self.task_generator.get_medium_tasks()
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        completed_tasks = []
        failed_tasks = []
        
        async def stress_worker(worker_id: int):
            """Individual stress test worker."""
            worker_results = []
            
            while time.time() < end_time:
                try:
                    task = tasks[len(worker_results) % len(tasks)]
                    
                    task_start = time.time()
                    result = await agent.quantum_run(task, algorithm_ensemble=False)
                    task_time = time.time() - task_start
                    
                    worker_results.append({
                        "worker_id": worker_id,
                        "task": task,
                        "execution_time": task_time,
                        "success": result.success,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Brief pause between tasks
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    failed_tasks.append({
                        "worker_id": worker_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return worker_results
        
        # Run stress test workers
        workers = [stress_worker(i) for i in range(concurrent_tasks)]
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        
        # Collect results
        for worker_result in worker_results:
            if isinstance(worker_result, Exception):
                failed_tasks.append({"error": str(worker_result)})
            else:
                completed_tasks.extend(worker_result)
        
        total_runtime = time.time() - start_time
        
        # Calculate stress test statistics
        total_completed = len(completed_tasks)
        total_failed = len(failed_tasks)
        
        if completed_tasks:
            avg_task_time = statistics.mean([t["execution_time"] for t in completed_tasks])
            throughput = total_completed / total_runtime
            success_rate = total_completed / (total_completed + total_failed)
        else:
            avg_task_time = 0
            throughput = 0
            success_rate = 0
        
        return {
            "stress_test_summary": {
                "duration_minutes": duration_minutes,
                "concurrent_workers": concurrent_tasks,
                "total_completed_tasks": total_completed,
                "total_failed_tasks": total_failed,
                "success_rate": success_rate,
                "avg_task_execution_time": avg_task_time,
                "throughput_tasks_per_second": throughput,
                "total_runtime_seconds": total_runtime
            },
            "completed_tasks": completed_tasks[-50:],  # Last 50 for brevity
            "failed_tasks": failed_tasks,
            "system_info": self.system_profiler.get_system_info()
        }
    
    def compare_benchmark_suites(self, suite1: BenchmarkSuite, suite2: BenchmarkSuite) -> Dict[str, Any]:
        """Compare two benchmark suites for performance regression analysis."""
        
        comparison = {
            "suite1_name": suite1.suite_name,
            "suite2_name": suite2.suite_name,
            "performance_changes": {},
            "statistical_analysis": {},
            "recommendations": []
        }
        
        # Performance metric comparisons
        metrics = [
            ("avg_execution_time", "seconds"),
            ("avg_memory_usage", "MB"),
            ("avg_cpu_usage", "percent"),
            ("avg_quality_score", "score")
        ]
        
        for metric, unit in metrics:
            value1 = getattr(suite1, metric)
            value2 = getattr(suite2, metric)
            
            change_percent = ((value2 - value1) / value1 * 100) if value1 != 0 else 0
            
            comparison["performance_changes"][metric] = {
                "suite1_value": value1,
                "suite2_value": value2,
                "change_percent": change_percent,
                "improvement": change_percent < 0 if "time" in metric else change_percent > 0,
                "unit": unit
            }
        
        # Statistical significance testing (simplified)
        suite1_times = [r.execution_time for r in suite1.results if r.success]
        suite2_times = [r.execution_time for r in suite2.results if r.success]
        
        if len(suite1_times) > 5 and len(suite2_times) > 5:
            # T-test approximation
            mean1, mean2 = statistics.mean(suite1_times), statistics.mean(suite2_times)
            var1 = statistics.variance(suite1_times) if len(suite1_times) > 1 else 0
            var2 = statistics.variance(suite2_times) if len(suite2_times) > 1 else 0
            
            pooled_var = ((len(suite1_times) - 1) * var1 + (len(suite2_times) - 1) * var2) / \
                        (len(suite1_times) + len(suite2_times) - 2)
            
            if pooled_var > 0:
                t_stat = (mean1 - mean2) / (pooled_var * (1/len(suite1_times) + 1/len(suite2_times))) ** 0.5
                comparison["statistical_analysis"]["t_statistic"] = t_stat
                comparison["statistical_analysis"]["significant"] = abs(t_stat) > 2.0  # Simplified threshold
        
        # Generate recommendations
        exec_time_change = comparison["performance_changes"]["avg_execution_time"]["change_percent"]
        quality_change = comparison["performance_changes"]["avg_quality_score"]["change_percent"]
        
        if exec_time_change > 20:
            comparison["recommendations"].append("Performance regression detected - investigate execution time increase")
        
        if quality_change < -10:
            comparison["recommendations"].append("Quality regression detected - review output quality")
        
        if exec_time_change < -20:
            comparison["recommendations"].append("Significant performance improvement achieved")
        
        return comparison
    
    def generate_performance_report(self, suite: BenchmarkSuite) -> str:
        """Generate comprehensive performance report."""
        
        report_lines = [
            f"# Performance Benchmark Report",
            f"**Suite**: {suite.suite_name}",
            f"**Timestamp**: {suite.timestamp.isoformat()}",
            f"**Total Tests**: {suite.total_tests}",
            f"**Success Rate**: {suite.successful_tests/suite.total_tests:.1%}",
            f"",
            f"## Summary Statistics",
            f"- Average Execution Time: {suite.avg_execution_time:.2f}s",
            f"- Average Memory Usage: {suite.avg_memory_usage:.1f} MB",
            f"- Average CPU Usage: {suite.avg_cpu_usage:.1f}%",
            f"- Average Quality Score: {suite.avg_quality_score:.2f}/1.0",
            f"",
            f"## System Information",
            f"- CPU Cores: {suite.system_info.get('cpu_count', 'N/A')}",
            f"- Memory Total: {suite.system_info.get('memory_total', 0):.1f} GB",
            f"- Platform: {suite.system_info.get('platform', 'Unknown')}",
            f""
        ]
        
        # Performance breakdown by complexity
        complexity_stats = defaultdict(list)
        for result in suite.results:
            if result.success:
                complexity = result.metadata.get('complexity', 'unknown')
                complexity_stats[complexity].append(result)
        
        if complexity_stats:
            report_lines.append("## Performance by Complexity")
            
            for complexity, results in complexity_stats.items():
                avg_time = statistics.mean([r.execution_time for r in results])
                avg_quality = statistics.mean([r.output_quality_score for r in results])
                
                report_lines.extend([
                    f"### {complexity.title()} Tasks",
                    f"- Tests: {len(results)}",
                    f"- Avg Time: {avg_time:.2f}s",
                    f"- Avg Quality: {avg_quality:.2f}",
                    f""
                ])
        
        # Top performers and failures
        successful_results = [r for r in suite.results if r.success]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x.execution_time)
            highest_quality = max(successful_results, key=lambda x: x.output_quality_score)
            
            report_lines.extend([
                f"## Notable Results",
                f"**Fastest Task**: {fastest.test_name} ({fastest.execution_time:.2f}s)",
                f"**Highest Quality**: {highest_quality.test_name} ({highest_quality.output_quality_score:.2f})",
                f""
            ])
        
        # Failed tests
        failed_results = [r for r in suite.results if not r.success]
        if failed_results:
            report_lines.extend([
                f"## Failed Tests ({len(failed_results)})",
            ])
            
            for failure in failed_results[:5]:  # Show up to 5 failures
                report_lines.append(f"- {failure.test_name}: {failure.error_message}")
            
            if len(failed_results) > 5:
                report_lines.append(f"- ... and {len(failed_results) - 5} more failures")
        
        return "\n".join(report_lines)


# Global benchmark suite instance
performance_benchmarks = PerformanceBenchmarkSuite()