#!/usr/bin/env python3
"""
Advanced Scalability Benchmarks for ReflexionAgent

This module provides comprehensive benchmarks to test the scalability
and performance characteristics of the reflexion framework under various loads.
"""

import asyncio
import json
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion import (
    ReflexionAgent,
    OptimizedReflexionAgent,
    AutoScalingReflexionAgent,
    ReflectionType
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    test_name: str
    tasks_completed: int
    total_time: float
    avg_time_per_task: float
    throughput_tps: float
    success_rate: float
    cache_hit_rate: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_utilization: Optional[float] = None
    error_count: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceMonitor:
    """Simple resource monitoring utility."""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return average metrics."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if not self.samples:
            return {"avg_cpu": 0.0, "avg_memory": 0.0}
        
        avg_cpu = sum(s["cpu"] for s in self.samples) / len(self.samples)
        avg_memory = sum(s["memory"] for s in self.samples) / len(self.samples)
        
        return {"avg_cpu": avg_cpu, "avg_memory": avg_memory}
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        import psutil
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.samples.append({
                    "cpu": cpu_percent,
                    "memory": memory_mb,
                    "timestamp": time.time()
                })
                
                time.sleep(0.5)  # Sample every 500ms
            except:
                pass  # Ignore monitoring errors


class ScalabilityBenchmark:
    """Comprehensive scalability benchmark suite."""
    
    def __init__(self):
        self.results = []
        self.resource_monitor = ResourceMonitor()
    
    @contextmanager
    def benchmark_context(self, test_name: str):
        """Context manager for benchmark execution."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {test_name}")
        print('='*60)
        
        start_time = time.time()
        self.resource_monitor.start_monitoring()
        
        try:
            yield
        finally:
            end_time = time.time()
            resource_stats = self.resource_monitor.stop_monitoring()
            print(f"Completed {test_name} in {end_time - start_time:.2f}s")
            print(f"Resource usage - CPU: {resource_stats['avg_cpu']:.1f}%, Memory: {resource_stats['avg_memory']:.1f}MB")
    
    def benchmark_basic_throughput(self, agent_class=ReflexionAgent):
        """Benchmark basic throughput with increasing load."""
        with self.benchmark_context("Basic Throughput"):
            agent = agent_class(llm="gpt-4", max_iterations=2)
            
            task_counts = [10, 25, 50, 100]
            tasks = [
                "Write a function to sort a list",
                "Implement binary search algorithm", 
                "Create a hash table data structure",
                "Design a caching mechanism",
                "Build a priority queue"
            ]
            
            for count in task_counts:
                print(f"\nTesting with {count} tasks...")
                start_time = time.time()
                successes = 0
                
                # Execute tasks sequentially 
                for i in range(count):
                    task = tasks[i % len(tasks)]
                    result = agent.run(task)
                    if result.success:
                        successes += 1
                
                total_time = time.time() - start_time
                throughput = count / total_time
                success_rate = successes / count
                
                result = BenchmarkResult(
                    test_name=f"Basic-{count}tasks",
                    tasks_completed=count,
                    total_time=total_time,
                    avg_time_per_task=total_time / count,
                    throughput_tps=throughput,
                    success_rate=success_rate
                )
                
                self.results.append(result)
                print(f"  Results: {throughput:.2f} tasks/sec, {success_rate:.1%} success rate")
    
    def benchmark_optimized_performance(self):
        """Benchmark optimized agent performance."""
        with self.benchmark_context("Optimized Performance"):
            # Test different optimization configurations
            configs = [
                {"enable_caching": False, "enable_parallel_execution": False, "name": "Baseline"},
                {"enable_caching": True, "enable_parallel_execution": False, "name": "With-Caching"},
                {"enable_caching": False, "enable_parallel_execution": True, "name": "With-Parallel"},
                {"enable_caching": True, "enable_parallel_execution": True, "name": "Full-Optimized"}
            ]
            
            task_count = 30
            tasks = [
                "Implement a sorting algorithm",
                "Create a data validation function", 
                "Design a logging system",
                "Build a configuration parser",
                "Develop a task scheduler"
            ]
            
            for config in configs:
                print(f"\nTesting {config['name']} configuration...")
                
                agent = OptimizedReflexionAgent(
                    llm="gpt-4",
                    max_iterations=2,
                    enable_caching=config["enable_caching"],
                    enable_parallel_execution=config["enable_parallel_execution"],
                    max_concurrent_tasks=4
                )
                
                start_time = time.time()
                successes = 0
                
                for i in range(task_count):
                    task = tasks[i % len(tasks)]
                    result = agent.run(task)
                    if result.success:
                        successes += 1
                
                total_time = time.time() - start_time
                performance_stats = agent.get_performance_stats()
                
                result = BenchmarkResult(
                    test_name=f"Optimized-{config['name']}",
                    tasks_completed=task_count,
                    total_time=total_time,
                    avg_time_per_task=total_time / task_count,
                    throughput_tps=task_count / total_time,
                    success_rate=successes / task_count,
                    cache_hit_rate=performance_stats["derived_metrics"]["cache_hit_rate"],
                    metadata=performance_stats
                )
                
                self.results.append(result)
                print(f"  Results: {result.throughput_tps:.2f} tasks/sec, "
                      f"Cache hit rate: {result.cache_hit_rate:.1%}")
    
    async def benchmark_batch_processing(self):
        """Benchmark batch processing capabilities."""
        with self.benchmark_context("Batch Processing"):
            agent = OptimizedReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_parallel_execution=True,
                max_concurrent_tasks=8
            )
            
            batch_sizes = [5, 10, 20, 50]
            base_tasks = [
                "Create a file parser utility",
                "Implement error handling logic",
                "Design a REST API endpoint", 
                "Build a data transformer",
                "Develop a unit test suite"
            ]
            
            for batch_size in batch_sizes:
                print(f"\nTesting batch size: {batch_size}")
                tasks = [base_tasks[i % len(base_tasks)] for i in range(batch_size)]
                
                start_time = time.time()
                results = await agent.run_batch(tasks)
                total_time = time.time() - start_time
                
                successes = sum(1 for r in results if r.success)
                
                result = BenchmarkResult(
                    test_name=f"Batch-{batch_size}",
                    tasks_completed=batch_size,
                    total_time=total_time,
                    avg_time_per_task=total_time / batch_size,
                    throughput_tps=batch_size / total_time,
                    success_rate=successes / batch_size
                )
                
                self.results.append(result)
                print(f"  Results: {result.throughput_tps:.2f} tasks/sec")
    
    async def benchmark_autoscaling(self):
        """Benchmark auto-scaling agent under varying load."""
        with self.benchmark_context("Auto-Scaling"):
            agent = AutoScalingReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_parallel_execution=True
            )
            
            # Simulate load patterns
            load_patterns = [
                {"name": "Constant-Low", "tasks": [5, 5, 5, 5]},
                {"name": "Ramp-Up", "tasks": [2, 5, 10, 15]},  
                {"name": "Spike", "tasks": [3, 3, 20, 3]},
                {"name": "High-Sustained", "tasks": [15, 15, 15, 15]}
            ]
            
            base_task = "Implement a utility function"
            
            for pattern in load_patterns:
                print(f"\nTesting {pattern['name']} load pattern...")
                total_tasks = 0
                total_time = 0
                total_successes = 0
                
                pattern_start = time.time()
                
                for wave, task_count in enumerate(pattern["tasks"]):
                    print(f"  Wave {wave + 1}: {task_count} tasks")
                    wave_start = time.time()
                    
                    # Execute wave of tasks
                    tasks = []
                    for i in range(task_count):
                        try:
                            result = await agent.run_with_autoscaling(
                                f"{base_task} #{total_tasks + i}",
                                success_criteria="basic functionality"
                            )
                            tasks.append(result)
                            if result.success:
                                total_successes += 1
                        except Exception as e:
                            print(f"    Task failed: {str(e)}")
                    
                    wave_time = time.time() - wave_start
                    total_tasks += task_count
                    total_time += wave_time
                    
                    print(f"    Wave completed in {wave_time:.2f}s")
                    
                    # Brief pause between waves
                    await asyncio.sleep(0.5)
                
                pattern_time = time.time() - pattern_start
                scaling_stats = agent.get_scaling_stats()
                
                result = BenchmarkResult(
                    test_name=f"AutoScale-{pattern['name']}",
                    tasks_completed=total_tasks,
                    total_time=pattern_time,
                    avg_time_per_task=pattern_time / total_tasks if total_tasks > 0 else 0,
                    throughput_tps=total_tasks / pattern_time if pattern_time > 0 else 0,
                    success_rate=total_successes / total_tasks if total_tasks > 0 else 0,
                    metadata=scaling_stats
                )
                
                self.results.append(result)
                print(f"  Pattern Results: {result.throughput_tps:.2f} tasks/sec, "
                      f"Workers: {scaling_stats['current_workers']}")
    
    def benchmark_memory_efficiency(self):
        """Benchmark memory usage patterns."""
        with self.benchmark_context("Memory Efficiency"):
            # Test with and without memory management
            configs = [
                {"memory_limit": None, "name": "No-Limit"},
                {"memory_limit": 100, "name": "Limited-Memory"}
            ]
            
            task_count = 50
            base_task = "Process data and generate report"
            
            for config in configs:
                print(f"\nTesting {config['name']} configuration...")
                
                agent = OptimizedReflexionAgent(
                    llm="gpt-4", 
                    max_iterations=2,
                    enable_caching=True,
                    cache_size=config["memory_limit"] or 1000
                )
                
                start_time = time.time()
                successes = 0
                
                for i in range(task_count):
                    result = agent.run(f"{base_task} #{i}")
                    if result.success:
                        successes += 1
                
                total_time = time.time() - start_time
                performance_stats = agent.get_performance_stats()
                
                result = BenchmarkResult(
                    test_name=f"Memory-{config['name']}",
                    tasks_completed=task_count,
                    total_time=total_time,
                    avg_time_per_task=total_time / task_count,
                    throughput_tps=task_count / total_time,
                    success_rate=successes / task_count,
                    cache_hit_rate=performance_stats["derived_metrics"]["cache_hit_rate"],
                    metadata=performance_stats["cache_stats"]
                )
                
                self.results.append(result)
                print(f"  Results: Cache hit rate {result.cache_hit_rate:.1%}")
    
    def benchmark_concurrent_load(self):
        """Benchmark performance under concurrent load."""
        with self.benchmark_context("Concurrent Load"):
            agent = OptimizedReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_parallel_execution=True,
                max_concurrent_tasks=10
            )
            
            concurrent_levels = [1, 2, 4, 8]
            tasks_per_thread = 10
            base_task = "Solve algorithmic problem"
            
            for concurrency in concurrent_levels:
                print(f"\nTesting {concurrency} concurrent threads...")
                
                start_time = time.time()
                total_tasks = concurrency * tasks_per_thread
                results = []
                
                def worker_thread(thread_id):
                    """Worker thread function."""
                    thread_results = []
                    for i in range(tasks_per_thread):
                        try:
                            result = agent.run(f"{base_task} T{thread_id}-{i}")
                            thread_results.append(result)
                        except Exception as e:
                            print(f"Task failed in thread {thread_id}: {str(e)}")
                    return thread_results
                
                # Execute concurrent threads
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(worker_thread, tid) 
                        for tid in range(concurrency)
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            thread_results = future.result()
                            results.extend(thread_results)
                        except Exception as e:
                            print(f"Thread execution failed: {str(e)}")
                
                total_time = time.time() - start_time
                successes = sum(1 for r in results if hasattr(r, 'success') and r.success)
                
                result = BenchmarkResult(
                    test_name=f"Concurrent-{concurrency}threads",
                    tasks_completed=total_tasks,
                    total_time=total_time,
                    avg_time_per_task=total_time / total_tasks if total_tasks > 0 else 0,
                    throughput_tps=total_tasks / total_time if total_time > 0 else 0,
                    success_rate=successes / total_tasks if total_tasks > 0 else 0,
                    metadata={"concurrency_level": concurrency, "completed_tasks": len(results)}
                )
                
                self.results.append(result)
                print(f"  Results: {result.throughput_tps:.2f} tasks/sec with {concurrency} threads")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by test category
        categories = {}
        for result in self.results:
            category = result.test_name.split('-')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Calculate summary statistics
        all_throughputs = [r.throughput_tps for r in self.results if r.throughput_tps > 0]
        all_success_rates = [r.success_rate for r in self.results]
        
        summary = {
            "total_tests": len(self.results),
            "total_tasks_processed": sum(r.tasks_completed for r in self.results),
            "avg_throughput": statistics.mean(all_throughputs) if all_throughputs else 0,
            "max_throughput": max(all_throughputs) if all_throughputs else 0,
            "avg_success_rate": statistics.mean(all_success_rates) if all_success_rates else 0,
            "performance_categories": {}
        }
        
        # Category-specific analysis
        for category, results in categories.items():
            throughputs = [r.throughput_tps for r in results if r.throughput_tps > 0]
            success_rates = [r.success_rate for r in results]
            
            summary["performance_categories"][category] = {
                "test_count": len(results),
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "best_result": max(results, key=lambda r: r.throughput_tps).to_dict() if results else None
            }
        
        # Performance insights
        insights = []
        
        if "Optimized" in categories:
            optimized_results = categories["Optimized"]
            baseline = next((r for r in optimized_results if "Baseline" in r.test_name), None)
            full_opt = next((r for r in optimized_results if "Full-Optimized" in r.test_name), None)
            
            if baseline and full_opt:
                improvement = (full_opt.throughput_tps - baseline.throughput_tps) / baseline.throughput_tps * 100
                insights.append(f"Full optimization improved throughput by {improvement:.1f}%")
        
        if summary["max_throughput"] > 10:
            insights.append("System demonstrates high-throughput capabilities (>10 tasks/sec)")
        
        if summary["avg_success_rate"] > 0.9:
            insights.append("High reliability maintained across all tests (>90% success rate)")
        
        return {
            "summary": summary,
            "insights": insights,
            "detailed_results": [r.to_dict() for r in self.results],
            "timestamp": time.time()
        }
    
    def save_results(self, filename: str = "scalability_benchmark_results.json"):
        """Save benchmark results to file."""
        report = self.generate_performance_report()
        
        results_file = Path(filename)
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return results_file


async def run_full_benchmark_suite():
    """Run the complete scalability benchmark suite."""
    print("🚀 REFLEXION SCALABILITY BENCHMARK SUITE")
    print("=" * 60)
    
    benchmark = ScalabilityBenchmark()
    
    try:
        # Run all benchmarks
        benchmark.benchmark_basic_throughput()
        benchmark.benchmark_optimized_performance()
        await benchmark.benchmark_batch_processing()
        await benchmark.benchmark_autoscaling()
        benchmark.benchmark_memory_efficiency()
        benchmark.benchmark_concurrent_load()
        
        # Generate and save report
        results_file = benchmark.save_results()
        report = benchmark.generate_performance_report()
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print('='*60)
        
        summary = report["summary"]
        print(f"📊 Total tests completed: {summary['total_tests']}")
        print(f"🚀 Total tasks processed: {summary['total_tasks_processed']}")
        print(f"⚡ Average throughput: {summary['avg_throughput']:.2f} tasks/sec")
        print(f"🏆 Maximum throughput: {summary['max_throughput']:.2f} tasks/sec")
        print(f"✅ Average success rate: {summary['avg_success_rate']:.1%}")
        
        print(f"\n📈 Performance by Category:")
        for category, stats in summary["performance_categories"].items():
            print(f"  {category}: {stats['avg_throughput']:.2f} tasks/sec avg, "
                  f"{stats['max_throughput']:.2f} tasks/sec max")
        
        if report["insights"]:
            print(f"\n💡 Key Insights:")
            for insight in report["insights"]:
                print(f"  • {insight}")
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        print("\n🎉 Scalability benchmark suite completed!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_full_benchmark_suite())