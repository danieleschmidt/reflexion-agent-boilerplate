"""Benchmark runner for reflexion performance testing."""

import time
import statistics
from typing import List, Dict, Any

from src.reflexion.core.agent import ReflexionAgent
from src.reflexion.core.types import ReflectionType


def benchmark_reflexion_performance() -> Dict[str, Any]:
    """Benchmark reflexion agent performance."""
    tasks = [
        "Write a function to calculate factorial",
        "Implement binary search algorithm", 
        "Create a simple REST API endpoint",
        "Design a caching mechanism",
        "Implement depth-first search"
    ]
    
    results = {
        "total_tasks": len(tasks),
        "execution_times": [],
        "success_rates": [],
        "iteration_counts": [],
        "reflection_counts": []
    }
    
    agent = ReflexionAgent(
        llm="benchmark-model",
        max_iterations=3,
        reflection_type=ReflectionType.BINARY
    )
    
    for task in tasks:
        start_time = time.time()
        result = agent.run(task)
        execution_time = time.time() - start_time
        
        results["execution_times"].append(execution_time)
        results["success_rates"].append(1 if result.success else 0)
        results["iteration_counts"].append(result.iterations)
        results["reflection_counts"].append(len(result.reflections))
    
    # Calculate statistics
    results["avg_execution_time"] = statistics.mean(results["execution_times"])
    results["success_rate"] = statistics.mean(results["success_rates"])
    results["avg_iterations"] = statistics.mean(results["iteration_counts"])
    results["avg_reflections"] = statistics.mean(results["reflection_counts"])
    
    return results


def benchmark_reflection_types() -> Dict[str, Any]:
    """Compare performance across different reflection types."""
    task = "Implement a thread-safe counter class"
    reflection_types = [ReflectionType.BINARY, ReflectionType.SCALAR]
    
    results = {}
    
    for reflection_type in reflection_types:
        agent = ReflexionAgent(
            llm="benchmark-model",
            max_iterations=3,
            reflection_type=reflection_type
        )
        
        start_time = time.time()
        result = agent.run(task)
        execution_time = time.time() - start_time
        
        results[reflection_type.value] = {
            "execution_time": execution_time,
            "success": result.success,
            "iterations": result.iterations,
            "reflections": len(result.reflections)
        }
    
    return results


def main():
    """Run all benchmarks and print results."""
    print("Running Reflexion Performance Benchmarks...")
    print("=" * 50)
    
    # Performance benchmark
    print("\n1. General Performance Benchmark")
    perf_results = benchmark_reflexion_performance()
    print(f"Tasks completed: {perf_results['total_tasks']}")
    print(f"Average execution time: {perf_results['avg_execution_time']:.2f}s")
    print(f"Success rate: {perf_results['success_rate']:.1%}")
    print(f"Average iterations: {perf_results['avg_iterations']:.1f}")
    print(f"Average reflections: {perf_results['avg_reflections']:.1f}")
    
    # Reflection type comparison
    print("\n2. Reflection Type Comparison")
    type_results = benchmark_reflection_types()
    for reflection_type, metrics in type_results.items():
        print(f"\n{reflection_type.upper()}:")
        print(f"  Execution time: {metrics['execution_time']:.2f}s")
        print(f"  Success: {metrics['success']}")
        print(f"  Iterations: {metrics['iterations']}")
        print(f"  Reflections: {metrics['reflections']}")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()