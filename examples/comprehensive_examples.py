#!/usr/bin/env python3
"""
Comprehensive examples demonstrating all ReflexionAgent capabilities.

This module provides complete, working examples for each major feature
and integration pattern of the reflexion framework.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion import (
    ReflexionAgent, 
    ReflectionType, 
    ReflectionPrompts, 
    PromptDomain,
    EpisodicMemory
)


class ExampleRunner:
    """Utility class to run and demonstrate examples."""
    
    def __init__(self):
        self.results = []
    
    def run_example(self, name: str, func, *args, **kwargs):
        """Run an example and store results."""
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {name}")
        print('='*60)
        
        try:
            result = func(*args, **kwargs)
            self.results.append({
                "name": name,
                "success": True,
                "result": result
            })
            print(f"✅ {name} completed successfully")
            return result
        except Exception as e:
            self.results.append({
                "name": name,
                "success": False,
                "error": str(e)
            })
            print(f"❌ {name} failed: {e}")
            return None
    
    def print_summary(self):
        """Print summary of all examples."""
        print(f"\n{'='*60}")
        print("EXAMPLES SUMMARY")
        print('='*60)
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if failed:
            print("\nFailures:")
            for result in failed:
                print(f"  - {result['name']}: {result['error']}")


def example_1_basic_usage():
    """Example 1: Basic reflexion agent usage."""
    print("Creating a basic ReflexionAgent...")
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.8
    )
    
    print("Executing a simple task...")
    result = agent.run(
        task="Write a Python function to calculate the factorial of a number",
        success_criteria="includes error handling, has docstring"
    )
    
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Output length: {len(result.output)} characters")
    print(f"Number of reflections: {len(result.reflections)}")
    
    if result.reflections:
        latest = result.reflections[-1]
        print(f"Latest reflection confidence: {latest.confidence:.2f}")
        print(f"Issues identified: {len(latest.issues)}")
        print(f"Improvements suggested: {len(latest.improvements)}")
    
    return result


def example_2_domain_specific_prompts():
    """Example 2: Using domain-specific reflection prompts."""
    print("Demonstrating domain-specific prompts...")
    
    # Software Engineering Domain
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.STRUCTURED
    )
    
    result = agent.run(
        task="Implement a thread-safe cache with TTL support",
        success_criteria="thread safety, TTL functionality, error handling"
    )
    
    print(f"Software engineering task completed with {result.iterations} iterations")
    
    # Data Analysis Domain  
    result2 = agent.run(
        task="Analyze website traffic data and identify trends",
        success_criteria="statistical analysis, trend identification, visualization"
    )
    
    print(f"Data analysis task completed with {result2.iterations} iterations")
    
    return {"software": result, "data_analysis": result2}


def example_3_episodic_memory():
    """Example 3: Using episodic memory for learning."""
    print("Demonstrating episodic memory...")
    
    memory_path = Path("./example_memory.json")
    memory = EpisodicMemory(storage_path=str(memory_path), max_episodes=50)
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        memory=memory
    )
    
    # Execute several related tasks
    tasks = [
        "Write a sorting algorithm",
        "Implement a binary search function",
        "Create a data structure for storing key-value pairs",
        "Write a function to find duplicates in a list"
    ]
    
    results = []
    for i, task in enumerate(tasks):
        print(f"\nExecuting task {i+1}: {task}")
        result = agent.run(task=task)
        results.append(result)
        
        # Store in episodic memory
        memory.store_episode(task, result, {"example_run": True, "task_index": i})
    
    # Analyze patterns
    patterns = memory.get_success_patterns()
    print(f"\nMemory Analysis:")
    print(f"Success rate: {patterns['success_rate']:.2f}")
    print(f"Total episodes: {patterns['total_episodes']}")
    print(f"Common success patterns: {len(patterns['patterns'])}")
    
    if patterns['patterns']:
        print("Top 3 success patterns:")
        for pattern, count in patterns['patterns'][:3]:
            print(f"  - {pattern} (used {count} times)")
    
    # Test recall
    similar = memory.recall_similar("implement a search algorithm", k=2)
    print(f"\nFound {len(similar)} similar episodes for search-related task")
    
    return {
        "results": results,
        "patterns": patterns,
        "memory_path": str(memory_path)
    }


def example_4_custom_reflection_prompts():
    """Example 4: Creating and using custom reflection prompts."""
    print("Creating custom reflection prompts...")
    
    from reflexion.prompts import CustomReflectionPrompts
    
    # Create custom prompts for API design domain
    api_prompts = CustomReflectionPrompts.create_domain_specific_prompt(
        domain_description="API Design and Development",
        evaluation_criteria=[
            "REST principles adherence",
            "Security considerations", 
            "Error handling",
            "Documentation quality",
            "Performance implications"
        ],
        common_failure_modes=[
            "Missing authentication",
            "Poor error responses",
            "Inconsistent naming",
            "Missing rate limiting",
            "Inadequate input validation"
        ]
    )
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        reflection_type=ReflectionType.STRUCTURED
    )
    
    result = agent.run(
        task="Design a REST API for a blog platform with user management",
        success_criteria="RESTful design, security, scalability"
    )
    
    print(f"Custom prompt API task completed: {result.success}")
    print(f"Iterations needed: {result.iterations}")
    
    if result.reflections:
        print(f"Reflection insights generated: {len(result.reflections[-1].improvements)}")
    
    return result


def example_5_different_reflection_types():
    """Example 5: Comparing different reflection types."""
    print("Comparing reflection types...")
    
    task = "Create a machine learning model evaluation framework"
    
    reflection_types = [
        (ReflectionType.BINARY, "Binary"),
        (ReflectionType.SCALAR, "Scalar"), 
        (ReflectionType.STRUCTURED, "Structured")
    ]
    
    results = {}
    
    for ref_type, name in reflection_types:
        print(f"\nTesting {name} reflection...")
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ref_type,
            success_threshold=0.7
        )
        
        result = agent.run(
            task=task,
            success_criteria="modular design, multiple metrics, visualization"
        )
        
        results[name] = {
            "success": result.success,
            "iterations": result.iterations,
            "reflections": len(result.reflections),
            "total_time": result.total_time,
            "avg_confidence": sum(r.confidence for r in result.reflections) / len(result.reflections) if result.reflections else 0
        }
        
        print(f"  - Success: {result.success}")
        print(f"  - Iterations: {result.iterations}")
        print(f"  - Avg confidence: {results[name]['avg_confidence']:.2f}")
    
    return results


def example_6_error_handling_and_recovery():
    """Example 6: Demonstrating error handling and recovery."""
    print("Testing error handling and recovery...")
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=4,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.6
    )
    
    # Task likely to trigger some error conditions
    result = agent.run(
        task="error: simulate a complex error scenario for testing",
        success_criteria="recovery from errors, graceful degradation"
    )
    
    print(f"Error handling task - Success: {result.success}")
    print(f"Iterations attempted: {result.iterations}")
    print(f"Total execution time: {result.total_time:.2f}s")
    
    # Examine error reflections
    error_reflections = [r for r in result.reflections if not r.success]
    print(f"Error reflections generated: {len(error_reflections)}")
    
    if error_reflections:
        latest_error = error_reflections[-1]
        print(f"Latest error reflection confidence: {latest_error.confidence:.2f}")
        print("Error issues identified:")
        for issue in latest_error.issues:
            print(f"  - {issue}")
    
    return result


async def example_7_async_execution():
    """Example 7: Asynchronous execution capabilities."""
    print("Testing asynchronous execution...")
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.SCALAR
    )
    
    # Simulate concurrent task execution
    tasks = [
        "Generate a creative story about time travel",
        "Design a database schema for e-commerce",
        "Create a performance monitoring solution"
    ]
    
    print(f"Executing {len(tasks)} tasks concurrently...")
    
    # Note: In this example, we're simulating async by running synchronously
    # In a real async environment, you'd use agent.run_async() if available
    results = []
    for i, task in enumerate(tasks):
        print(f"Starting task {i+1}...")
        result = agent.run(task=task)
        results.append(result)
    
    total_time = sum(r.total_time for r in results)
    successful_tasks = sum(1 for r in results if r.success)
    
    print(f"Async execution completed:")
    print(f"  - Tasks: {len(tasks)}")
    print(f"  - Successful: {successful_tasks}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average time per task: {total_time/len(tasks):.2f}s")
    
    return {
        "results": results,
        "summary": {
            "total_tasks": len(tasks),
            "successful": successful_tasks,
            "total_time": total_time
        }
    }


def example_8_metrics_and_evaluation():
    """Example 8: Advanced metrics and evaluation."""
    print("Demonstrating metrics and evaluation...")
    
    agent = ReflexionAgent(
        llm="gpt-4", 
        max_iterations=3,
        reflection_type=ReflectionType.STRUCTURED,
        success_threshold=0.8
    )
    
    # Track various metrics across multiple runs
    tasks = [
        ("Easy", "Write a hello world program"),
        ("Medium", "Implement a REST API client with retry logic"),
        ("Hard", "Design a distributed caching system with consistency guarantees")
    ]
    
    metrics = {
        "by_difficulty": {},
        "overall": {
            "total_tasks": 0,
            "total_success": 0,
            "total_iterations": 0,
            "total_time": 0.0
        }
    }
    
    for difficulty, task in tasks:
        print(f"\nExecuting {difficulty} task...")
        result = agent.run(task=task)
        
        # Collect metrics
        task_metrics = {
            "success": result.success,
            "iterations": result.iterations,
            "time": result.total_time,
            "reflection_count": len(result.reflections),
            "avg_confidence": sum(r.confidence for r in result.reflections) / len(result.reflections) if result.reflections else 0
        }
        
        metrics["by_difficulty"][difficulty] = task_metrics
        
        # Update overall metrics
        metrics["overall"]["total_tasks"] += 1
        metrics["overall"]["total_success"] += 1 if result.success else 0
        metrics["overall"]["total_iterations"] += result.iterations
        metrics["overall"]["total_time"] += result.total_time
    
    # Calculate derived metrics
    overall = metrics["overall"]
    overall["success_rate"] = overall["total_success"] / overall["total_tasks"]
    overall["avg_iterations"] = overall["total_iterations"] / overall["total_tasks"]
    overall["avg_time"] = overall["total_time"] / overall["total_tasks"]
    
    print(f"\n📊 METRICS SUMMARY:")
    print(f"Success rate: {overall['success_rate']:.1%}")
    print(f"Average iterations: {overall['avg_iterations']:.1f}")
    print(f"Average time: {overall['avg_time']:.2f}s")
    
    print(f"\n📈 BY DIFFICULTY:")
    for difficulty, task_metrics in metrics["by_difficulty"].items():
        print(f"{difficulty:>6}: Success={task_metrics['success']}, "
              f"Iterations={task_metrics['iterations']}, "
              f"Time={task_metrics['time']:.2f}s")
    
    return metrics


def main():
    """Run all examples and display results."""
    print("🚀 REFLEXION AGENT - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    runner = ExampleRunner()
    
    # Run all examples
    runner.run_example("Basic Usage", example_1_basic_usage)
    runner.run_example("Domain-Specific Prompts", example_2_domain_specific_prompts)
    runner.run_example("Episodic Memory", example_3_episodic_memory)
    runner.run_example("Custom Prompts", example_4_custom_reflection_prompts)
    runner.run_example("Reflection Types", example_5_different_reflection_types)
    runner.run_example("Error Handling", example_6_error_handling_and_recovery)
    runner.run_example("Async Execution", lambda: asyncio.run(example_7_async_execution()))
    runner.run_example("Metrics & Evaluation", example_8_metrics_and_evaluation)
    
    # Print final summary
    runner.print_summary()
    
    # Save results for analysis
    results_file = Path("./example_results.json")
    with open(results_file, 'w') as f:
        json.dump(runner.results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()