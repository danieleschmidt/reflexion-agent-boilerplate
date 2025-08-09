#!/usr/bin/env python3
"""Research capabilities example for the Reflexion Agent."""

import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reflexion.research import ResearchAgent, ResearchObjectiveType
from reflexion.research.experiment_runner import ExperimentRunner


async def research_example():
    """Demonstrate advanced research capabilities."""
    print("=== Reflexion Research Agent Example ===")
    
    # Initialize research agent
    research_agent = ResearchAgent(
        base_llm="gpt-4",
        output_dir="./research_output"
    )
    
    # Create performance optimization objective
    objective = research_agent.create_performance_optimization_objective(
        name="iteration_optimization",
        target_metric="success_rate",
        improvement_threshold=0.15
    )
    
    print(f"Research Objective: {objective.name}")
    print(f"Description: {objective.description}")
    print(f"Hypotheses: {len(objective.hypotheses)}")
    for i, hypothesis in enumerate(objective.hypotheses, 1):
        print(f"  {i}. {hypothesis}")
    
    # Define test scenarios for research
    test_scenarios = [
        "Write a function to find the longest common subsequence",
        "Implement a binary tree traversal algorithm",
        "Create a function to validate JSON structure",
        "Design a caching mechanism with TTL",
        "Write a parser for mathematical expressions"
    ]
    
    print(f"\nTest Scenarios: {len(test_scenarios)}")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"  {i}. {scenario}")
    
    print("\nConducting research study...")
    
    try:
        # Conduct research study
        findings = await research_agent.conduct_research_study(
            objective=objective,
            test_scenarios=test_scenarios,
            num_trials=10  # Reduced for demo
        )
        
        print(f"\nResearch Study Completed!")
        print(f"Total Findings: {len(findings)}")
        
        # Display findings
        for i, finding in enumerate(findings, 1):
            print(f"\nFinding {i}:")
            print(f"  Hypothesis: {finding.hypothesis}")
            print(f"  Supported: {finding.supported}")
            print(f"  Confidence: {finding.confidence:.3f}")
            print(f"  Practical Significance: {finding.practical_significance}")
            
            if finding.statistical_significance is not None:
                print(f"  Statistical Significance: p={finding.statistical_significance:.4f}")
            
            if finding.effect_size is not None:
                print(f"  Effect Size: {finding.effect_size:.3f}")
        
        # Research summary
        summary = research_agent.get_research_summary()
        print(f"\nResearch Agent Summary:")
        print(f"  Completed Studies: {summary['completed_studies']}")
        print(f"  Total Findings: {summary['total_findings']}")
        print(f"  Supported Findings: {summary['supported_findings']}")
        print(f"  Research Areas: {', '.join(summary['research_areas'])}")
        
    except Exception as e:
        print(f"Research study failed: {e}")
        import traceback
        traceback.print_exc()


async def algorithm_comparison_example():
    """Demonstrate algorithm comparison research."""
    print("\n\n=== Algorithm Comparison Research ===")
    
    research_agent = ResearchAgent(output_dir="./algorithm_research")
    
    # Create algorithm comparison objective
    objective = research_agent.create_algorithm_comparison_objective(
        name="reflection_type_comparison",
        algorithms=["binary_reflection", "scalar_reflection", "structured_reflection"]
    )
    
    print(f"Comparing Algorithms: {', '.join(objective.hypotheses)}")
    
    # Simpler test scenarios for comparison
    test_scenarios = [
        "Debug this code snippet",
        "Optimize this algorithm",
        "Add error handling"
    ]
    
    try:
        findings = await research_agent.conduct_research_study(
            objective=objective,
            test_scenarios=test_scenarios,
            num_trials=6  # Minimal for demo
        )
        
        print(f"\nAlgorithm Comparison Results:")
        supported_algorithms = [f for f in findings if f.supported]
        
        if supported_algorithms:
            print(f"Best performing algorithms:")
            for finding in supported_algorithms:
                print(f"  - {finding.hypothesis} (confidence: {finding.confidence:.3f})")
        else:
            print("No algorithm showed clear superiority")
            
    except Exception as e:
        print(f"Algorithm comparison failed: {e}")


def synchronous_research_example():
    """Synchronous version for compatibility."""
    print("\n\n=== Synchronous Research Demo ===")
    
    # Simple experiment runner demo
    from reflexion.research.experiment_runner import ExperimentConfig, ExperimentCondition, ExperimentType
    from reflexion import ReflexionAgent, ReflectionType
    
    # Create simple experiment
    baseline_agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY
    )
    
    test_task = "Write a function to calculate Fibonacci numbers"
    
    print(f"Testing task: {test_task}")
    
    # Run baseline
    print("Running baseline...")
    baseline_result = baseline_agent.run(test_task)
    print(f"Baseline result: Success={baseline_result.success}, "
          f"Iterations={baseline_result.iterations}")
    
    # Run with more iterations
    enhanced_agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=4,
        reflection_type=ReflectionType.BINARY
    )
    
    print("Running enhanced version...")
    enhanced_result = enhanced_agent.run(test_task)
    print(f"Enhanced result: Success={enhanced_result.success}, "
          f"Iterations={enhanced_result.iterations}")
    
    # Simple comparison
    if enhanced_result.success and not baseline_result.success:
        print("Hypothesis supported: More iterations improve success rate")
    elif enhanced_result.success == baseline_result.success:
        print("Hypothesis inconclusive: Similar performance")
    else:
        print("Hypothesis not supported: More iterations did not help")


async def main():
    """Run all research examples."""
    print("Reflexion Research Capabilities Demo")
    print("=" * 50)
    
    try:
        await research_example()
        await algorithm_comparison_example()
        synchronous_research_example()
        
        print("\n" + "=" * 50)
        print("Research examples completed!")
        
    except Exception as e:
        print(f"Error in research examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async examples
    asyncio.run(main())