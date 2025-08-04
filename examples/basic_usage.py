#!/usr/bin/env python3
"""Basic usage examples for the Reflexion Agent Boilerplate."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reflexion import ReflexionAgent, ReflectionType
from reflexion.memory.episodic import EpisodicMemory


def example_basic_usage():
    """Demonstrate basic reflexion agent usage."""
    print("=== Basic Reflexion Agent Usage ===")
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.7
    )
    
    task = "Write a Python function to calculate the factorial of a number"
    print(f"Task: {task}")
    
    result = agent.run(task, success_criteria="complete,tested,documented")
    
    print(f"\\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {result.total_time:.2f}s")
    print(f"  Output: {result.output}")
    
    if result.reflections:
        print(f"\\n  Reflections ({len(result.reflections)}):")
        for i, reflection in enumerate(result.reflections, 1):
            print(f"    {i}. Issues: {reflection.issues}")
            print(f"       Improvements: {reflection.improvements}")
    
    return result


def example_with_memory():
    """Demonstrate reflexion agent with memory."""
    print("\\n\\n=== Reflexion Agent with Memory ===")
    
    # Initialize memory
    memory = EpisodicMemory(storage_path="./example_memory.json")
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY
    )
    
    # Run several tasks
    tasks = [
        "Create a simple calculator function",
        "Write a function to reverse a string",
        "Implement binary search algorithm"
    ]
    
    for task in tasks:
        print(f"\\nExecuting: {task}")
        result = agent.run(task)
        
        # Store in memory
        memory.store_episode(task, result, metadata={"example": "with_memory"})
        
        print(f"  Success: {result.success}, Iterations: {result.iterations}")
    
    # Show memory patterns
    patterns = memory.get_success_patterns()
    print(f"\\nMemory Statistics:")
    print(f"  Total episodes: {patterns['total_episodes']}")
    print(f"  Success rate: {patterns['success_rate']:.2%}")
    
    if patterns['patterns']:
        print(f"  Top patterns:")
        for pattern, count in patterns['patterns'][:3]:
            print(f"    - {pattern} (used {count} times)")
    
    # Recall similar episodes
    similar = memory.recall_similar("algorithm implementation", k=2)
    print(f"\\nSimilar episodes for 'algorithm implementation':")
    for episode in similar:
        print(f"  - {episode.task} (success: {episode.result.success})")


def example_different_reflection_types():
    """Demonstrate different reflection types."""
    print("\\n\\n=== Different Reflection Types ===")
    
    task = "Optimize a slow database query"
    
    reflection_types = [
        (ReflectionType.BINARY, "Binary"),
        (ReflectionType.SCALAR, "Scalar"),
        (ReflectionType.STRUCTURED, "Structured")
    ]
    
    for reflection_type, name in reflection_types:
        print(f"\\n{name} Reflection:")
        
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=reflection_type,
            success_threshold=0.8
        )
        
        result = agent.run(task, success_criteria="performance,maintainable")
        
        print(f"  Success: {result.success}")
        print(f"  Reflections: {len(result.reflections)}")
        
        if result.reflections:
            latest = result.reflections[-1]
            print(f"  Confidence: {latest.confidence:.2f}")
            print(f"  Key improvements: {latest.improvements[:2] if latest.improvements else 'None'}")


def example_autogen_adapter():
    """Demonstrate AutoGen adapter usage."""
    print("\\n\\n=== AutoGen Adapter Example ===")
    
    try:
        from reflexion.adapters import AutoGenReflexion
        
        # Create reflexive AutoGen-style agent
        agent = AutoGenReflexion(
            name="ReflexiveCoder",
            system_message="You are a helpful coding assistant that learns from mistakes.",
            llm_config={"model": "gpt-4"},
            max_self_iterations=2,
            memory_window=5
        )
        
        # Simulate conversation
        messages = [
            "Help me write a function to validate email addresses",
            "The function should handle edge cases",
            "Can you add error handling?"
        ]
        
        for message in messages:
            print(f"\\nUser: {message}")
            response = agent.initiate_chat(message=message)
            print(f"Agent: {response[:100]}...")
        
        # Show reflection summary
        summary = agent.get_reflection_summary()
        print(f"\\nReflection Summary:")
        print(f"  Conversations: {summary['total_conversations']}")
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Avg reflections: {summary['avg_reflections_per_conversation']:.1f}")
        
    except ImportError:
        print("AutoGen adapter example skipped (dependencies not available)")


def main():
    """Run all examples."""
    print("Reflexion Agent Boilerplate - Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_with_memory()
        example_different_reflection_types()
        example_autogen_adapter()
        
        print("\\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()