"""Command line interface for reflexion agent."""

import argparse
import sys

from .core.agent import ReflexionAgent, ReflectionType


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Reflexion Agent CLI")
    parser.add_argument("task", help="Task to execute with reflexion")
    parser.add_argument("--llm", default="gpt-4", help="LLM model to use")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--reflection-type", choices=["binary", "scalar"], default="binary")
    
    args = parser.parse_args()
    
    agent = ReflexionAgent(
        llm=args.llm,
        max_iterations=args.max_iterations,
        reflection_type=ReflectionType(args.reflection_type)
    )
    
    result = agent.run(args.task)
    
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Output: {result.output}")
    
    if result.reflections:
        print("\nReflections:")
        for i, reflection in enumerate(result.reflections, 1):
            print(f"  {i}. Issues: {', '.join(reflection.issues)}")
            print(f"     Improvements: {', '.join(reflection.improvements)}")


if __name__ == "__main__":
    main()