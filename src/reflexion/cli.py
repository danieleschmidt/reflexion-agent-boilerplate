"""Enhanced command line interface for reflexion agent with research capabilities."""

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

from .core.agent import ReflexionAgent, ReflectionType
from .memory.episodic import EpisodicMemory
from .core.logging_config import logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reflexion Agent CLI - Self-improving language agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  reflexion "Write a Python function to sort a list"
  reflexion "Analyze this dataset" --llm gpt-3.5-turbo --max-iterations 5
  reflexion "Create a web scraper" --success-criteria "handles errors,robust"
  reflexion --memory-stats  # Show memory statistics
        """
    )
    
    # Task execution arguments
    parser.add_argument("task", nargs="?", help="Task to execute with reflexion")
    parser.add_argument("--llm", default="gpt-4", help="LLM model to use (default: gpt-4)")
    parser.add_argument("--max-iterations", type=int, default=3, 
                       help="Maximum reflection iterations (default: 3)")
    parser.add_argument("--reflection-type", choices=["binary", "scalar", "structured"], 
                       default="binary", help="Type of reflection to perform (default: binary)")
    parser.add_argument("--success-criteria", help="Comma-separated success criteria")
    parser.add_argument("--success-threshold", type=float, default=0.7,
                       help="Success threshold (0.0-1.0, default: 0.7)")
    
    # Memory and analysis arguments
    parser.add_argument("--memory-path", default="./reflexion_memory.json",
                       help="Path to memory storage (default: ./reflexion_memory.json)")
    parser.add_argument("--memory-stats", action="store_true",
                       help="Show memory statistics and patterns")
    parser.add_argument("--recall-similar", metavar="TASK",
                       help="Recall similar episodes for the given task")
    
    # Output arguments
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output with detailed reflections")
    parser.add_argument("--json-output", action="store_true",
                       help="Output results in JSON format")
    
    args = parser.parse_args()
    
    try:
        # Initialize memory system with error handling
        try:
            memory = EpisodicMemory(storage_path=args.memory_path)
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            print(f"Warning: Memory system unavailable ({e}). Continuing without persistence.")
            memory = None
        
        # Handle memory-related commands
        if args.memory_stats:
            if memory:
                show_memory_stats(memory)
            else:
                print("Memory system unavailable.")
            return
        
        if args.recall_similar:
            if memory:
                recall_similar_episodes(memory, args.recall_similar)
            else:
                print("Memory system unavailable.")
            return
        
        # Require task for execution
        if not args.task:
            parser.error("Task is required for execution. Use --help for usage information.")
        
        # Execute task with reflexion and robust error handling
        result = execute_task_with_reflexion(args, memory)
        
        # Output results
        if args.json_output:
            output_json_result(result)
        else:
            output_human_result(result, verbose=args.verbose)
            
    except KeyboardInterrupt:
        print("\\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        if args.verbose:
            print(f"\\nDetailed error traceback:")
            traceback.print_exc()
        print(f"Error: {e}")
        sys.exit(1)


def execute_task_with_reflexion(args, memory) -> Any:
    """Execute task with reflexion and robust error handling."""
    try:
        # Create agent with enhanced configuration
        agent_config = {
            "llm": args.llm,
            "max_iterations": args.max_iterations,
            "reflection_type": ReflectionType(args.reflection_type),
            "success_threshold": args.success_threshold
        }
            
        agent = ReflexionAgent(**agent_config)
        
        # Execute with timeout protection
        start_time = time.time()
        
        try:
            result = agent.run(args.task, success_criteria=args.success_criteria)
                
        except Exception as execution_error:
            execution_time = time.time() - start_time
            logger.warning(f"Task execution failed after {execution_time:.2f}s: {execution_error}")
            raise Exception(f"Task execution failed: {execution_error}")
            
        # Store result in memory if available
        if memory:
            try:
                memory.store_episode(args.task, result, metadata={
                    "llm": args.llm,
                    "reflection_type": args.reflection_type,
                    "threshold": args.success_threshold,
                    "cli_execution": True,
                    "execution_time": time.time() - start_time
                })
            except Exception as e:
                logger.warning(f"Failed to store episode in memory: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise


def show_memory_stats(memory: EpisodicMemory):
    """Show memory statistics and patterns."""
    patterns = memory.get_success_patterns()
    
    print("=== Reflexion Memory Statistics ===")
    print(f"Total episodes: {patterns['total_episodes']}")
    print(f"Successful episodes: {patterns['successful_episodes']}")
    print(f"Success rate: {patterns['success_rate']:.2%}")
    
    if patterns['patterns']:
        print("\\nTop improvement patterns:")
        for i, (pattern, count) in enumerate(patterns['patterns'][:5], 1):
            print(f"  {i}. {pattern} (used {count} times)")
    else:
        print("\\nNo patterns found yet. Execute some tasks to build memory!")


def recall_similar_episodes(memory: EpisodicMemory, task: str):
    """Recall and display similar episodes."""
    similar = memory.recall_similar(task, k=5)
    
    print(f"=== Similar Episodes for: {task} ===")
    if not similar:
        print("No similar episodes found.")
        return
    
    for i, episode in enumerate(similar, 1):
        print(f"\\n{i}. Task: {episode.task}")
        print(f"   Success: {'✓' if episode.result.success else '✗'}")
        print(f"   Iterations: {episode.result.iterations}")
        print(f"   Time: {episode.result.total_time:.2f}s")
        
        if episode.result.reflections:
            latest_reflection = episode.result.reflections[-1]
            if latest_reflection.improvements:
                print(f"   Key improvements: {', '.join(latest_reflection.improvements[:2])}")


def output_json_result(result):
    """Output result in JSON format."""
    result_dict = {
        "task": result.task,
        "success": result.success,
        "iterations": result.iterations,
        "output": result.output,
        "total_time": result.total_time,
        "reflections": [
            {
                "issues": r.issues,
                "improvements": r.improvements,
                "confidence": r.confidence,
                "score": r.score,
                "timestamp": r.timestamp
            } for r in result.reflections
        ],
        "metadata": result.metadata
    }
    print(json.dumps(result_dict, indent=2))


def output_human_result(result, verbose: bool = False):
    """Output result in human-readable format."""
    status_emoji = "✅" if result.success else "❌"
    
    print(f"\\n{status_emoji} Task: {result.task}")
    print(f"📊 Result: {'Success' if result.success else 'Failed'} after {result.iterations} iteration(s)")
    print(f"⏱️  Time: {result.total_time:.2f} seconds")
    print(f"\\n📝 Output:\\n{result.output}")
    
    if result.reflections:
        print(f"\\n🤔 Reflections ({len(result.reflections)}):")
        for i, reflection in enumerate(result.reflections, 1):
            print(f"\\n  Iteration {i} (confidence: {reflection.confidence:.2f}):")
            
            if reflection.issues:
                print(f"    ❗ Issues identified:")
                for issue in reflection.issues:
                    print(f"      • {issue}")
            
            if reflection.improvements:
                print(f"    💡 Improvements suggested:")
                for improvement in reflection.improvements:
                    print(f"      • {improvement}")
            
            if verbose and reflection.score:
                print(f"    📈 Score: {reflection.score:.2f}")
    
    if verbose and result.metadata:
        print(f"\\n🔧 Metadata: {result.metadata}")


if __name__ == "__main__":
    main()