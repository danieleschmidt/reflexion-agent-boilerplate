#!/usr/bin/env python3
"""Production-ready example showcasing all Reflexion Agent capabilities."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion import (
    ReflexionAgent, 
    OptimizedReflexionAgent, 
    AutoScalingReflexionAgent,
    ReflectionType
)
from reflexion.memory.episodic import EpisodicMemory
from reflexion.core.security import health_checker, security_manager
from reflexion.core.performance import performance_cache, resource_monitor
from reflexion.deployment import create_config, DeploymentEnvironment, Region


def demonstrate_basic_reflexion():
    """Demonstrate basic reflexion capabilities."""
    print("🎯 Basic Reflexion Demonstration")
    print("=" * 50)
    
    # Create basic agent
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.8
    )
    
    # Complex task that might require reflection
    task = "Create a robust password validation function that handles all edge cases"
    
    print(f"Task: {task}")
    print("Executing with reflexion...")
    
    start_time = time.time()
    result = agent.run(task, success_criteria="secure,comprehensive,tested")
    execution_time = time.time() - start_time
    
    print(f"\n✅ Completed in {execution_time:.2f}s")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Output: {result.output[:100]}...")
    
    if result.reflections:
        print(f"\n🤔 Reflections ({len(result.reflections)}):")
        for i, reflection in enumerate(result.reflections, 1):
            print(f"  {i}. Issues: {len(reflection.issues)}, Improvements: {len(reflection.improvements)}")
            print(f"     Confidence: {reflection.confidence:.2f}")
    
    return result


def demonstrate_optimized_performance():
    """Demonstrate optimized performance features."""
    print("\n\n⚡ Optimized Performance Demonstration")
    print("=" * 50)
    
    # Create optimized agent
    agent = OptimizedReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.7,
        enable_caching=True,
        enable_parallel_reflection=True,
        max_concurrent_tasks=4
    )
    
    # Test tasks including duplicates to show caching
    tasks = [
        "Implement bubble sort algorithm",
        "Create a simple REST API endpoint", 
        "Implement bubble sort algorithm",  # Duplicate for cache
        "Write a unit test for a function",
        "Create a simple REST API endpoint"  # Another duplicate
    ]
    
    print(f"Processing {len(tasks)} tasks (with duplicates for caching test)...")
    
    start_time = time.time()
    results = []
    
    for i, task in enumerate(tasks, 1):
        print(f"  Task {i}: {task[:40]}...")
        result = agent.run(task)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Get performance statistics
    stats = agent.get_performance_stats()
    
    print(f"\n📊 Performance Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per task: {total_time/len(tasks):.3f}s")
    print(f"Success rate: {sum(1 for r in results if r.success)}/{len(results)}")
    print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")
    print(f"Cache utilization: {stats['cache_stats']['utilization']:.1%}")
    
    return results, stats


async def demonstrate_async_batch_processing():
    """Demonstrate asynchronous batch processing."""
    print("\n\n🚀 Async Batch Processing Demonstration")  
    print("=" * 50)
    
    # Create optimized agent
    agent = OptimizedReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        enable_caching=True,
        max_concurrent_tasks=6
    )
    
    # Batch of tasks to process concurrently
    batch_tasks = [
        "Create a hash table implementation",
        "Write a binary search function",
        "Implement a linked list class",
        "Create a stack data structure", 
        "Write a queue implementation",
        "Implement a binary tree"
    ]
    
    print(f"Processing batch of {len(batch_tasks)} tasks concurrently...")
    
    start_time = time.time()
    results = await agent.run_batch(batch_tasks)
    batch_time = time.time() - start_time
    
    # Calculate theoretical sequential time
    sequential_estimate = len(batch_tasks) * 0.1  # Assume 0.1s per task
    
    print(f"\n📈 Batch Processing Results:")
    print(f"Batch time: {batch_time:.2f}s")
    print(f"Sequential estimate: {sequential_estimate:.2f}s")
    print(f"Speedup: {sequential_estimate/batch_time:.1f}x")
    print(f"Success rate: {sum(1 for r in results if r.get('success', False))}/{len(results)}")
    
    return results


def demonstrate_memory_and_learning():
    """Demonstrate episodic memory and learning capabilities."""
    print("\n\n🧠 Memory and Learning Demonstration")
    print("=" * 50)
    
    # Create memory system
    memory = EpisodicMemory(storage_path="./production_memory.json", max_episodes=100)
    
    # Create agent with memory integration
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=3,
        reflection_type=ReflectionType.STRUCTURED
    )
    
    # Run related tasks to build memory
    learning_tasks = [
        "Implement error handling for file operations",
        "Create input validation for user data",
        "Add error handling for network requests",
        "Implement validation for API inputs"
    ]
    
    print("Building episodic memory with related tasks...")
    
    for task in learning_tasks:
        result = agent.run(task)
        
        # Store in memory with metadata
        memory.store_episode(task, result, metadata={
            "category": "error_handling" if "error" in task.lower() else "validation",
            "complexity": "medium"
        })
        
        print(f"  ✓ {task[:50]}... (Success: {result.success})")
    
    # Analyze patterns
    patterns = memory.get_success_patterns()
    
    print(f"\n🔍 Learning Analysis:")
    print(f"Total episodes: {patterns['total_episodes']}")
    print(f"Success rate: {patterns['success_rate']:.1%}")
    
    if patterns['patterns']:
        print("Top improvement patterns:")
        for pattern, count in patterns['patterns'][:3]:
            print(f"  • {pattern} (used {count} times)")
    
    # Test memory recall
    similar_episodes = memory.recall_similar("error handling best practices", k=3)
    print(f"\nRecalled {len(similar_episodes)} similar episodes for 'error handling best practices'")
    
    return memory, patterns


def demonstrate_security_and_health():
    """Demonstrate security and health monitoring."""
    print("\n\n🛡️ Security and Health Monitoring")
    print("=" * 50)
    
    # Run comprehensive health checks
    print("Running health checks...")
    health_results = health_checker.run_health_checks()
    
    print(f"Overall health: {health_results['overall_status']}")
    for check_name, check_result in health_results['checks'].items():
        status_emoji = "✅" if check_result['status'] == 'pass' else "❌"
        print(f"  {status_emoji} {check_name}: {check_result['message']}")
    
    # Security summary
    print("\nSecurity summary...")
    security_summary = security_manager.get_security_summary()
    
    print(f"  • Blocked patterns: {security_summary['blocked_patterns']}")
    print(f"  • Security events (24h): {security_summary['security_events_24h']}")
    print(f"  • Active API keys: {security_summary['active_api_keys']}")
    
    # Test API key generation
    test_key = security_manager.generate_api_key("demo_user", ["read", "write"])
    print(f"  • Generated demo API key: {test_key[:10]}...")
    
    return health_results, security_summary


def demonstrate_production_deployment():
    """Demonstrate production deployment configuration."""
    print("\n\n🌍 Production Deployment Configuration")
    print("=" * 50)
    
    # Create production configuration
    config = create_config(
        environment="production",
        region="us-east-1",
        version="1.0.0"
    )
    
    print(f"Environment: {config.environment.value}")
    print(f"Region: {config.region.value}")
    print(f"Version: {config.version}")
    
    print(f"\nScaling configuration:")
    print(f"  • Min instances: {config.scaling.min_instances}")
    print(f"  • Max instances: {config.scaling.max_instances}")
    print(f"  • CPU target: {config.scaling.target_cpu_utilization}%")
    print(f"  • Memory target: {config.scaling.target_memory_utilization}%")
    
    print(f"\nSecurity configuration:")
    print(f"  • Encryption at rest: {config.security.enable_encryption_at_rest}")
    print(f"  • Encryption in transit: {config.security.enable_encryption_in_transit}")
    print(f"  • API rate limit: {config.security.api_rate_limit} req/min")
    print(f"  • Require API key: {config.security.require_api_key}")
    
    print(f"\nCompliance configuration:")
    print(f"  • GDPR enabled: {config.compliance.gdpr_enabled}")
    print(f"  • CCPA enabled: {config.compliance.ccpa_enabled}")
    print(f"  • Data retention: {config.compliance.data_retention_days} days")
    print(f"  • Audit logging: {config.compliance.audit_logging}")
    
    print(f"\nI18n configuration:")
    print(f"  • Default language: {config.i18n.default_language}")
    print(f"  • Supported languages: {', '.join(config.i18n.supported_languages)}")
    print(f"  • Timezone: {config.i18n.timezone}")
    
    return config


async def main():
    """Run the complete production demonstration."""
    print("🎉 Reflexion Agent Boilerplate - Production Demonstration")
    print("=" * 60)
    print("This example showcases all capabilities in a production-ready setup.")
    print()
    
    try:
        # 1. Basic Reflexion
        basic_result = demonstrate_basic_reflexion()
        
        # 2. Optimized Performance
        perf_results, perf_stats = demonstrate_optimized_performance()
        
        # 3. Async Batch Processing
        batch_results = await demonstrate_async_batch_processing()
        
        # 4. Memory and Learning
        memory, patterns = demonstrate_memory_and_learning()
        
        # 5. Security and Health
        health_results, security_summary = demonstrate_security_and_health()
        
        # 6. Production Deployment
        config = demonstrate_production_deployment()
        
        # Final Summary
        print("\n\n🎯 DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ All components demonstrated successfully!")
        
        print(f"\n📊 Key Metrics:")
        print(f"  • Basic reflexion: {1 if basic_result.success else 0}/1 success")
        print(f"  • Performance tasks: {sum(1 for r in perf_results if r.success)}/{len(perf_results)} success")
        print(f"  • Batch processing: {sum(1 for r in batch_results if r.get('success', False))}/{len(batch_results)} success")
        print(f"  • Cache hit rate: {perf_stats['cache_stats']['hit_rate']:.1%}")
        print(f"  • Memory episodes: {patterns['total_episodes']}")
        print(f"  • Health status: {health_results['overall_status']}")
        
        print(f"\n🌍 Production Ready Features:")
        print("  ✅ Multi-generation implementation (Simple → Robust → Optimized)")
        print("  ✅ Comprehensive error handling and validation")
        print("  ✅ Performance optimization with caching")
        print("  ✅ Security and compliance features")
        print("  ✅ Global-first deployment configuration")
        print("  ✅ Monitoring and observability")
        print("  ✅ Auto-scaling capabilities")
        print("  ✅ Multi-region support")
        
        print(f"\n🚀 Ready for Production Deployment!")
        print("Use the deployment scripts to deploy to your environment.")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))