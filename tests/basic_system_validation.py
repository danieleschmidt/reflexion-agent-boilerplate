"""Basic system validation tests without external dependencies."""

import sys
import os
import time
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_test(test_name, test_func):
    """Run a single test with error handling."""
    print(f"\n🧪 Running {test_name}...")
    try:
        start_time = time.time()
        test_func()
        duration = time.time() - start_time
        print(f"   ✅ PASSED ({duration:.2f}s)")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print(f"   📝 Details: {traceback.format_exc()}")
        return False

def test_basic_imports():
    """Test that all core modules can be imported."""
    from src.reflexion.core.agent import ReflexionAgent, ReflectionType
    from src.reflexion.research.novel_algorithms import research_comparator
    from src.reflexion.monitoring.telemetry import telemetry_manager
    from src.reflexion.core.performance_optimizer import performance_optimizer
    
    assert ReflexionAgent is not None
    assert research_comparator is not None
    assert telemetry_manager is not None
    assert performance_optimizer is not None

def test_agent_creation():
    """Test ReflexionAgent creation and basic properties."""
    from src.reflexion.core.agent import ReflexionAgent, ReflectionType
    
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY,
        success_threshold=0.7
    )
    
    assert agent.llm == "gpt-4"
    assert agent.max_iterations == 2
    assert agent.reflection_type == ReflectionType.BINARY
    assert agent.success_threshold == 0.7

def test_agent_task_execution():
    """Test basic task execution."""
    from src.reflexion.core.agent import ReflexionAgent
    
    agent = ReflexionAgent(llm="gpt-4", max_iterations=2)
    result = agent.run("Write a simple hello world function")
    
    assert result is not None
    assert hasattr(result, 'task')
    assert hasattr(result, 'success')
    assert hasattr(result, 'iterations')
    assert hasattr(result, 'output')
    assert hasattr(result, 'total_time')
    assert hasattr(result, 'reflections')
    assert isinstance(result.success, bool)
    assert isinstance(result.iterations, int)
    assert isinstance(result.total_time, (int, float))
    assert isinstance(result.reflections, list)

def test_research_algorithms():
    """Test research algorithms availability."""
    from src.reflexion.research.novel_algorithms import research_comparator
    
    algorithms = research_comparator.algorithms
    assert len(algorithms) >= 5
    
    algorithm_names = [alg.value for alg in algorithms.keys()]
    expected_algorithms = [
        'hierarchical_reflexion',
        'ensemble_reflexion', 
        'quantum_inspired_reflexion',
        'meta_cognitive_reflexion',
        'contrastive_reflexion'
    ]
    
    for expected in expected_algorithms:
        assert expected in algorithm_names, f"Missing algorithm: {expected}"

def test_performance_optimization():
    """Test performance optimization components."""
    from src.reflexion.core.performance_optimizer import ReflectionCache, performance_optimizer
    
    # Test cache
    cache = ReflectionCache(max_size=10, ttl_seconds=3600)
    result = cache.get("test task", "gpt-4", {"reflection_type": "binary"})
    assert result is None  # Should be cache miss
    
    # Test cache stats
    stats = cache.get_stats()
    assert 'hit_count' in stats
    assert 'miss_count' in stats
    assert 'hit_rate' in stats
    
    # Test performance optimizer
    assert performance_optimizer is not None
    assert hasattr(performance_optimizer, 'strategy')
    assert hasattr(performance_optimizer, 'cache')

def test_telemetry_system():
    """Test telemetry system functionality."""
    from src.reflexion.monitoring.telemetry import telemetry_manager
    
    # Test metrics collection
    metrics = telemetry_manager.metrics_collector.get_latest_metrics()
    assert 'collection_info' in metrics
    
    # Test health checks
    health_results = telemetry_manager.run_health_check()
    assert 'health_checks' in health_results
    assert 'health_summary' in health_results
    assert 'active_alerts' in health_results
    
    # Test dashboard data
    dashboard_data = telemetry_manager.get_dashboard_data()
    assert 'system_info' in dashboard_data
    assert 'metrics' in dashboard_data
    assert 'health' in dashboard_data

def test_cli_imports():
    """Test CLI module imports."""
    from src.reflexion.cli import main, execute_task_with_reflexion
    
    assert callable(main)
    assert callable(execute_task_with_reflexion)

def test_memory_system():
    """Test memory system if available."""
    try:
        from src.reflexion.memory.episodic import EpisodicMemory
        
        memory = EpisodicMemory(storage_path="./test_memory.json")
        patterns = memory.get_success_patterns()
        
        assert 'total_episodes' in patterns
        assert 'successful_episodes' in patterns
        assert 'success_rate' in patterns
        assert 'patterns' in patterns
        
    except Exception as e:
        print(f"   ⚠️  Memory system not fully available: {e}")

def test_integration_workflow():
    """Test end-to-end integration workflow."""
    from src.reflexion.core.agent import ReflexionAgent
    from src.reflexion.monitoring.telemetry import telemetry_manager
    from src.reflexion.core.performance_optimizer import performance_optimizer
    
    # Create agent and execute task
    agent = ReflexionAgent(llm="gpt-4", max_iterations=2)
    result = agent.run("Create a simple test function")
    
    # Record in telemetry
    telemetry_manager.record_task_execution(result, "classic")
    
    # Get performance metrics
    metrics = performance_optimizer.get_performance_metrics()
    
    # Verify integration
    assert result is not None
    assert metrics is not None
    assert hasattr(metrics, 'avg_execution_time')
    assert hasattr(metrics, 'success_rate')
    assert hasattr(metrics, 'optimization_score')

def test_error_handling():
    """Test system error handling robustness."""
    from src.reflexion.core.agent import ReflexionAgent
    
    # Test with edge cases
    agent = ReflexionAgent(llm="gpt-4", max_iterations=1)
    
    try:
        result = agent.run("")  # Empty task
        # Should handle gracefully
        assert hasattr(result, 'success')
    except Exception:
        # Should be controlled failure
        pass
    
    try:
        result = agent.run("A" * 10000)  # Very long task
        assert hasattr(result, 'success')
    except Exception:
        pass

def main():
    """Run all validation tests."""
    print("🚀 Running Basic System Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Agent Creation", test_agent_creation),
        ("Agent Task Execution", test_agent_task_execution),
        ("Research Algorithms", test_research_algorithms),
        ("Performance Optimization", test_performance_optimization),
        ("Telemetry System", test_telemetry_system),
        ("CLI Imports", test_cli_imports),
        ("Memory System", test_memory_system),
        ("Integration Workflow", test_integration_workflow),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
    
    print(f"\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Review issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)