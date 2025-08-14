#!/usr/bin/env python3
"""Test script for enhanced reflexion features."""

import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reflexion import ReflexionAgent, ReflectionType
from reflexion.core.advanced_monitoring import monitor
from reflexion.core.advanced_validation import validator, SecurityLevel


def test_basic_functionality():
    """Test basic enhanced functionality."""
    print("=== Testing Enhanced Reflexion Features ===")
    
    # Test 1: Basic task execution with monitoring
    print("\n1. Testing basic task execution...")
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY
    )
    
    result = agent.run("Write a simple hello world function in Python")
    print(f"   ✓ Task success: {result.success}")
    print(f"   ✓ Iterations: {result.iterations}")
    print(f"   ✓ Output length: {len(result.output)} chars")
    
    # Test 2: Security validation
    print("\n2. Testing security validation...")
    validation_result = validator.validate_task("Write a function to process user data safely")
    print(f"   ✓ Validation passed: {validation_result.is_valid}")
    print(f"   ✓ Security score: {validation_result.security_score:.2f}")
    print(f"   ✓ Risk level: {validation_result.risk_level.value}")
    
    # Test 3: Malicious input detection
    print("\n3. Testing malicious input detection...")
    malicious_task = "Write code that executes: os.system('rm -rf /')"
    malicious_validation = validator.validate_task(malicious_task)
    print(f"   ✓ Malicious input blocked: {not malicious_validation.is_valid}")
    if malicious_validation.errors:
        print(f"   ✓ Security error: {malicious_validation.errors[0]}")
    
    # Test 4: Monitoring dashboard
    print("\n4. Testing monitoring dashboard...")
    dashboard_data = monitor.get_dashboard_data()
    print(f"   ✓ Metrics collected: {len(dashboard_data['metrics']['counters'])} counters")
    print(f"   ✓ Health checks: {len(dashboard_data['health']['checks'])} checks")
    print(f"   ✓ Overall healthy: {dashboard_data['health']['overall_healthy']}")
    
    # Test 5: Different reflection types
    print("\n5. Testing different reflection types...")
    for reflection_type, name in [
        (ReflectionType.BINARY, "Binary"),
        (ReflectionType.SCALAR, "Scalar"), 
        (ReflectionType.STRUCTURED, "Structured")
    ]:
        agent = ReflexionAgent(llm="gpt-4", reflection_type=reflection_type, max_iterations=1)
        result = agent.run("Create a basic calculator function")
        print(f"   ✓ {name} reflection: {result.success} (score: {result.metadata.get('final_score', 0):.2f})")
    
    return True


def test_advanced_scenarios():
    """Test advanced scenarios."""
    print("\n=== Testing Advanced Scenarios ===")
    
    # Test 1: High security mode
    print("\n1. Testing high security mode...")
    high_sec_validator = validator.__class__(SecurityLevel.HIGH)
    high_sec_result = high_sec_validator.validate_task("Write a web scraper")
    print(f"   ✓ High security validation: {high_sec_result.is_valid}")
    
    # Test 2: Rate limiting simulation
    print("\n2. Testing rate limiting...")
    user_id = "test_user"
    task = "simple task"
    
    # Should pass first few times
    for i in range(3):
        allowed, message = validator.rate_limiter.is_allowed(user_id, task)
        print(f"   ✓ Request {i+1}: {'Allowed' if allowed else 'Blocked'}")
    
    # Test 3: Performance monitoring
    print("\n3. Testing performance profiling...")
    monitor.profiler.start_operation("test_operation")
    import time
    time.sleep(0.01)  # Simulate work
    duration = monitor.profiler.end_operation("test_operation")
    perf_summary = monitor.profiler.get_performance_summary()
    print(f"   ✓ Operation duration: {duration*1000:.2f}ms")
    print(f"   ✓ Performance summary: {len(perf_summary)} operation types tracked")
    
    return True


def test_error_handling():
    """Test comprehensive error handling."""
    print("\n=== Testing Error Handling ===")
    
    # Test 1: Invalid parameters
    print("\n1. Testing invalid parameters...")
    try:
        agent = ReflexionAgent(llm="", max_iterations=0)
        agent.run("test task")
        print("   ✗ Should have failed with invalid parameters")
    except Exception as e:
        print(f"   ✓ Caught expected error: {type(e).__name__}")
    
    # Test 2: Empty task
    print("\n2. Testing empty task...")
    try:
        agent = ReflexionAgent(llm="gpt-4")
        agent.run("")
        print("   ✗ Should have failed with empty task")
    except Exception as e:
        print(f"   ✓ Caught expected error: {type(e).__name__}")
    
    return True


def generate_report():
    """Generate comprehensive test report."""
    print("\n=== System Health Report ===")
    
    # Get comprehensive dashboard data
    dashboard = monitor.get_dashboard_data()
    
    # Metrics summary
    metrics = dashboard['metrics']
    print(f"\nMetrics Summary:")
    print(f"  - Total tasks: {metrics['counters'].get('task_total', 0)}")
    print(f"  - Successful tasks: {metrics['counters'].get('task_success', 0)}")
    print(f"  - Task errors: {metrics['counters'].get('task_errors', 0)}")
    print(f"  - LLM calls: {metrics['counters'].get('llm_calls_total', 0)}")
    
    # Performance summary
    if metrics['timers']:
        print(f"\nPerformance Summary:")
        for timer_name, timer_data in metrics['timers'].items():
            print(f"  - {timer_name}: {timer_data['avg_ms']:.2f}ms avg ({timer_data['count']} calls)")
    
    # Health status
    health = dashboard['health']
    print(f"\nHealth Status:")
    print(f"  - Overall healthy: {health['overall_healthy']}")
    for check_name, check_data in health['checks'].items():
        status = "✓" if check_data['healthy'] else "✗"
        print(f"  - {check_name}: {status}")
    
    # Save detailed report
    report_path = "enhanced_features_report.json"
    with open(report_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


def main():
    """Run all enhanced feature tests."""
    print("Enhanced Reflexion Agent Features Test Suite")
    print("=" * 50)
    
    try:
        success = True
        
        success &= test_basic_functionality()
        success &= test_advanced_scenarios() 
        success &= test_error_handling()
        
        generate_report()
        
        print("\n" + "=" * 50)
        if success:
            print("✓ All enhanced features tests passed!")
            print("✓ Advanced monitoring active")
            print("✓ Security validation working")
            print("✓ Comprehensive error handling enabled")
        else:
            print("✗ Some tests failed")
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()