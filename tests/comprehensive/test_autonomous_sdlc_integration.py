"""Comprehensive Integration Tests for Autonomous SDLC Implementation.

This module tests the complete autonomous SDLC implementation including
research execution, error recovery, monitoring, and distributed processing.
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reflexion.research.advanced_research_execution import (
    AutonomousResearchOrchestrator,
    ResearchDomain,
    execute_autonomous_research_session
)
from reflexion.core.advanced_error_recovery_v2 import (
    AdvancedErrorRecoverySystem,
    FailureCategory,
    CircuitBreakerConfig,
    error_recovery_system
)
from reflexion.core.comprehensive_monitoring_v2 import (
    ComprehensiveMonitoringSystem,
    MetricType,
    AlertSeverity,
    monitoring_system
)
from reflexion.scaling.distributed_reflexion_engine import (
    DistributedReflexionEngine,
    ProcessingNode,
    NodeStatus,
    TaskPriority,
    DistributionStrategy
)


class TestAutonomousSDLCIntegration(unittest.TestCase):
    """Integration tests for the complete autonomous SDLC system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_output_dir = Path("/tmp/test_autonomous_sdlc")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Configure logging for tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up test files
        if self.test_output_dir.exists():
            import shutil
            shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    def test_research_orchestrator_initialization(self):
        """Test research orchestrator initializes correctly."""
        orchestrator = AutonomousResearchOrchestrator(
            research_directory=str(self.test_output_dir)
        )
        
        # Test basic properties
        self.assertIsInstance(orchestrator.research_directory, Path)
        self.assertEqual(len(orchestrator.active_hypotheses), 0)
        self.assertEqual(len(orchestrator.experiments), 0)
        self.assertEqual(len(orchestrator.results), 0)
        self.assertIsNotNone(orchestrator.research_comparator)
        
        self.logger.info("✅ Research orchestrator initialization test passed")
    
    def test_error_recovery_system(self):
        """Test error recovery system functionality."""
        recovery_system = AdvancedErrorRecoverySystem()
        
        # Test circuit breaker creation
        circuit = recovery_system.get_or_create_circuit_breaker("test_component")
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.name, "test_component")
        
        # Test failure context creation
        test_exception = ValueError("Test error")
        
        async def test_failure_handling():
            context = await recovery_system._create_failure_context(
                test_exception, "test_component", 1.0
            )
            self.assertEqual(context.component, "test_component")
            self.assertEqual(context.error_type, "ValueError")
            self.assertEqual(context.error_message, "Test error")
            self.assertIn(context.category, FailureCategory)
            
            # Test recovery attempt
            recovery_result = await recovery_system._attempt_recovery(context)
            self.assertIsNotNone(recovery_result)
            self.assertIn(recovery_result.strategy_used.value, [s.value for s in recovery_system.recovery_strategies[context.category]])
        
        asyncio.run(test_failure_handling())
        
        # Test statistics
        stats = recovery_system.get_recovery_statistics()
        self.assertIn("overview", stats)
        self.assertIn("performance", stats)
        self.assertIn("strategy_effectiveness", stats)
        
        self.logger.info("✅ Error recovery system test passed")
    
    def test_monitoring_system(self):
        """Test comprehensive monitoring system."""
        monitoring = ComprehensiveMonitoringSystem()
        
        # Test metric creation and recording
        monitoring.create_metric("test.metric", MetricType.GAUGE, description="Test metric")
        monitoring.set_gauge("test.metric", 75.5)
        
        # Test counter
        monitoring.increment_counter("test.counter", 5)
        
        # Test timer
        monitoring.record_timer("test.timer", 1.25)
        
        # Test metrics retrieval
        stats = monitoring.get_metric_statistics("test.metric")
        self.assertIn("mean", stats)
        self.assertEqual(stats["mean"], 75.5)
        
        # Test system health
        health = monitoring.get_system_health()
        self.assertIn("overall_status", health)
        self.assertIn("alerts", health)
        self.assertIn("system_resources", health)
        
        # Test metrics export
        export_data = monitoring.export_metrics("json", 1)
        self.assertIsInstance(export_data, str)
        export_parsed = json.loads(export_data)
        self.assertIn("metrics", export_parsed)
        
        monitoring.stop_monitoring()
        
        self.logger.info("✅ Monitoring system test passed")
    
    def test_distributed_engine_basic(self):
        """Test distributed reflexion engine basic functionality."""
        async def test_engine():
            engine = DistributedReflexionEngine(
                node_id="test_primary",
                max_concurrent_tasks=3,
                distribution_strategy=DistributionStrategy.LEAST_LOADED
            )
            
            # Test node management
            test_node = ProcessingNode(
                node_id="test_worker",
                address="localhost",
                port=8081,
                status=NodeStatus.ACTIVE,
                max_capacity=5,
                capabilities={"reflexion", "analysis"}
            )
            
            engine.add_processing_node(test_node)
            self.assertIn("test_worker", engine.processing_nodes)
            
            # Test task submission and processing
            await engine.start_processing()
            
            task_id = await engine.submit_task(
                task_type="basic_reflexion",
                input_data={"task": "test_reflexion", "complexity": "low"},
                priority=TaskPriority.NORMAL
            )
            
            self.assertIsNotNone(task_id)
            
            # Wait for task completion
            result = await engine.get_task_result(task_id, timeout=10.0)
            self.assertIsNotNone(result)
            self.assertTrue(result.success)
            
            # Test cluster status
            status = engine.get_cluster_status()
            self.assertIn("cluster_overview", status)
            self.assertIn("task_statistics", status)
            self.assertIn("nodes", status)
            
            await engine.stop_processing()
            
            return engine
        
        engine = asyncio.run(test_engine())
        self.assertIsNotNone(engine)
        
        self.logger.info("✅ Distributed engine basic test passed")
    
    def test_system_integration(self):
        """Test integration between all major components."""
        async def integration_test():
            # Initialize all systems
            orchestrator = AutonomousResearchOrchestrator(str(self.test_output_dir))
            recovery = AdvancedErrorRecoverySystem()
            monitoring = ComprehensiveMonitoringSystem()
            engine = DistributedReflexionEngine(node_id="integration_test")
            
            # Test cross-system functionality
            
            # 1. Monitor a distributed task
            with monitoring.profile_operation("integration_test", {"system": "distributed_engine"}):
                await engine.start_processing()
                
                task_id = await engine.submit_task(
                    task_type="research",
                    input_data={"topic": "integration_testing", "depth": "basic"},
                    priority=TaskPriority.HIGH
                )
                
                # Use error recovery protection
                async with recovery.protected_execution(
                    "integration_task", 
                    fallback_result="integration_fallback"
                ):
                    result = await engine.get_task_result(task_id, timeout=15.0)
                    
                    if result:
                        monitoring.increment_counter("integration.tasks.success")
                        monitoring.set_gauge("integration.task.quality", 0.85)
                    else:
                        monitoring.increment_counter("integration.tasks.failed")
                        raise Exception("Task processing failed")
                
                await engine.stop_processing()
            
            # 2. Test research orchestrator with monitoring
            with monitoring.profile_operation("research_orchestration"):
                # Create a simple hypothesis for testing
                hypothesis = orchestrator._generate_domain_hypothesis(
                    ResearchDomain.ALGORITHM_OPTIMIZATION, 1
                )
                
                self.assertIsNotNone(hypothesis)
                orchestrator.active_hypotheses[hypothesis.hypothesis_id] = hypothesis
                
                # Design experiment
                experiment = await orchestrator._design_single_experiment(hypothesis)
                self.assertIsNotNone(experiment)
                orchestrator.experiments[experiment.experiment_id] = experiment
            
            # 3. Check monitoring metrics
            monitoring_stats = monitoring.get_all_metrics_summary()
            self.assertIn("integration.tasks.success", monitoring_stats)
            
            # 4. Check error recovery statistics
            recovery_stats = recovery.get_recovery_statistics()
            self.assertIn("overview", recovery_stats)
            
            # 5. Test system health
            health = await recovery.health_check()
            self.assertIn("overall_health", health)
            self.assertIn("recommendations", health)
            
            monitoring.stop_monitoring()
            
            return {
                "orchestrator": orchestrator,
                "recovery": recovery,
                "monitoring": monitoring,
                "engine": engine
            }
        
        systems = asyncio.run(integration_test())
        self.assertEqual(len(systems), 4)
        
        self.logger.info("✅ System integration test passed")
    
    def test_performance_benchmarks(self):
        """Test system performance under load."""
        async def performance_test():
            engine = DistributedReflexionEngine(
                node_id="perf_test",
                max_concurrent_tasks=10
            )
            
            # Add multiple worker nodes
            for i in range(3):
                worker = ProcessingNode(
                    node_id=f"worker_{i}",
                    address=f"worker{i}.local",
                    port=8080 + i,
                    status=NodeStatus.ACTIVE,
                    max_capacity=8,
                    capabilities={"reflexion", "analysis", "optimization"}
                )
                engine.add_processing_node(worker)
            
            await engine.start_processing()
            
            # Submit multiple tasks concurrently
            start_time = time.time()
            task_ids = []
            
            for i in range(20):  # 20 concurrent tasks
                task_id = await engine.submit_task(
                    task_type="basic_reflexion" if i % 2 == 0 else "advanced_analysis",
                    input_data={"task": f"perf_task_{i}", "iteration": i},
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            
            # Wait for completion
            completed_tasks = 0
            start_wait = time.time()
            
            while completed_tasks < len(task_ids) and (time.time() - start_wait) < 30:
                for task_id in task_ids:
                    result = await engine.get_task_result(task_id, timeout=0.1)
                    if result:
                        completed_tasks += 1
                
                await asyncio.sleep(0.5)
            
            total_time = time.time() - start_time
            
            # Performance assertions
            self.assertLess(submission_time, 2.0, "Task submission should be fast")
            self.assertGreater(completed_tasks, len(task_ids) * 0.7, "At least 70% of tasks should complete")
            self.assertLess(total_time, 25.0, "All tasks should complete within 25 seconds")
            
            # Check cluster status
            status = engine.get_cluster_status()
            success_rate = status["system_health"]["overall_success_rate"]
            self.assertGreater(success_rate, 80.0, "Success rate should be above 80%")
            
            await engine.stop_processing()
            
            return {
                "total_tasks": len(task_ids),
                "completed_tasks": completed_tasks,
                "success_rate": success_rate,
                "total_time": total_time,
                "throughput": completed_tasks / total_time
            }
        
        perf_results = asyncio.run(performance_test())
        
        self.logger.info(f"Performance results: {perf_results}")
        self.assertGreater(perf_results["throughput"], 0.5, "Should process at least 0.5 tasks per second")
        
        self.logger.info("✅ Performance benchmark test passed")
    
    def test_failure_resilience(self):
        """Test system resilience under failure conditions."""
        async def resilience_test():
            recovery = AdvancedErrorRecoverySystem()
            monitoring = ComprehensiveMonitoringSystem()
            
            # Test multiple failure scenarios
            failure_scenarios = [
                ("connection_error", ConnectionError("Network timeout")),
                ("memory_error", MemoryError("Insufficient memory")),
                ("value_error", ValueError("Invalid input data")),
                ("timeout_error", TimeoutError("Operation timed out"))
            ]
            
            recovery_success_count = 0
            
            for scenario_name, exception in failure_scenarios:
                self.logger.info(f"Testing failure scenario: {scenario_name}")
                
                # Test with error recovery protection
                recovered = False
                try:
                    async with recovery.protected_execution(
                        f"test_{scenario_name}",
                        fallback_result=f"fallback_for_{scenario_name}"
                    ) as result:
                        if result == f"fallback_for_{scenario_name}":
                            recovered = True
                            recovery_success_count += 1
                        else:
                            # Simulate the failure
                            raise exception
                
                except Exception as e:
                    # If we get here, recovery failed
                    self.logger.warning(f"Recovery failed for {scenario_name}: {e}")
                
                if recovered:
                    monitoring.increment_counter("test.recovery.success")
                else:
                    monitoring.increment_counter("test.recovery.failure")
            
            # Check recovery statistics
            stats = recovery.get_recovery_statistics()
            total_failures = stats["overview"]["total_failures"]
            
            # We should have attempted recovery for each scenario
            self.assertGreaterEqual(total_failures, len(failure_scenarios))
            
            # At least some recoveries should have succeeded
            self.assertGreater(recovery_success_count, 0, "At least some recovery attempts should succeed")
            
            monitoring.stop_monitoring()
            
            return {
                "scenarios_tested": len(failure_scenarios),
                "successful_recoveries": recovery_success_count,
                "recovery_rate": recovery_success_count / len(failure_scenarios) * 100
            }
        
        resilience_results = asyncio.run(resilience_test())
        
        self.logger.info(f"Resilience results: {resilience_results}")
        self.assertGreater(resilience_results["recovery_rate"], 50.0, "Recovery rate should be above 50%")
        
        self.logger.info("✅ Failure resilience test passed")
    
    def test_scalability(self):
        """Test system scalability and auto-scaling capabilities."""
        async def scalability_test():
            engine = DistributedReflexionEngine(
                node_id="scale_test",
                max_concurrent_tasks=5
            )
            
            await engine.start_processing()
            
            # Test initial cluster size
            initial_status = engine.get_cluster_status()
            initial_nodes = initial_status["cluster_overview"]["total_nodes"]
            self.assertEqual(initial_nodes, 1)  # Just the primary node
            
            # Test scaling up
            await engine.scale_cluster(4)
            
            scaled_up_status = engine.get_cluster_status()
            scaled_up_nodes = scaled_up_status["cluster_overview"]["active_nodes"]
            self.assertGreaterEqual(scaled_up_nodes, 3, "Should scale up to at least 3 active nodes")
            
            # Test task distribution with scaled cluster
            task_ids = []
            for i in range(10):
                task_id = await engine.submit_task(
                    task_type="optimization",
                    input_data={"target": "scalability", "iteration": i},
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            # Wait for some tasks to be distributed
            await asyncio.sleep(3)
            
            # Check task distribution
            final_status = engine.get_cluster_status()
            active_tasks = final_status["task_statistics"]["active_tasks"]
            self.assertGreater(active_tasks, 0, "Tasks should be actively distributed")
            
            # Test scaling down
            await engine.scale_cluster(2)
            
            scaled_down_status = engine.get_cluster_status()
            final_nodes = scaled_down_status["cluster_overview"]["active_nodes"]
            
            await engine.stop_processing()
            
            return {
                "initial_nodes": initial_nodes,
                "max_nodes": scaled_up_nodes,
                "final_nodes": final_nodes,
                "scaling_successful": final_nodes >= 2
            }
        
        scalability_results = asyncio.run(scalability_test())
        
        self.logger.info(f"Scalability results: {scalability_results}")
        self.assertTrue(scalability_results["scaling_successful"], "Scaling operations should succeed")
        
        self.logger.info("✅ Scalability test passed")


class TestSystemQualityGates(unittest.TestCase):
    """Quality gate tests for the autonomous SDLC system."""
    
    def setUp(self):
        """Set up quality gate tests."""
        self.logger = logging.getLogger(__name__)
        
    def test_code_structure_quality(self):
        """Test code structure and organization quality."""
        # Test import paths
        import_tests = [
            ("reflexion.research.advanced_research_execution", "AutonomousResearchOrchestrator"),
            ("reflexion.core.advanced_error_recovery_v2", "AdvancedErrorRecoverySystem"),
            ("reflexion.core.comprehensive_monitoring_v2", "ComprehensiveMonitoringSystem"),
            ("reflexion.scaling.distributed_reflexion_engine", "DistributedReflexionEngine")
        ]
        
        for module_path, class_name in import_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.assertTrue(callable(cls), f"{class_name} should be a callable class")
                
                # Test class has required methods
                if hasattr(cls, '__init__'):
                    init_method = getattr(cls, '__init__')
                    self.assertTrue(callable(init_method))
                    
            except ImportError as e:
                self.fail(f"Failed to import {class_name} from {module_path}: {e}")
        
        self.logger.info("✅ Code structure quality test passed")
    
    def test_configuration_completeness(self):
        """Test that all systems have proper configuration."""
        # Test research orchestrator configuration
        try:
            from reflexion.research.advanced_research_execution import ResearchDomain
            domains = list(ResearchDomain)
            self.assertGreater(len(domains), 3, "Should have multiple research domains")
        except ImportError:
            self.fail("Failed to import ResearchDomain")
        
        # Test error recovery configuration
        try:
            from reflexion.core.advanced_error_recovery_v2 import FailureCategory, RecoveryStrategy
            categories = list(FailureCategory)
            strategies = list(RecoveryStrategy)
            self.assertGreater(len(categories), 5, "Should have multiple failure categories")
            self.assertGreater(len(strategies), 5, "Should have multiple recovery strategies")
        except ImportError:
            self.fail("Failed to import recovery enums")
        
        # Test monitoring configuration
        try:
            from reflexion.core.comprehensive_monitoring_v2 import MetricType, AlertSeverity
            metric_types = list(MetricType)
            alert_severities = list(AlertSeverity)
            self.assertGreater(len(metric_types), 3, "Should have multiple metric types")
            self.assertGreater(len(alert_severities), 3, "Should have multiple alert severities")
        except ImportError:
            self.fail("Failed to import monitoring enums")
        
        self.logger.info("✅ Configuration completeness test passed")
    
    def test_error_handling_coverage(self):
        """Test error handling coverage across systems."""
        # Test that systems handle common error scenarios
        async def error_coverage_test():
            recovery = AdvancedErrorRecoverySystem()
            
            # Test various exception types
            test_exceptions = [
                ValueError("Test value error"),
                ConnectionError("Test connection error"),
                TimeoutError("Test timeout error"),
                RuntimeError("Test runtime error"),
                Exception("Generic test error")
            ]
            
            handled_exceptions = 0
            
            for exc in test_exceptions:
                try:
                    context = await recovery._create_failure_context(exc, "test_component", 1.0)
                    self.assertIsNotNone(context)
                    self.assertIn(context.category, FailureCategory)
                    handled_exceptions += 1
                except Exception as e:
                    self.logger.warning(f"Failed to handle exception {type(exc).__name__}: {e}")
            
            # Should handle most common exceptions
            coverage_rate = handled_exceptions / len(test_exceptions)
            self.assertGreater(coverage_rate, 0.8, "Should handle at least 80% of exception types")
            
            return coverage_rate
        
        from reflexion.core.advanced_error_recovery_v2 import FailureCategory
        coverage = asyncio.run(error_coverage_test())
        
        self.logger.info(f"Error handling coverage: {coverage * 100:.1f}%")
        self.logger.info("✅ Error handling coverage test passed")
    
    def test_performance_requirements(self):
        """Test that systems meet basic performance requirements."""
        async def performance_requirements_test():
            # Test monitoring system responsiveness
            monitoring = ComprehensiveMonitoringSystem()
            
            start_time = time.time()
            for i in range(100):
                monitoring.increment_counter("perf.test.counter")
                monitoring.set_gauge("perf.test.gauge", i)
            
            monitoring_time = time.time() - start_time
            self.assertLess(monitoring_time, 1.0, "Monitoring should handle 100 operations in under 1 second")
            
            # Test distributed engine initialization time
            start_time = time.time()
            engine = DistributedReflexionEngine(node_id="perf_test")
            init_time = time.time() - start_time
            self.assertLess(init_time, 2.0, "Engine initialization should take under 2 seconds")
            
            # Test task submission speed
            await engine.start_processing()
            
            start_time = time.time()
            task_ids = []
            for i in range(10):
                task_id = await engine.submit_task(
                    task_type="basic_reflexion",
                    input_data={"test": f"perf_{i}"}
                )
                task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            self.assertLess(submission_time, 0.5, "Should submit 10 tasks in under 0.5 seconds")
            
            await engine.stop_processing()
            monitoring.stop_monitoring()
            
            return {
                "monitoring_time": monitoring_time,
                "init_time": init_time,
                "submission_time": submission_time
            }
        
        perf_results = asyncio.run(performance_requirements_test())
        
        self.logger.info(f"Performance requirements: {perf_results}")
        self.logger.info("✅ Performance requirements test passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the systems."""
        try:
            import psutil
            process = psutil.Process()
            
            # Get baseline memory
            baseline_memory = process.memory_info().rss
            
            # Create systems and measure memory growth
            systems = []
            
            # Research orchestrator
            from reflexion.research.advanced_research_execution import AutonomousResearchOrchestrator
            orchestrator = AutonomousResearchOrchestrator("/tmp")
            systems.append(orchestrator)
            
            # Error recovery system
            from reflexion.core.advanced_error_recovery_v2 import AdvancedErrorRecoverySystem
            recovery = AdvancedErrorRecoverySystem()
            systems.append(recovery)
            
            # Monitoring system
            from reflexion.core.comprehensive_monitoring_v2 import ComprehensiveMonitoringSystem
            monitoring = ComprehensiveMonitoringSystem()
            systems.append(monitoring)
            
            # Distributed engine
            from reflexion.scaling.distributed_reflexion_engine import DistributedReflexionEngine
            engine = DistributedReflexionEngine(node_id="memory_test")
            systems.append(engine)
            
            # Measure memory after initialization
            after_init_memory = process.memory_info().rss
            memory_growth = after_init_memory - baseline_memory
            
            # Memory growth should be reasonable (less than 100MB)
            max_acceptable_growth = 100 * 1024 * 1024  # 100MB
            self.assertLess(memory_growth, max_acceptable_growth, 
                          f"Memory growth should be under 100MB, got {memory_growth / 1024 / 1024:.1f}MB")
            
            # Cleanup
            monitoring.stop_monitoring()
            
            self.logger.info(f"Memory efficiency: {memory_growth / 1024 / 1024:.1f}MB growth")
            self.logger.info("✅ Memory efficiency test passed")
            
        except ImportError:
            self.logger.warning("psutil not available, skipping memory efficiency test")
            pass


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add integration tests
    integration_tests = unittest.TestLoader().loadTestsFromTestCase(TestAutonomousSDLCIntegration)
    test_suite.addTests(integration_tests)
    
    # Add quality gate tests
    quality_tests = unittest.TestLoader().loadTestsFromTestCase(TestSystemQualityGates)
    test_suite.addTests(quality_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate test report
    test_report = {
        "total_tests": result.testsRun,
        "successful_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        "timestamp": time.time()
    }
    
    # Write test report
    report_path = Path("/tmp/autonomous_sdlc_test_report.json")
    with open(report_path, "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("AUTONOMOUS SDLC TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {test_report['total_tests']}")
    print(f"Successful: {test_report['successful_tests']}")
    print(f"Failed: {test_report['failed_tests']}")
    print(f"Errors: {test_report['error_tests']}")
    print(f"Success Rate: {test_report['success_rate']:.1f}%")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}")
    
    return result.wasSuccessful(), test_report


if __name__ == "__main__":
    success, report = run_comprehensive_tests()
    sys.exit(0 if success else 1)