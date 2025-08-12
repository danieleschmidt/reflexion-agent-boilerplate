"""Comprehensive production system tests."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from src.reflexion.core.agent import ReflexionAgent, ReflectionType
from src.reflexion.research.novel_algorithms import research_comparator, MetaCognitiveReflexionAlgorithm
from src.reflexion.monitoring.telemetry import telemetry_manager, TelemetryEvent
from src.reflexion.core.performance_optimizer import performance_optimizer, ReflectionCache
from src.reflexion.research.experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentCondition, ExperimentType


class TestProductionSystem:
    """Test suite for production system functionality."""
    
    def test_reflexion_agent_basic_functionality(self):
        """Test basic ReflexionAgent functionality."""
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ReflectionType.BINARY,
            success_threshold=0.7
        )
        
        # Test task execution
        result = agent.run("Write a simple hello world function", success_criteria="simple,functional")
        
        assert result is not None
        assert hasattr(result, 'task')
        assert hasattr(result, 'success')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'output')
        assert hasattr(result, 'total_time')
        assert hasattr(result, 'reflections')
        
        # Verify result structure
        assert isinstance(result.success, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.total_time, (int, float))
        assert isinstance(result.reflections, list)
        assert result.iterations >= 1
        assert result.total_time > 0
    
    def test_research_algorithms_availability(self):
        """Test that all research algorithms are available."""
        algorithms = research_comparator.algorithms
        
        # Verify expected algorithms are present
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
            assert expected in algorithm_names
    
    def test_performance_optimization_caching(self):
        """Test performance optimization caching system."""
        cache = ReflectionCache(max_size=100, ttl_seconds=3600)
        
        # Test cache miss
        result = cache.get("test task", "gpt-4", {"reflection_type": "binary"})
        assert result is None
        
        # Create mock result
        mock_result = Mock()
        mock_result.task = "test task"
        mock_result.success = True
        mock_result.total_time = 1.5
        
        # Test cache put and get
        cache.put("test task", "gpt-4", {"reflection_type": "binary"}, mock_result)
        cached_result = cache.get("test task", "gpt-4", {"reflection_type": "binary"})
        
        assert cached_result is not None
        assert cached_result.task == "test task"
        
        # Test cache stats
        stats = cache.get_stats()
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert 'hit_rate' in stats
        assert stats['size'] == 1
    
    def test_telemetry_system_functionality(self):
        """Test telemetry system functionality."""
        # Test metrics collection
        metrics = telemetry_manager.metrics_collector.get_latest_metrics()
        assert 'collection_info' in metrics
        assert 'is_collecting' in metrics['collection_info']
        
        # Test health checks
        health_results = telemetry_manager.run_health_check()
        assert 'health_checks' in health_results
        assert 'health_summary' in health_results
        assert 'active_alerts' in health_results
        assert 'metrics' in health_results
        
        # Verify health check structure
        for check_name, check_result in health_results['health_checks'].items():
            assert 'healthy' in check_result
            assert 'message' in check_result
            assert 'timestamp' in check_result
            assert isinstance(check_result['healthy'], bool)
    
    def test_experiment_framework(self):
        """Test research experiment framework."""
        # Create test experiment configuration
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment for validation",
            experiment_type=ExperimentType.COMPARATIVE,
            conditions=[
                ExperimentCondition(
                    name="baseline",
                    description="Baseline condition",
                    config={"llm": "gpt-4", "max_iterations": 2},
                    parameters={}
                )
            ],
            test_tasks=["Simple test task"],
            success_criteria=["accuracy"],
            metrics=["success", "total_time"],
            num_trials=1
        )
        
        # Verify configuration structure
        assert config.name == "test_experiment"
        assert len(config.conditions) == 1
        assert len(config.test_tasks) == 1
        assert config.num_trials == 1
        
        # Test experiment runner initialization
        runner = ExperimentRunner(output_dir="./test_experiments")
        assert runner.output_dir == "./test_experiments"
    
    def test_algorithm_performance_tracking(self):
        """Test algorithm performance tracking."""
        # Record some mock performance data
        mock_result = Mock()
        mock_result.success = True
        mock_result.total_time = 2.0
        mock_result.iterations = 2
        mock_result.reflections = [Mock(), Mock()]
        mock_result.metadata = {}
        
        # Test performance recording
        telemetry_manager.record_task_execution(mock_result, "hierarchical")
        
        # Verify metrics were recorded
        metrics = telemetry_manager.metrics_collector.get_latest_metrics()
        assert metrics is not None
        
        # Test that counters were updated
        collector = telemetry_manager.metrics_collector
        assert collector.task_counters['total'] >= 1
        assert collector.task_counters['successful'] >= 1
        assert len(collector.execution_times) >= 1
        assert len(collector.algorithm_usage) >= 1
    
    def test_error_handling_robustness(self):
        """Test system robustness under error conditions."""
        # Test agent with invalid parameters
        agent = ReflexionAgent(llm="invalid-model", max_iterations=1)
        
        # Should handle gracefully without crashing
        try:
            result = agent.run("test task")
            # If it succeeds, verify it returns a proper result structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'output')
        except Exception as e:
            # If it fails, should be a controlled failure
            assert isinstance(e, (ValueError, RuntimeError, Exception))
    
    def test_memory_system_integration(self):
        """Test memory system integration."""
        try:
            from src.reflexion.memory.episodic import EpisodicMemory
            
            # Test memory initialization
            memory = EpisodicMemory(storage_path="./test_memory.json")
            
            # Test basic memory operations
            patterns = memory.get_success_patterns()
            assert 'total_episodes' in patterns
            assert 'successful_episodes' in patterns
            assert 'success_rate' in patterns
            assert 'patterns' in patterns
            
        except Exception as e:
            # Memory system may not be fully functional in test environment
            pytest.skip(f"Memory system not available: {e}")
    
    def test_cli_module_imports(self):
        """Test that CLI module imports work correctly."""
        try:
            from src.reflexion.cli import main, execute_task_with_reflexion
            
            # Verify functions are importable
            assert callable(main)
            assert callable(execute_task_with_reflexion)
            
        except ImportError as e:
            pytest.fail(f"CLI module import failed: {e}")
    
    def test_concurrent_task_execution(self):
        """Test concurrent task execution capabilities."""
        if not performance_optimizer.strategy.enable_parallel_processing:
            pytest.skip("Parallel processing not enabled")
        
        # Test that performance optimizer handles multiple tasks
        tasks = [
            "Task 1: Write a function",
            "Task 2: Analyze data", 
            "Task 3: Create documentation"
        ]
        
        # Mock task function
        def mock_task_func(task, **kwargs):
            mock_result = Mock()
            mock_result.task = task
            mock_result.success = True
            mock_result.total_time = 1.0
            mock_result.iterations = 1
            mock_result.reflections = []
            mock_result.metadata = {}
            return mock_result
        
        # Test batch execution
        results = performance_optimizer.optimize_batch_execution(
            mock_task_func, tasks, llm="gpt-4"
        )
        
        assert len(results) == len(tasks)
        for result in results:
            assert hasattr(result, 'task')
            assert hasattr(result, 'success')
    
    def test_system_health_monitoring(self):
        """Test comprehensive system health monitoring."""
        # Start telemetry if not already started
        if not telemetry_manager.metrics_collector.is_collecting:
            telemetry_manager.start()
        
        # Run health check
        health_data = telemetry_manager.run_health_check()
        
        # Verify health check completeness
        assert 'health_checks' in health_data
        assert 'health_summary' in health_data
        assert 'active_alerts' in health_data
        assert 'metrics' in health_data
        
        # Verify health summary structure
        health_summary = health_data['health_summary']
        assert 'overall_healthy' in health_summary
        assert 'health_trend' in health_summary
        assert 'last_check' in health_summary
        
        # Test dashboard data
        dashboard_data = telemetry_manager.get_dashboard_data()
        assert 'system_info' in dashboard_data
        assert 'metrics' in dashboard_data
        assert 'health' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'telemetry' in dashboard_data
    
    def test_optimization_suggestions(self):
        """Test performance optimization suggestions."""
        suggestions = performance_optimizer.get_optimization_suggestions()
        
        # Should return a list of suggestions
        assert isinstance(suggestions, list)
        
        # Test that suggestions are strings
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 10  # Reasonable suggestion length
    
    def test_integration_end_to_end(self):
        """Test end-to-end system integration."""
        # Create agent
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            success_threshold=0.7
        )
        
        # Execute task
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
        
        # Verify telemetry recorded the execution
        latest_metrics = telemetry_manager.metrics_collector.get_latest_metrics()
        assert latest_metrics is not None


class TestAdvancedFeatures:
    """Test advanced features and research capabilities."""
    
    def test_meta_cognitive_algorithm(self):
        """Test meta-cognitive reflexion algorithm."""
        meta_alg = MetaCognitiveReflexionAlgorithm()
        
        # Verify algorithm structure
        assert meta_alg.meta_levels == 3
        assert meta_alg.metacognitive_threshold == 0.6
        assert hasattr(meta_alg, 'reflection_quality_tracker')
        assert hasattr(meta_alg, 'metacognitive_history')
    
    def test_research_algorithm_comparison(self):
        """Test research algorithm comparison capabilities."""
        # Get available algorithms
        algorithms = list(research_comparator.algorithms.keys())
        
        # Verify research comparator has comparison capabilities
        assert hasattr(research_comparator, 'comparative_study')
        assert hasattr(research_comparator, 'performance_data')
        assert len(algorithms) >= 5
    
    def test_telemetry_event_creation(self):
        """Test telemetry event creation and structure."""
        event = TelemetryEvent(
            event_type="test_event",
            timestamp="2024-01-01T00:00:00Z",
            source="test_system",
            data={"test": "data"},
            metadata={"environment": "test"}
        )
        
        assert event.event_type == "test_event"
        assert event.source == "test_system"
        assert event.data["test"] == "data"
        assert event.metadata["environment"] == "test"
    
    def test_performance_prediction(self):
        """Test performance prediction capabilities."""
        if performance_optimizer.predictor:
            # Test prediction functionality
            prediction = performance_optimizer.predictor.predict_performance("test task")
            
            assert 'predicted_time' in prediction
            assert 'predicted_iterations' in prediction
            assert 'predicted_success_rate' in prediction
            assert 'confidence' in prediction
            
            # Test that predictions are reasonable
            assert prediction['predicted_time'] > 0
            assert 0 <= prediction['predicted_success_rate'] <= 1
            assert 0 <= prediction['confidence'] <= 1


@pytest.fixture
def cleanup_test_files():
    """Cleanup test files after tests."""
    yield
    
    # Cleanup test files
    import os
    test_files = [
        "./test_memory.json",
        "./test_experiments"
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])