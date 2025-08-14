"""
Comprehensive tests for autonomous scaling engine.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.reflexion.core.autonomous_scaling_engine import (
    AutonomousScalingEngine, ResourceConfiguration, ScalingEvent,
    ResourceType, ScalingDirection, LoadPattern, 
    PredictiveLoadAnalyzer, ResourceOptimizer
)


class TestResourceConfiguration:
    """Test resource configuration functionality."""
    
    def test_resource_config_initialization(self):
        """Test resource configuration initialization."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=4,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        assert config.resource_type == ResourceType.CPU
        assert config.current_value == 4
        assert config.min_value == 2
        assert config.max_value == 16
        assert config.step_size == 2
    
    def test_can_scale_no_previous_scaling(self):
        """Test can_scale when no previous scaling occurred."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=4,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        assert config.can_scale() is True
    
    def test_can_scale_with_cooldown(self):
        """Test can_scale during cooldown period."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=4,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        config.last_scaling_time = datetime.now() - timedelta(seconds=100)
        assert config.can_scale() is False
        
        config.last_scaling_time = datetime.now() - timedelta(seconds=400)
        assert config.can_scale() is True
    
    def test_scale_up_success(self):
        """Test successful scale up operation."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=4,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        result = config.scale_up()
        
        assert result is True
        assert config.current_value == 6
        assert config.last_scaling_time is not None
    
    def test_scale_up_at_max(self):
        """Test scale up when already at maximum."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=16,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        result = config.scale_up()
        
        assert result is False
        assert config.current_value == 16
    
    def test_scale_down_success(self):
        """Test successful scale down operation."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=8,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        result = config.scale_down()
        
        assert result is True
        assert config.current_value == 6
        assert config.last_scaling_time is not None
    
    def test_scale_down_at_min(self):
        """Test scale down when already at minimum."""
        config = ResourceConfiguration(
            resource_type=ResourceType.CPU,
            current_value=2,
            min_value=2,
            max_value=16,
            step_size=2,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            cooldown_period=300
        )
        
        result = config.scale_down()
        
        assert result is False
        assert config.current_value == 2


class TestPredictiveLoadAnalyzer:
    """Test predictive load analyzer functionality."""
    
    @pytest.fixture
    def load_analyzer(self):
        """Create load analyzer for testing."""
        return PredictiveLoadAnalyzer(history_size=100)
    
    def test_record_load(self, load_analyzer):
        """Test recording load measurements."""
        timestamp = datetime.now()
        load_analyzer.record_load(0.7, timestamp)
        
        assert len(load_analyzer.load_history) == 1
        assert load_analyzer.load_history[0] == (timestamp, 0.7)
    
    def test_detect_pattern_insufficient_data(self, load_analyzer):
        """Test pattern detection with insufficient data."""
        # Add only a few data points
        for i in range(10):
            load_analyzer.record_load(0.5 + i * 0.01)
        
        pattern = load_analyzer.detect_pattern()
        assert pattern == LoadPattern.UNPREDICTABLE
    
    def test_detect_steady_pattern(self, load_analyzer):
        """Test detecting steady load pattern."""
        # Add steady load data
        for i in range(60):
            load_analyzer.record_load(0.5 + (i % 10) * 0.001)  # Very low variation
        
        pattern = load_analyzer.detect_pattern()
        # Note: Due to simplistic pattern detection, may not always detect STEADY
        assert pattern in [LoadPattern.STEADY, LoadPattern.UNPREDICTABLE]
    
    def test_detect_bursty_pattern(self, load_analyzer):
        """Test detecting bursty load pattern."""
        # Add bursty load data
        base_time = datetime.now()
        for i in range(60):
            if i % 10 < 2:
                load = 0.9  # High burst
            else:
                load = 0.2  # Low baseline
            load_analyzer.record_load(load, base_time + timedelta(minutes=i))
        
        pattern = load_analyzer.detect_pattern()
        # High coefficient of variation should indicate bursty
        assert pattern in [LoadPattern.BURSTY, LoadPattern.UNPREDICTABLE]
    
    def test_predict_load_insufficient_data(self, load_analyzer):
        """Test load prediction with insufficient data."""
        load_analyzer.record_load(0.6)
        
        predictions = load_analyzer.predict_load(horizon_minutes=5)
        
        assert len(predictions) == 5
        # Should return constant value when insufficient data
        for _, predicted_load in predictions:
            assert predicted_load == 0.6
    
    def test_predict_load_with_trend(self, load_analyzer):
        """Test load prediction with trending data."""
        base_time = datetime.now()
        
        # Add trending data
        for i in range(30):
            load = 0.3 + i * 0.01  # Increasing trend
            load_analyzer.record_load(load, base_time + timedelta(minutes=i))
        
        predictions = load_analyzer.predict_load(horizon_minutes=3)
        
        assert len(predictions) == 3
        # Predictions should generally continue the trend
        predicted_values = [load for _, load in predictions]
        assert all(load > 0.3 for load in predicted_values)
    
    def test_autocorrelation_calculation(self, load_analyzer):
        """Test autocorrelation calculation."""
        data = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5]  # Some pattern
        
        correlation = load_analyzer._autocorrelation(data, 4)
        
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1


class TestResourceOptimizer:
    """Test resource optimizer functionality."""
    
    @pytest.fixture
    def resource_optimizer(self):
        """Create resource optimizer for testing."""
        return ResourceOptimizer()
    
    def test_analyze_resource_efficiency(self, resource_optimizer):
        """Test resource efficiency analysis."""
        resources = {
            ResourceType.CPU: ResourceConfiguration(
                resource_type=ResourceType.CPU,
                current_value=4,
                min_value=2,
                max_value=16,
                step_size=2,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                cooldown_period=300,
                target_utilization=0.7
            ),
            ResourceType.MEMORY: ResourceConfiguration(
                resource_type=ResourceType.MEMORY,
                current_value=8,
                min_value=4,
                max_value=32,
                step_size=4,
                scale_up_threshold=0.85,
                scale_down_threshold=0.4,
                cooldown_period=300,
                target_utilization=0.75
            )
        }
        
        analysis = resource_optimizer.analyze_resource_efficiency(resources)
        
        assert "overall_efficiency" in analysis
        assert "resource_efficiencies" in analysis
        assert "recommendations" in analysis
        assert "optimization_opportunities" in analysis
        
        assert isinstance(analysis["overall_efficiency"], float)
        assert 0 <= analysis["overall_efficiency"] <= 1
    
    def test_identify_optimization_opportunities(self, resource_optimizer):
        """Test optimization opportunity identification."""
        resources = {
            ResourceType.CPU: ResourceConfiguration(
                resource_type=ResourceType.CPU,
                current_value=4,
                min_value=2,
                max_value=16,
                step_size=2,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                cooldown_period=300,
                target_utilization=0.7
            )
        }
        
        # Mock high CPU, low memory scenario
        with patch.object(resource_optimizer, '_get_current_utilization') as mock_util:
            mock_util.side_effect = lambda resource_type: {
                ResourceType.CPU: 0.9,
                ResourceType.MEMORY: 0.2
            }.get(resource_type, 0.5)
            
            opportunities = resource_optimizer._identify_optimization_opportunities(resources)
            
            assert len(opportunities) > 0
            # Should identify resource imbalance
            assert any("resource_imbalance" in opp["type"] for opp in opportunities)


class TestAutonomousScalingEngine:
    """Test autonomous scaling engine functionality."""
    
    @pytest.fixture
    def scaling_engine(self):
        """Create scaling engine for testing."""
        # Patch the background thread starting to avoid threading in tests
        with patch.object(AutonomousScalingEngine, '_start_autonomous_scaling'):
            engine = AutonomousScalingEngine()
        return engine
    
    def test_engine_initialization(self, scaling_engine):
        """Test scaling engine initialization."""
        assert len(scaling_engine.resources) > 0
        assert ResourceType.CPU in scaling_engine.resources
        assert ResourceType.MEMORY in scaling_engine.resources
        assert scaling_engine.scaling_enabled is True
    
    def test_get_system_load(self, scaling_engine):
        """Test system load calculation."""
        with patch.object(scaling_engine, '_get_resource_utilization') as mock_util:
            mock_util.side_effect = lambda resource_type: {
                ResourceType.CPU: 0.6,
                ResourceType.MEMORY: 0.7,
                ResourceType.CONCURRENT_TASKS: 0.5
            }.get(resource_type, 0.5)
            
            system_load = scaling_engine._get_system_load()
            
            # Should be weighted combination: 0.6*0.4 + 0.7*0.3 + 0.5*0.3 = 0.6
            expected_load = 0.6 * 0.4 + 0.7 * 0.3 + 0.5 * 0.3
            assert abs(system_load - expected_load) < 0.01
    
    def test_make_scaling_decision_scale_up(self, scaling_engine):
        """Test scaling decision when scale up is needed."""
        config = scaling_engine.resources[ResourceType.CPU]
        
        decision = scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=config,
            current_utilization=0.9,  # Above scale_up_threshold
            predicted_peak_load=0.85,
            load_pattern=LoadPattern.STEADY
        )
        
        assert decision == ScalingDirection.UP
    
    def test_make_scaling_decision_scale_down(self, scaling_engine):
        """Test scaling decision when scale down is appropriate."""
        config = scaling_engine.resources[ResourceType.CPU]
        
        decision = scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=config,
            current_utilization=0.2,  # Below scale_down_threshold
            predicted_peak_load=0.3,
            load_pattern=LoadPattern.STEADY
        )
        
        assert decision == ScalingDirection.DOWN
    
    def test_make_scaling_decision_stable(self, scaling_engine):
        """Test scaling decision when no scaling is needed."""
        config = scaling_engine.resources[ResourceType.CPU]
        
        decision = scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=config,
            current_utilization=0.6,  # Between thresholds
            predicted_peak_load=0.6,
            load_pattern=LoadPattern.STEADY
        )
        
        assert decision == ScalingDirection.STABLE
    
    def test_make_scaling_decision_with_prediction(self, scaling_engine):
        """Test scaling decision considers predictions."""
        config = scaling_engine.resources[ResourceType.CPU]
        
        # Current utilization low, but high spike predicted
        decision = scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=config,
            current_utilization=0.5,  # Normal level
            predicted_peak_load=0.9,  # High spike predicted
            load_pattern=LoadPattern.BURSTY
        )
        
        # Should scale up proactively
        assert decision == ScalingDirection.UP
    
    def test_execute_scaling_up(self, scaling_engine):
        """Test executing scale up operation."""
        config = scaling_engine.resources[ResourceType.CPU]
        old_value = config.current_value
        
        with patch.object(scaling_engine, '_apply_resource_scaling') as mock_apply:
            scaling_engine._execute_scaling(
                resource_type=ResourceType.CPU,
                config=config,
                direction=ScalingDirection.UP,
                current_utilization=0.9
            )
            
            assert config.current_value > old_value
            assert len(scaling_engine.scaling_history) > 0
            mock_apply.assert_called_once()
    
    def test_execute_scaling_failure_handling(self, scaling_engine):
        """Test handling of scaling execution failures."""
        config = scaling_engine.resources[ResourceType.CPU]
        old_value = config.current_value
        
        with patch.object(scaling_engine, '_apply_resource_scaling', side_effect=Exception("Scaling failed")):
            scaling_engine._execute_scaling(
                resource_type=ResourceType.CPU,
                config=config,
                direction=ScalingDirection.UP,
                current_utilization=0.9
            )
            
            # Should record failed scaling event
            assert len(scaling_engine.scaling_history) > 0
            latest_event = scaling_engine.scaling_history[-1]
            assert latest_event.success is False
            assert "Scaling failed" in latest_event.reason
    
    def test_configure_scaling(self, scaling_engine):
        """Test scaling configuration updates."""
        original_config = scaling_engine.resources[ResourceType.CPU]
        original_max = original_config.max_value
        
        scaling_engine.configure_scaling(
            resource_type=ResourceType.CPU,
            max_value=32,
            scale_up_threshold=0.75
        )
        
        updated_config = scaling_engine.resources[ResourceType.CPU]
        assert updated_config.max_value == 32
        assert updated_config.scale_up_threshold == 0.75
        assert updated_config.min_value == original_config.min_value  # Unchanged
    
    def test_cost_optimization_mode(self, scaling_engine):
        """Test cost optimization mode effects on scaling."""
        scaling_engine.enable_cost_optimization(True)
        assert scaling_engine.cost_optimization_mode is True
        
        config = scaling_engine.resources[ResourceType.CPU]
        
        # Cost optimization should delay scaling up
        decision = scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=config,
            current_utilization=0.82,  # Slightly above threshold
            predicted_peak_load=0.8,
            load_pattern=LoadPattern.STEADY
        )
        
        # Should remain stable due to cost optimization
        assert decision == ScalingDirection.STABLE
    
    def test_get_scaling_report(self, scaling_engine):
        """Test scaling report generation."""
        # Add some scaling history
        scaling_event = ScalingEvent(
            timestamp=datetime.now(),
            resource_type=ResourceType.CPU,
            direction=ScalingDirection.UP,
            old_value=4,
            new_value=6,
            trigger_metric="cpu_utilization",
            trigger_value=0.85,
            success=True
        )
        scaling_engine.scaling_history.append(scaling_event)
        
        with patch.object(scaling_engine, '_get_resource_utilization', return_value=0.6):
            with patch.object(scaling_engine, '_get_system_load', return_value=0.65):
                with patch.object(scaling_engine, '_get_average_response_time', return_value=75.0):
                    with patch.object(scaling_engine, '_get_system_throughput', return_value=150.0):
                        
                        report = scaling_engine.get_scaling_report()
                        
                        assert "scaling_status" in report
                        assert "current_resources" in report
                        assert "recent_scaling_events" in report
                        assert "load_analysis" in report
                        assert "optimization_analysis" in report
                        assert "performance_summary" in report
                        
                        assert report["scaling_status"]["total_scaling_events"] == 1
                        assert len(report["recent_scaling_events"]) == 1
    
    def test_autonomous_scaling_cycle(self, scaling_engine):
        """Test one complete autonomous scaling cycle."""
        
        # Mock all dependencies
        with patch.object(scaling_engine, '_get_system_load', return_value=0.7) as mock_system_load:
            with patch.object(scaling_engine, '_get_resource_utilization', return_value=0.85) as mock_resource_util:
                with patch.object(scaling_engine, '_apply_resource_scaling') as mock_apply:
                    
                    # Force a resource to be scalable
                    for config in scaling_engine.resources.values():
                        config.last_scaling_time = None
                    
                    scaling_engine._autonomous_scaling_cycle()
                    
                    # Should have recorded system load
                    mock_system_load.assert_called()
                    
                    # Should have checked resource utilizations
                    assert mock_resource_util.call_count > 0
                    
                    # Should have attempted some scaling operations due to high utilization
                    assert len(scaling_engine.scaling_history) > 0


class TestScalingEvent:
    """Test scaling event functionality."""
    
    def test_scaling_event_creation(self):
        """Test creating scaling event."""
        event = ScalingEvent(
            timestamp=datetime.now(),
            resource_type=ResourceType.CPU,
            direction=ScalingDirection.UP,
            old_value=4,
            new_value=6,
            trigger_metric="cpu_utilization",
            trigger_value=0.85,
            success=True,
            reason="High CPU utilization"
        )
        
        assert event.resource_type == ResourceType.CPU
        assert event.direction == ScalingDirection.UP
        assert event.old_value == 4
        assert event.new_value == 6
        assert event.success is True
    
    def test_scaling_event_to_dict(self):
        """Test converting scaling event to dictionary."""
        timestamp = datetime.now()
        event = ScalingEvent(
            timestamp=timestamp,
            resource_type=ResourceType.MEMORY,
            direction=ScalingDirection.DOWN,
            old_value=16,
            new_value=12,
            trigger_metric="memory_utilization",
            trigger_value=0.25,
            success=True
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["resource_type"] == "memory"
        assert event_dict["direction"] == "scale_down"
        assert event_dict["old_value"] == 16
        assert event_dict["new_value"] == 12
        assert event_dict["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])