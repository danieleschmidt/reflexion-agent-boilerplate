"""Tests for advanced research capabilities."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from reflexion.research import (
    ExperimentRunner, ExperimentConfig, ExperimentCondition,
    ResearchAgent, ResearchObjective, ResearchObjectiveType
)
from reflexion.research.experiment_runner import ExperimentType


class TestExperimentRunner:
    """Test experiment runner functionality."""
    
    @pytest.fixture
    def experiment_config(self):
        """Create test experiment configuration."""
        conditions = [
            ExperimentCondition(
                name="baseline",
                description="Baseline condition",
                config={"llm": "gpt-4", "max_iterations": 2},
                parameters={}
            ),
            ExperimentCondition(
                name="experimental", 
                description="Enhanced condition",
                config={"llm": "gpt-4", "max_iterations": 4},
                parameters={}
            )
        ]
        
        return ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            experiment_type=ExperimentType.COMPARATIVE,
            conditions=conditions,
            test_tasks=["Write a function", "Debug code"],
            success_criteria=["correctness"],
            metrics=["success", "iterations"],
            num_trials=2
        )
    
    @pytest.fixture
    def experiment_runner(self, tmp_path):
        """Create experiment runner with temporary output directory."""
        return ExperimentRunner(str(tmp_path))
    
    @pytest.mark.asyncio
    async def test_experiment_runner_initialization(self, experiment_runner):
        """Test experiment runner initialization."""
        assert experiment_runner.output_dir
        assert experiment_runner.logger
    
    @pytest.mark.asyncio
    async def test_run_experiment_basic(self, experiment_runner, experiment_config):
        """Test basic experiment execution."""
        with patch('reflexion.research.experiment_runner.ReflexionAgent') as mock_agent:
            # Mock agent execution
            mock_agent.return_value.run.return_value = Mock(
                success=True,
                iterations=2,
                reflections=[],
                total_time=1.0,
                task="test",
                output="completed",
                metadata={}
            )
            
            result = await experiment_runner.run_experiment(experiment_config)
            
            assert result.config.name == "test_experiment"
            assert len(result.trials) > 0
            assert result.summary_statistics
            assert result.conclusions
    
    def test_calculate_trial_metrics(self, experiment_runner):
        """Test trial metrics calculation."""
        from reflexion.core.types import ReflexionResult
        
        result = ReflexionResult(
            task="test",
            output="completed",
            success=True,
            iterations=3,
            reflections=[],
            total_time=2.5,
            metadata={}
        )
        
        condition = ExperimentCondition(
            name="test",
            description="Test condition",
            config={},
            parameters={}
        )
        
        metrics = experiment_runner._calculate_trial_metrics(result, condition)
        
        assert metrics["success"] == 1.0
        assert metrics["iterations"] == 3.0
        assert metrics["total_time"] == 2.5


class TestResearchAgent:
    """Test research agent functionality."""
    
    @pytest.fixture
    def research_objective(self):
        """Create test research objective."""
        return ResearchObjective(
            name="performance_test",
            description="Test performance optimization",
            objective_type=ResearchObjectiveType.PERFORMANCE_OPTIMIZATION,
            hypotheses=["More iterations improve success"],
            success_metrics=["success_rate"],
            expected_outcomes=["Optimal configuration found"]
        )
    
    @pytest.fixture
    def research_agent(self, tmp_path):
        """Create research agent with temporary output directory."""
        return ResearchAgent(output_dir=str(tmp_path))
    
    def test_research_agent_initialization(self, research_agent):
        """Test research agent initialization."""
        assert research_agent.base_llm == "gpt-4"
        assert research_agent.experiment_runner
        assert research_agent.active_objectives == []
        assert research_agent.findings_database == []
    
    @pytest.mark.asyncio
    async def test_generate_experimental_conditions(self, research_agent, research_objective):
        """Test experimental condition generation."""
        conditions = await research_agent._generate_experimental_conditions(research_objective)
        
        assert len(conditions) > 1  # Should have baseline + experimental conditions
        assert any(c.name == "baseline" for c in conditions)
        
        # Check that performance optimization conditions are generated
        iteration_conditions = [c for c in conditions if "iterations" in c.name]
        assert len(iteration_conditions) > 0
    
    def test_create_performance_optimization_objective(self, research_agent):
        """Test creation of performance optimization objective."""
        objective = research_agent.create_performance_optimization_objective(
            name="test_optimization",
            target_metric="accuracy"
        )
        
        assert objective.name == "test_optimization"
        assert objective.objective_type == ResearchObjectiveType.PERFORMANCE_OPTIMIZATION
        assert len(objective.hypotheses) > 0
        assert "accuracy" in objective.description
    
    def test_create_algorithm_comparison_objective(self, research_agent):
        """Test creation of algorithm comparison objective."""
        algorithms = ["binary_reflection", "scalar_reflection"]
        objective = research_agent.create_algorithm_comparison_objective(
            name="algorithm_test",
            algorithms=algorithms
        )
        
        assert objective.name == "algorithm_test"
        assert objective.objective_type == ResearchObjectiveType.ALGORITHM_COMPARISON
        assert len(objective.hypotheses) == len(algorithms)
    
    def test_get_research_summary(self, research_agent):
        """Test research summary generation."""
        summary = research_agent.get_research_summary()
        
        assert "active_objectives" in summary
        assert "completed_studies" in summary
        assert "total_findings" in summary
        assert "supported_findings" in summary


@pytest.mark.integration
class TestResearchIntegration:
    """Integration tests for research capabilities."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_study(self, tmp_path):
        """Test complete research study workflow."""
        # Create research agent
        research_agent = ResearchAgent(output_dir=str(tmp_path))
        
        # Create simple research objective
        objective = ResearchObjective(
            name="simple_test",
            description="Simple test study", 
            objective_type=ResearchObjectiveType.PERFORMANCE_OPTIMIZATION,
            hypotheses=["Test hypothesis"],
            success_metrics=["success"],
            expected_outcomes=["Test outcome"]
        )
        
        # Mock experiment runner to avoid actual execution
        with patch.object(research_agent.experiment_runner, 'run_experiment') as mock_run:
            mock_run.return_value = Mock(
                config=Mock(conditions=[]),
                trials=[],
                summary_statistics={},
                statistical_tests={},
                conclusions=["Test conclusion"],
                recommendations=["Test recommendation"]
            )
            
            # Conduct research study
            findings = await research_agent.conduct_research_study(
                objective=objective,
                test_scenarios=["Test scenario"],
                num_trials=1
            )
            
            assert len(findings) >= 1
            assert findings[0].objective == objective
            
            # Check that study was recorded
            assert len(research_agent.completed_studies) == 1
            assert research_agent.completed_studies[0]["objective"]["name"] == "simple_test"
    
    def test_research_metrics_calculation(self):
        """Test research-specific metrics calculation."""
        from reflexion.research.experiment_runner import ExperimentTrial
        from reflexion.core.types import ReflexionResult, Reflection
        from datetime import datetime
        
        # Create sample trial with reflections
        result = ReflexionResult(
            task="test task",
            output="test output",
            success=True,
            iterations=2,
            reflections=[
                Reflection(
                    task="test",
                    output="test",
                    success=False,
                    score=0.5,
                    issues=["issue1"],
                    improvements=["improvement1"],
                    confidence=0.7,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=3.0,
            metadata={}
        )
        
        trial = ExperimentTrial(
            condition_name="test_condition",
            task="test task",
            trial_number=1,
            result=result,
            metrics={},
            timestamp=datetime.now().isoformat(),
            duration_seconds=3.5
        )
        
        # Test that reflection metrics are properly calculated
        assert trial.result.iterations == 2
        assert len(trial.result.reflections) == 1
        assert trial.result.reflections[0].confidence == 0.7