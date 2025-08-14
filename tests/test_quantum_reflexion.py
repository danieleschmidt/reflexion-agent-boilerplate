"""
Comprehensive tests for quantum-inspired reflexion agents.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.reflexion.core.quantum_reflexion_agent import (
    QuantumReflexionAgent, QuantumSuperposition, QuantumReflexionMetrics
)
from src.reflexion.core.types import ReflectionType, ReflexionResult
from src.reflexion.research.novel_algorithms import ReflexionState


class TestQuantumSuperposition:
    """Test quantum superposition functionality."""
    
    def test_superposition_creation(self):
        """Test creating quantum superposition."""
        states = [
            {"algorithm": "test1", "performance": 0.8},
            {"algorithm": "test2", "performance": 0.9}
        ]
        amplitudes = [0.6, 0.8]
        
        superposition = QuantumSuperposition(states, amplitudes)
        
        assert len(superposition.states) == 2
        assert len(superposition.amplitudes) == 2
        assert superposition.measurement_count == 0
    
    def test_entanglement_addition(self):
        """Test adding quantum entanglement."""
        states = [{"test": 1}, {"test": 2}]
        amplitudes = [0.7, 0.7]
        
        superposition = QuantumSuperposition(states, amplitudes)
        superposition.add_entanglement(0, 1, 0.5)
        
        assert len(superposition.entangled_pairs) == 1
        assert superposition.entangled_pairs[0] == (0, 1, 0.5)
    
    def test_superposition_collapse(self):
        """Test quantum superposition collapse."""
        states = [
            {"algorithm": "test1", "value": 10},
            {"algorithm": "test2", "value": 20}
        ]
        amplitudes = [0.8, 0.6]
        
        superposition = QuantumSuperposition(states, amplitudes)
        result = superposition.collapse()
        
        assert "selected_state" in result
        assert "measurement_probability" in result
        assert "quantum_coherence" in result
        assert superposition.measurement_count == 1
    
    def test_entanglement_effects_on_collapse(self):
        """Test that entanglement affects collapse probabilities."""
        states = [{"value": 1}, {"value": 2}, {"value": 3}]
        amplitudes = [0.5, 0.5, 0.5]
        
        superposition = QuantumSuperposition(states, amplitudes)
        superposition.add_entanglement(0, 1, 0.8)
        
        result = superposition.collapse()
        
        assert result["entanglement_effects"] is True
        assert "selected_state" in result


class TestQuantumReflexionMetrics:
    """Test quantum reflexion metrics calculation."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = QuantumReflexionMetrics()
        
        assert metrics.superposition_coherence == 0.0
        assert metrics.entanglement_strength == 0.0
        assert metrics.quantum_advantage == 0.0
    
    def test_quantum_score_calculation(self):
        """Test quantum score calculation."""
        metrics = QuantumReflexionMetrics(
            superposition_coherence=0.8,
            entanglement_strength=0.7,
            quantum_advantage=0.9,
            measurement_efficiency=0.6,
            decoherence_resistance=0.8,
            uncertainty_reduction=0.7
        )
        
        score = metrics.calculate_quantum_score()
        
        # Score should be weighted combination
        expected_score = (0.8 * 0.2 + 0.7 * 0.15 + 0.9 * 0.25 + 
                         0.6 * 0.15 + 0.8 * 0.15 + 0.7 * 0.1)
        
        assert abs(score - expected_score) < 0.001


class TestQuantumReflexionAgent:
    """Test quantum reflexion agent functionality."""
    
    @pytest.fixture
    def quantum_agent(self):
        """Create quantum reflexion agent for testing."""
        return QuantumReflexionAgent(
            llm="test-model",
            max_iterations=2,
            quantum_states=3,
            entanglement_strength=0.5
        )
    
    def test_agent_initialization(self, quantum_agent):
        """Test quantum agent initialization."""
        assert quantum_agent.quantum_states == 3
        assert quantum_agent.entanglement_strength == 0.5
        assert quantum_agent.enable_superposition is True
        assert quantum_agent.current_superposition is None
    
    @pytest.mark.asyncio
    async def test_quantum_run_basic(self, quantum_agent):
        """Test basic quantum run functionality."""
        
        # Mock the LLM provider
        with patch.object(quantum_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Test quantum response"
            
            result = await quantum_agent.quantum_run(
                task="Test quantum task",
                algorithm_ensemble=False
            )
            
            assert isinstance(result, ReflexionResult)
            assert result.task == "Test quantum task"
            assert "quantum_algorithm" in result.metadata
    
    @pytest.mark.asyncio
    async def test_quantum_ensemble_run(self, quantum_agent):
        """Test quantum ensemble run functionality."""
        
        # Mock all algorithm executions
        with patch.multiple(
            quantum_agent,
            quantum_algorithm=Mock(),
            meta_algorithm=Mock(),
            hierarchical_algorithm=Mock(),
            ensemble_algorithm=Mock(),
            contrastive_algorithm=Mock()
        ) as mocks:
            
            # Configure mock returns
            for algorithm_mock in mocks.values():
                algorithm_mock.execute = AsyncMock(return_value=(
                    True,  # success
                    ReflexionState(
                        iteration=1,
                        task="test",
                        current_output="test output",
                        historical_outputs=[],
                        success_scores=[],
                        reflections=[],
                        meta_reflections=[]
                    )
                ))
            
            with patch.object(quantum_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = "Enhanced quantum output"
                
                result = await quantum_agent.quantum_run(
                    task="Test ensemble task",
                    algorithm_ensemble=True
                )
                
                assert isinstance(result, ReflexionResult)
                assert result.metadata["quantum_enhanced"] is True
                assert "quantum_metrics" in result.metadata
    
    @pytest.mark.asyncio
    async def test_superposition_creation(self, quantum_agent):
        """Test quantum superposition creation."""
        
        # Mock algorithm executions
        test_state = ReflexionState(
            iteration=1,
            task="test",
            current_output="test output",
            historical_outputs=[],
            success_scores=[],
            reflections=[],
            meta_reflections=[]
        )
        
        with patch.multiple(
            quantum_agent,
            _run_quantum_algorithm=AsyncMock(return_value=(True, test_state))
        ):
            
            result = await quantum_agent._execute_quantum_superposition(test_state)
            
            assert result["superposition_created"] is True
            assert len(result["superposition_states"]) > 0
            assert "quantum_amplitudes" in result
    
    def test_performance_score_calculation(self, quantum_agent):
        """Test performance score calculation."""
        
        state = ReflexionState(
            iteration=1,
            task="test",
            current_output="A comprehensive test output with substantial content",
            historical_outputs=["shorter output"],
            success_scores=[],
            reflections=[],
            meta_reflections=[{"integrated_approach": ["improvement1", "improvement2"]}]
        )
        
        score = quantum_agent._calculate_performance_score(state, True)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be above baseline for successful state
    
    def test_get_quantum_performance_report(self, quantum_agent):
        """Test quantum performance report generation."""
        
        # Add some history
        quantum_agent.quantum_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": QuantumReflexionMetrics(quantum_advantage=0.5),
            "quantum_score": 0.7
        })
        
        report = quantum_agent.get_quantum_performance_report()
        
        assert "quantum_performance_summary" in report
        assert "current_quantum_metrics" in report
        assert "quantum_algorithm_distribution" in report
        assert report["quantum_performance_summary"]["total_quantum_executions"] == 1
    
    @pytest.mark.asyncio
    async def test_quantum_benchmark(self, quantum_agent):
        """Test quantum benchmarking functionality."""
        
        benchmark_tasks = ["task1", "task2"]
        
        # Mock quantum_run method
        with patch.object(quantum_agent, 'quantum_run', new_callable=AsyncMock) as mock_quantum_run:
            mock_quantum_run.return_value = ReflexionResult(
                task="test",
                output="test output",
                success=True,
                iterations=1,
                reflections=[],
                total_time=1.0,
                metadata={
                    "quantum_metrics": {
                        "quantum_score": 0.8
                    }
                }
            )
            
            # Mock classical agent
            with patch('src.reflexion.core.quantum_reflexion_agent.ReflexionAgent') as mock_classical:
                mock_classical_instance = Mock()
                mock_classical.return_value = mock_classical_instance
                mock_classical_instance.run.return_value = ReflexionResult(
                    task="test",
                    output="classical output",
                    success=True,
                    iterations=2,
                    reflections=[],
                    total_time=2.0,
                    metadata={}
                )
                
                benchmark_results = await quantum_agent.quantum_benchmark(
                    benchmark_tasks=benchmark_tasks,
                    classical_comparison=True
                )
                
                assert "benchmark_metadata" in benchmark_results
                assert "quantum_results" in benchmark_results
                assert "classical_results" in benchmark_results
                assert "comparative_analysis" in benchmark_results
                
                assert len(benchmark_results["quantum_results"]) == 2
                assert len(benchmark_results["classical_results"]) == 2


class TestQuantumAlgorithmIntegration:
    """Integration tests for quantum algorithms."""
    
    @pytest.mark.asyncio
    async def test_algorithm_failure_handling(self):
        """Test handling of algorithm failures."""
        
        quantum_agent = QuantumReflexionAgent(
            llm="test-model",
            quantum_states=2
        )
        
        # Mock one algorithm to fail
        with patch.object(quantum_agent.quantum_algorithm, 'execute', side_effect=Exception("Algorithm failed")):
            with patch.object(quantum_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = "Fallback response"
                
                # Should handle gracefully and use fallback
                result = await quantum_agent._run_single_quantum("test task")
                
                assert isinstance(result, ReflexionResult)
    
    @pytest.mark.asyncio
    async def test_entanglement_correlation_calculation(self):
        """Test quantum entanglement correlation calculation."""
        
        quantum_agent = QuantumReflexionAgent(
            llm="test-model",
            quantum_states=3
        )
        
        quantum_results = {
            "superposition_states": [
                {
                    "algorithm": "algo1",
                    "performance": 0.8,
                    "quantum_phase": 0.0
                },
                {
                    "algorithm": "algo2", 
                    "performance": 0.6,
                    "quantum_phase": 3.14159
                }
            ]
        }
        
        # Create superposition
        quantum_agent.current_superposition = QuantumSuperposition(
            quantum_results["superposition_states"],
            [0.8, 0.6]
        )
        
        entangled_results = await quantum_agent._create_algorithm_entanglement(quantum_results)
        
        assert "entanglement_pairs" in entangled_results
        assert "entanglement_applied" in entangled_results
    
    def test_quantum_metrics_update(self):
        """Test quantum metrics updating."""
        
        quantum_agent = QuantumReflexionAgent(
            llm="test-model",
            quantum_states=2
        )
        
        quantum_results = {
            "superposition_created": True,
            "entanglement_applied": True,
            "entanglement_pairs": [(0, 1, 0.7)]
        }
        
        collapsed_result = {
            "quantum_coherence": 0.8,
            "measurement_probability": 0.9
        }
        
        final_state = ReflexionState(
            iteration=1,
            task="test",
            current_output="final output",
            historical_outputs=["initial output"],
            success_scores=[],
            reflections=[],
            meta_reflections=[{"confidence": 0.8}]
        )
        
        quantum_agent._update_quantum_metrics(quantum_results, collapsed_result, final_state)
        
        assert quantum_agent.quantum_metrics.superposition_coherence > 0
        assert quantum_agent.quantum_metrics.entanglement_strength > 0
        assert len(quantum_agent.quantum_history) > 0


if __name__ == "__main__":
    pytest.main([__file__])