import pytest
from unittest.mock import Mock, patch
import time

from reflexion.core import ReflexionAgent


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for reflexion components."""

    def test_reflexion_agent_creation_time(self, benchmark):
        """Benchmark agent creation time."""
        def create_agent():
            return ReflexionAgent(llm="gpt-4")
        
        result = benchmark(create_agent)
        assert result is not None

    def test_reflection_processing_time(self, benchmark, mock_evaluator):
        """Benchmark reflection processing time."""
        agent = ReflexionAgent(llm="gpt-4", evaluator=mock_evaluator)
        
        def run_reflection():
            evaluation = {"success": False, "score": 0.5, "details": {}}
            with patch.object(agent, '_generate_reflection') as mock_gen:
                mock_gen.return_value = "Test reflection"
                return agent.reflect("test task", "test output", evaluation)
        
        result = benchmark(run_reflection)
        assert result is not None

    def test_memory_storage_performance(self, benchmark, mock_memory):
        """Benchmark memory storage performance."""
        episodes = [
            {"task": f"task_{i}", "outcome": "success", "reflection": f"reflection_{i}"}
            for i in range(100)
        ]
        
        def store_episodes():
            for episode in episodes:
                mock_memory.store(episode)
        
        benchmark(store_episodes)

    def test_memory_recall_performance(self, benchmark, mock_memory):
        """Benchmark memory recall performance."""
        mock_memory.recall.return_value = [
            {"task": f"task_{i}", "similarity": 0.8}
            for i in range(10)
        ]
        
        def recall_memory():
            return mock_memory.recall("test query", k=10)
        
        result = benchmark(recall_memory)
        assert len(result) == 10

    @pytest.mark.slow
    def test_concurrent_agents_performance(self, benchmark):
        """Benchmark performance with multiple concurrent agents."""
        import asyncio
        
        async def run_concurrent_agents():
            agents = [ReflexionAgent(llm="gpt-4") for _ in range(5)]
            
            # Mock execution to avoid actual API calls
            for agent in agents:
                with patch.object(agent, '_execute_task') as mock_exec:
                    mock_exec.return_value = "test output"
            
            # Simulate concurrent execution
            tasks = []
            for i, agent in enumerate(agents):
                task = asyncio.create_task(
                    asyncio.to_thread(agent.run, f"task_{i}")
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # Run the benchmark
        def run_benchmark():
            return asyncio.run(run_concurrent_agents())
        
        results = benchmark(run_benchmark)
        assert len(results) == 5

    def test_large_output_handling(self, benchmark, mock_evaluator):
        """Benchmark handling of large outputs."""
        # Create a large output string
        large_output = "x" * 10000  # 10KB string
        
        agent = ReflexionAgent(llm="gpt-4", evaluator=mock_evaluator)
        
        def process_large_output():
            evaluation = {"success": False, "score": 0.5, "details": {}}
            with patch.object(agent, '_generate_reflection') as mock_gen:
                mock_gen.return_value = "Reflection on large output"
                return agent.reflect("test task", large_output, evaluation)
        
        result = benchmark(process_large_output)
        assert result is not None