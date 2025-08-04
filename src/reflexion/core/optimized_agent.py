"""Optimized reflexion agent with performance enhancements."""

from typing import Any, Dict, Optional, List

from .agent import ReflexionAgent
from .optimized_engine import OptimizedReflexionEngine
from .types import ReflectionType, ReflexionResult
from .logging_config import logger


class OptimizedReflexionAgent(ReflexionAgent):
    """High-performance reflexion agent with optimization features."""
    
    def __init__(
        self,
        llm: str,
        max_iterations: int = 3,
        reflection_type: ReflectionType = ReflectionType.BINARY,
        success_threshold: float = 0.8,
        enable_caching: bool = True,
        enable_parallel_reflection: bool = True,
        max_concurrent_tasks: int = 4,
        batch_size: int = 10,
        **kwargs
    ):
        """Initialize optimized reflexion agent.
        
        Args:
            llm: LLM model identifier
            max_iterations: Maximum reflection iterations
            reflection_type: Type of reflection to perform
            success_threshold: Threshold for considering task successful
            enable_caching: Enable result caching for performance
            enable_parallel_reflection: Enable parallel reflection generation
            max_concurrent_tasks: Maximum concurrent tasks for batch processing
            batch_size: Batch size for processing multiple tasks
            **kwargs: Additional configuration options
        """
        # Initialize base agent
        super().__init__(llm, max_iterations, reflection_type, success_threshold, **kwargs)
        
        # Replace engine with optimized version
        self.engine = OptimizedReflexionEngine(
            enable_caching=enable_caching,
            enable_parallel_reflection=enable_parallel_reflection,
            max_concurrent_tasks=max_concurrent_tasks,
            batch_size=batch_size,
            **kwargs
        )
        
        # Optimization settings
        self.enable_caching = enable_caching
        self.enable_parallel_reflection = enable_parallel_reflection
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        logger.info(
            f"OptimizedReflexionAgent initialized - Caching: {enable_caching}, "
            f"Parallel: {enable_parallel_reflection}, Concurrent: {max_concurrent_tasks}"
        )
    
    def run(self, task: str, success_criteria: Optional[str] = None, **kwargs) -> ReflexionResult:
        """Execute task with optimized reflexion enhancement."""
        return self.engine.execute_with_reflexion(
            task=task,
            llm=self.llm,
            max_iterations=self.max_iterations,
            reflection_type=self.reflection_type,
            success_threshold=self.success_threshold,
            success_criteria=success_criteria,
            **kwargs
        )
    
    async def run_batch(
        self,
        tasks: List[str],
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> List[ReflexionResult]:
        """Execute multiple tasks concurrently with optimization."""
        logger.info(f"Running batch of {len(tasks)} tasks with optimized agent")
        
        results = await self.engine.process_batch_async(
            tasks=tasks,
            llm=self.llm,
            max_iterations=self.max_iterations,
            reflection_type=self.reflection_type,
            success_threshold=self.success_threshold,
            success_criteria=success_criteria,
            **kwargs
        )
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        return self.engine.get_performance_summary()
    
    def clear_cache(self):
        """Clear performance cache."""
        from .performance import performance_cache
        performance_cache.clear()
        logger.info("Performance cache cleared")
    
    def optimize_for_throughput(self):
        """Optimize settings for maximum throughput."""
        self.engine.enable_caching = True
        self.engine.enable_parallel_reflection = True
        self.engine.max_concurrent_tasks = 8
        logger.info("Agent optimized for throughput")
    
    def optimize_for_accuracy(self):
        """Optimize settings for maximum accuracy."""
        self.engine.enable_parallel_reflection = True
        # Keep caching but with more conservative settings
        logger.info("Agent optimized for accuracy")
    
    def optimize_for_cost(self):
        """Optimize settings for cost efficiency."""
        self.engine.enable_caching = True
        # Reduce parallel processing to minimize LLM calls
        self.engine.enable_parallel_reflection = False
        logger.info("Agent optimized for cost efficiency")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.engine.cleanup()


class AutoScalingReflexionAgent(OptimizedReflexionAgent):
    """Auto-scaling reflexion agent that adapts to load."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.load_metrics = {
            "active_tasks": 0,
            "queue_size": 0,
            "avg_response_time": 0.0,
            "success_rate": 1.0
        }
        
        # Auto-scaling thresholds
        self.scale_up_threshold = 0.8    # 80% resource utilization
        self.scale_down_threshold = 0.3   # 30% resource utilization
        self.min_workers = 1
        self.max_workers = 16
        
        logger.info("AutoScalingReflexionAgent initialized")
    
    async def run_with_autoscaling(
        self,
        task: str,
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Run task with automatic scaling based on load."""
        import asyncio
        
        # Check current load and adjust resources
        await self._check_and_scale()
        
        # Execute task
        return await asyncio.create_task(
            self._run_task_async(task, success_criteria, **kwargs)
        )
    
    async def _run_task_async(
        self,
        task: str,
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Async wrapper for task execution."""
        import asyncio
        import time
        
        self.load_metrics["active_tasks"] += 1
        start_time = time.time()
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.run(task, success_criteria, **kwargs)
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_load_metrics(execution_time, result.success)
            
            return result
            
        finally:
            self.load_metrics["active_tasks"] -= 1
    
    async def _check_and_scale(self):
        """Check load and scale resources if needed."""
        try:
            from .performance import resource_monitor
            
            # Get current resource usage
            metrics = resource_monitor.collect_metrics()
            cpu_usage = metrics.get("cpu_percent", 0) / 100.0
            memory_usage = metrics.get("memory_percent", 0) / 100.0
            
            # Calculate combined load
            load_factor = max(cpu_usage, memory_usage)
            
            current_workers = self.engine.max_concurrent_tasks
            
            # Scale up if high load
            if load_factor > self.scale_up_threshold and current_workers < self.max_workers:
                new_workers = min(self.max_workers, current_workers * 2)
                self.engine.max_concurrent_tasks = new_workers
                logger.info(f"Scaled up from {current_workers} to {new_workers} workers")
            
            # Scale down if low load
            elif load_factor < self.scale_down_threshold and current_workers > self.min_workers:
                new_workers = max(self.min_workers, current_workers // 2)
                self.engine.max_concurrent_tasks = new_workers
                logger.info(f"Scaled down from {current_workers} to {new_workers} workers")
                
        except Exception as e:
            logger.error(f"Auto-scaling check failed: {str(e)}")
    
    def _update_load_metrics(self, execution_time: float, success: bool):
        """Update load metrics with latest execution."""
        # Update average response time (simple moving average)
        alpha = 0.1  # Smoothing factor
        self.load_metrics["avg_response_time"] = (
            alpha * execution_time + 
            (1 - alpha) * self.load_metrics["avg_response_time"]
        )
        
        # Update success rate
        self.load_metrics["success_rate"] = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * self.load_metrics["success_rate"]
        )
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            "current_workers": self.engine.max_concurrent_tasks,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "load_metrics": self.load_metrics.copy(),
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold
        }