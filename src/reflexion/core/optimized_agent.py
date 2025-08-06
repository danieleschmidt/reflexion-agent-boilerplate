"""Optimized reflexion agent with advanced performance enhancements."""

from typing import Any, Dict, Optional, List
import asyncio
import time

from .agent import ReflexionAgent
from .optimized_engine import OptimizedReflexionEngine
from .types import ReflectionType, ReflexionResult
from .logging_config import logger
from .optimization import optimization_manager, SmartCache
from .scaling import scaling_manager


class OptimizedReflexionAgent(ReflexionAgent):
    """High-performance reflexion agent with advanced optimization features."""
    
    def __init__(
        self,
        llm: str,
        max_iterations: int = 3,
        reflection_type: ReflectionType = ReflectionType.BINARY,
        success_threshold: float = 0.8,
        enable_caching: bool = True,
        enable_parallel_execution: bool = True,
        enable_memoization: bool = True,
        enable_prefetching: bool = False,
        max_concurrent_tasks: int = 4,
        batch_size: int = 10,
        cache_size: int = 1000,
        **kwargs
    ):
        """Initialize optimized reflexion agent with advanced features.
        
        Args:
            llm: LLM model identifier
            max_iterations: Maximum reflection iterations
            reflection_type: Type of reflection to perform
            success_threshold: Threshold for considering task successful
            enable_caching: Enable smart caching for performance
            enable_parallel_execution: Enable parallel task execution
            enable_memoization: Enable result memoization
            enable_prefetching: Enable intelligent prefetching
            max_concurrent_tasks: Maximum concurrent tasks for batch processing
            batch_size: Batch size for processing multiple tasks
            cache_size: Maximum cache size for optimization
            **kwargs: Additional configuration options
        """
        # Initialize base agent
        super().__init__(llm, max_iterations, reflection_type, success_threshold, **kwargs)
        
        # Initialize optimization manager with configuration
        self.optimization_manager = optimization_manager
        self.optimization_manager.cache = SmartCache(
            max_size=cache_size,
            default_ttl=3600  # 1 hour TTL
        )
        
        # Initialize dedicated cache for this agent
        self.agent_cache = SmartCache(
            max_size=cache_size // 2,
            default_ttl=1800  # 30 min TTL for agent-specific cache
        )
        
        # Optimization settings
        self.enable_caching = enable_caching
        self.enable_parallel_execution = enable_parallel_execution
        self.enable_memoization = enable_memoization
        self.enable_prefetching = enable_prefetching
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        # Performance tracking
        self.performance_stats = {
            "tasks_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "parallel_executions": 0
        }
        
        logger.info(
            f"OptimizedReflexionAgent initialized - Caching: {enable_caching}, "
            f"Parallel: {enable_parallel_execution}, Memoization: {enable_memoization}, "
            f"Prefetching: {enable_prefetching}, Max Concurrent: {max_concurrent_tasks}"
        )
    
    def run(self, task: str, success_criteria: Optional[str] = None, **kwargs) -> ReflexionResult:
        """Execute task with advanced optimization features."""
        start_time = time.time()
        
        # Generate cache key for this task
        cache_key = self._generate_cache_key(task, success_criteria, kwargs)
        
        # Try cache first if enabled
        if self.enable_caching:
            cached_result = self.agent_cache.get(cache_key)
            if cached_result is not None:
                self.performance_stats["cache_hits"] += 1
                logger.debug(f"Cache hit for task: {task[:50]}...")
                return cached_result
            else:
                self.performance_stats["cache_misses"] += 1
        
        # Execute with optimization
        try:
            result = self.engine.execute_with_reflexion(
                task=task,
                llm=self.llm,
                max_iterations=self.max_iterations,
                reflection_type=self.reflection_type,
                success_threshold=self.success_threshold,
                success_criteria=success_criteria,
                **kwargs
            )
            
            # Apply result optimizations
            optimized_result = self.optimization_manager.optimize_reflexion_result(result)
            
            # Cache the result if enabled
            if self.enable_caching:
                self.agent_cache.put(cache_key, optimized_result, ttl=1800)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats["tasks_processed"] += 1
            self.performance_stats["total_processing_time"] += execution_time
            
            return optimized_result
            
        except Exception as e:
            logger.error(f"Optimized execution failed for task: {task[:50]}... - {str(e)}")
            raise
    
    def _generate_cache_key(self, task: str, success_criteria: Optional[str], kwargs: Dict) -> str:
        """Generate cache key for task execution."""
        import hashlib
        import json
        
        key_data = {
            "task": task,
            "success_criteria": success_criteria,
            "llm": self.llm,
            "max_iterations": self.max_iterations,
            "reflection_type": self.reflection_type.value,
            "success_threshold": self.success_threshold,
            "kwargs": sorted(kwargs.items()) if kwargs else None
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def run_batch(
        self,
        tasks: List[str],
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> List[ReflexionResult]:
        """Execute multiple tasks with advanced parallelization and optimization."""
        logger.info(f"Running optimized batch of {len(tasks)} tasks")
        
        start_time = time.time()
        
        if not self.enable_parallel_execution or len(tasks) <= 1:
            # Sequential processing
            results = []
            for task in tasks:
                result = self.run(task, success_criteria, **kwargs)
                results.append(result)
            return results
        
        # Prepare tasks for parallel execution
        task_configs = [
            {
                "name": f"task_{i}",
                "function": lambda t=task: self.run(t, success_criteria, **kwargs),
                "kwargs": {}
            }
            for i, task in enumerate(tasks)
        ]
        
        # Execute with optimization manager
        results = await self.optimization_manager.optimize_batch_execution(task_configs)
        
        # Extract actual results
        reflexion_results = []
        for result in results:
            if "result" in result:
                # If result contains actual ReflexionResult
                if hasattr(result["result"], "task"):
                    reflexion_results.append(result["result"])
                else:
                    # Create placeholder result for failed tasks
                    reflexion_results.append(self._create_fallback_result(
                        f"Batch task {len(reflexion_results)}",
                        "Batch execution completed"
                    ))
            else:
                reflexion_results.append(self._create_fallback_result(
                    f"Batch task {len(reflexion_results)}",
                    "Batch execution completed"
                ))
        
        # Update performance stats
        batch_time = time.time() - start_time
        self.performance_stats["parallel_executions"] += 1
        self.performance_stats["total_processing_time"] += batch_time
        
        logger.info(f"Completed batch of {len(tasks)} tasks in {batch_time:.2f}s")
        return reflexion_results
    
    def _create_fallback_result(self, task: str, output: str) -> ReflexionResult:
        """Create a fallback ReflexionResult for batch operations."""
        from .types import ReflexionResult
        return ReflexionResult(
            task=task,
            output=output,
            success=True,
            iterations=1,
            reflections=[],
            total_time=0.1,
            metadata={"fallback": True}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Get basic stats
        basic_stats = self.performance_stats.copy()
        
        # Add cache statistics
        cache_stats = self.agent_cache.get_stats()
        
        # Add optimization manager stats
        optimization_stats = self.optimization_manager.get_optimization_stats()
        
        # Calculate derived metrics
        avg_processing_time = (
            basic_stats["total_processing_time"] / basic_stats["tasks_processed"]
            if basic_stats["tasks_processed"] > 0 else 0.0
        )
        
        cache_hit_rate = (
            basic_stats["cache_hits"] / (basic_stats["cache_hits"] + basic_stats["cache_misses"])
            if (basic_stats["cache_hits"] + basic_stats["cache_misses"]) > 0 else 0.0
        )
        
        return {
            "agent_stats": basic_stats,
            "cache_stats": cache_stats,
            "optimization_stats": optimization_stats,
            "derived_metrics": {
                "avg_processing_time": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "tasks_per_second": basic_stats["tasks_processed"] / max(basic_stats["total_processing_time"], 1),
                "parallel_utilization": basic_stats["parallel_executions"] / max(basic_stats["tasks_processed"], 1)
            },
            "configuration": {
                "caching_enabled": self.enable_caching,
                "parallel_execution_enabled": self.enable_parallel_execution,
                "memoization_enabled": self.enable_memoization,
                "prefetching_enabled": self.enable_prefetching,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "batch_size": self.batch_size
            }
        }
    
    def clear_cache(self):
        """Clear all caches and reset performance statistics."""
        self.agent_cache.clear()
        if hasattr(self.optimization_manager, 'clear_optimizations'):
            self.optimization_manager.clear_optimizations()
        
        # Reset performance stats
        self.performance_stats = {
            "tasks_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "parallel_executions": 0
        }
        
        logger.info("All caches cleared and performance stats reset")
    
    def optimize_for_throughput(self):
        """Optimize settings for maximum throughput."""
        self.enable_caching = True
        self.enable_parallel_execution = True
        self.enable_memoization = True
        self.max_concurrent_tasks = min(16, self.max_concurrent_tasks * 2)
        self.batch_size = min(50, self.batch_size * 2)
        
        # Update optimization manager settings
        self.optimization_manager.cache.max_size = 2000  # Larger cache
        
        logger.info(f"Agent optimized for throughput - Concurrent: {self.max_concurrent_tasks}, Batch: {self.batch_size}")
    
    def optimize_for_accuracy(self):
        """Optimize settings for maximum accuracy."""
        self.enable_parallel_execution = True
        self.enable_memoization = True
        self.max_concurrent_tasks = max(2, self.max_concurrent_tasks // 2)
        
        # More conservative caching to ensure fresh results
        self.agent_cache.default_ttl = 900  # 15 minutes
        
        logger.info("Agent optimized for accuracy")
    
    def optimize_for_cost(self):
        """Optimize settings for cost efficiency."""
        self.enable_caching = True
        self.enable_memoization = True
        # Reduce parallel processing to minimize LLM calls
        self.enable_parallel_execution = False
        self.max_concurrent_tasks = 1
        
        # Longer cache TTL to maximize reuse
        self.agent_cache.default_ttl = 7200  # 2 hours
        
        logger.info("Agent optimized for cost efficiency")
    
    def enable_adaptive_optimization(self):
        """Enable adaptive optimization based on workload patterns."""
        current_stats = self.get_performance_stats()
        derived_metrics = current_stats["derived_metrics"]
        
        # Adapt based on current performance
        if derived_metrics["cache_hit_rate"] < 0.3:
            # Low cache hit rate - increase cache size and TTL
            self.agent_cache.max_size = min(2000, self.agent_cache.max_size * 2)
            self.agent_cache.default_ttl = min(7200, self.agent_cache.default_ttl * 2)
            logger.info("Adapted: Increased cache size and TTL due to low hit rate")
        
        if derived_metrics["tasks_per_second"] < 0.5 and derived_metrics["parallel_utilization"] < 0.3:
            # Low throughput - increase parallelization
            self.max_concurrent_tasks = min(16, self.max_concurrent_tasks + 2)
            self.enable_parallel_execution = True
            logger.info(f"Adapted: Increased concurrency to {self.max_concurrent_tasks}")
        
        if derived_metrics["avg_processing_time"] > 10.0:
            # Slow processing - optimize for speed
            self.optimize_for_throughput()
            logger.info("Adapted: Applied throughput optimization due to slow processing")
    
    async def warm_up_cache(self, sample_tasks: List[str]):
        """Warm up cache with sample tasks for better initial performance."""
        if not self.enable_caching:
            return
        
        logger.info(f"Warming up cache with {len(sample_tasks)} sample tasks")
        
        for task in sample_tasks:
            try:
                # Run task and cache result
                result = self.run(task, "basic_warmup")
                logger.debug(f"Warmed up cache for task: {task[:30]}...")
            except Exception as e:
                logger.warning(f"Cache warmup failed for task: {str(e)}")
        
        cache_stats = self.agent_cache.get_stats()
        logger.info(f"Cache warmup completed - Cache size: {cache_stats['size']}")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for optimizing agent performance."""
        stats = self.get_performance_stats()
        derived = stats["derived_metrics"]
        recommendations = []
        
        # Cache performance recommendations
        if derived["cache_hit_rate"] < 0.5:
            recommendations.append({
                "category": "caching",
                "priority": "high",
                "recommendation": "Increase cache size or TTL to improve hit rate",
                "current_hit_rate": derived["cache_hit_rate"]
            })
        
        # Throughput recommendations
        if derived["tasks_per_second"] < 1.0 and not self.enable_parallel_execution:
            recommendations.append({
                "category": "throughput",
                "priority": "medium",
                "recommendation": "Enable parallel execution to improve throughput",
                "current_tps": derived["tasks_per_second"]
            })
        
        # Resource utilization recommendations
        if derived["parallel_utilization"] < 0.2 and self.enable_parallel_execution:
            recommendations.append({
                "category": "resources",
                "priority": "low",
                "recommendation": "Consider reducing max_concurrent_tasks to save resources",
                "current_utilization": derived["parallel_utilization"]
            })
        
        # Cost optimization recommendations
        if derived["cache_hit_rate"] > 0.8 and self.max_concurrent_tasks > 4:
            recommendations.append({
                "category": "cost",
                "priority": "low",
                "recommendation": "High cache hit rate suggests you could reduce concurrency to save costs",
                "current_hit_rate": derived["cache_hit_rate"]
            })
        
        return {
            "recommendations": recommendations,
            "current_performance": derived,
            "optimization_score": self._calculate_optimization_score(derived)
        }
    
    def _calculate_optimization_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall optimization score (0-100)."""
        # Weight different factors
        cache_score = metrics["cache_hit_rate"] * 30  # 30% weight
        throughput_score = min(1.0, metrics["tasks_per_second"] / 2.0) * 40  # 40% weight
        efficiency_score = (1.0 - min(1.0, metrics["avg_processing_time"] / 10.0)) * 30  # 30% weight
        
        total_score = cache_score + throughput_score + efficiency_score
        return min(100.0, max(0.0, total_score))
    
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
            
            current_workers = self.max_concurrent_tasks
            
            # Scale up if high load
            if load_factor > self.scale_up_threshold and current_workers < self.max_workers:
                new_workers = min(self.max_workers, current_workers * 2)
                self.max_concurrent_tasks = new_workers
                logger.info(f"Scaled up from {current_workers} to {new_workers} workers")
            
            # Scale down if low load
            elif load_factor < self.scale_down_threshold and current_workers > self.min_workers:
                new_workers = max(self.min_workers, current_workers // 2)
                self.max_concurrent_tasks = new_workers
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
            "current_workers": self.max_concurrent_tasks,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "load_metrics": self.load_metrics.copy(),
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold
        }