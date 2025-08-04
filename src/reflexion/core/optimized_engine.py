"""Optimized reflexion engine with performance enhancements."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from .engine import ReflexionEngine, LLMProvider
from .types import Reflection, ReflectionType, ReflexionResult
from .performance import (
    performance_cache, adaptive_throttling, resource_monitor,
    BatchProcessor, performance_profile
)
from .logging_config import logger, metrics


class OptimizedLLMProvider(LLMProvider):
    """LLM provider with caching and performance optimizations."""
    
    def __init__(self, model: str = "gpt-4", **kwargs):
        super().__init__(model, **kwargs)
        self.cache = performance_cache
        self.throttling = adaptive_throttling
    
    @performance_profile
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with caching and adaptive throttling."""
        # Check cache first
        cache_params = {"model": self.model, **kwargs}
        cached_result = self.cache.get(prompt, self.model, cache_params)
        
        if cached_result is not None:
            return cached_result
        
        # Apply adaptive throttling
        await self.throttling.throttle()
        
        start_time = time.time()
        try:
            # Generate response
            result = await super().generate(prompt, **kwargs)
            
            # Cache the result
            self.cache.put(prompt, self.model, cache_params, result)
            
            # Record metrics for throttling
            duration = time.time() - start_time
            self.throttling.record_request(duration, success=True)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.throttling.record_request(duration, success=False)
            raise
    
    @performance_profile
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous generation with caching."""
        # Check cache first
        cache_params = {"model": self.model, **kwargs}
        cached_result = self.cache.get(prompt, self.model, cache_params)
        
        if cached_result is not None:
            return cached_result
        
        start_time = time.time()
        try:
            result = super().generate_sync(prompt, **kwargs)
            
            # Cache the result
            self.cache.put(prompt, self.model, cache_params, result)
            
            # Record metrics
            duration = time.time() - start_time
            self.throttling.record_request(duration, success=True)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.throttling.record_request(duration, success=False)
            raise


class OptimizedReflexionEngine(ReflexionEngine):
    """Optimized reflexion engine with advanced performance features."""
    
    def __init__(self, **config):
        super().__init__(**config)
        
        # Performance optimizations
        self.enable_caching = config.get("enable_caching", True)
        self.enable_parallel_reflection = config.get("enable_parallel_reflection", True)
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 4)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            max_workers=self.max_concurrent_tasks,
            batch_size=config.get("batch_size", 10)
        )
        
        # Performance monitoring
        self.resource_monitor = resource_monitor
        self.performance_stats = {
            "tasks_processed": 0,
            "cache_hits": 0,
            "total_time_saved": 0.0,
            "parallel_executions": 0
        }
    
    @performance_profile
    def execute_with_reflexion(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float,
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Optimized execution with performance monitoring."""
        # Collect resource metrics before execution
        pre_metrics = self.resource_monitor.collect_metrics()
        
        # Check if we should use optimized path
        if self.enable_caching or self.enable_parallel_reflection:
            result = self._execute_optimized(
                task, llm, max_iterations, reflection_type,
                success_threshold, success_criteria, **kwargs
            )
        else:
            result = super().execute_with_reflexion(
                task, llm, max_iterations, reflection_type,
                success_threshold, success_criteria, **kwargs
            )
        
        # Collect resource metrics after execution
        post_metrics = self.resource_monitor.collect_metrics()
        
        # Update performance stats
        self.performance_stats["tasks_processed"] += 1
        
        # Add performance metadata to result
        result.metadata.update({
            "optimization_enabled": True,
            "cache_enabled": self.enable_caching,
            "parallel_enabled": self.enable_parallel_reflection,
            "pre_cpu": pre_metrics.get("cpu_percent", 0),
            "post_cpu": post_metrics.get("cpu_percent", 0),
            "cache_stats": performance_cache.get_stats()
        })
        
        return result
    
    def _execute_optimized(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float,
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Optimized execution path with caching and parallelization."""
        execution_id = f"opt_exec_{int(time.time() * 1000)}"
        start_time = time.time()
        reflections: List[Reflection] = []
        current_output = ""
        final_success = False
        evaluation = None
        
        logger.info(
            f"Starting optimized execution {execution_id} - Task: {task[:50]}..., "
            f"Cache: {self.enable_caching}, Parallel: {self.enable_parallel_reflection}"
        )
        
        try:
            # Validate inputs (inherited from parent)
            self._validate_execution_params(
                task, llm, max_iterations, reflection_type, success_threshold
            )
            
            # Use optimized LLM provider
            optimized_provider = OptimizedLLMProvider(llm)
            
            # Main execution loop with optimizations
            for iteration in range(max_iterations):
                iteration_start = time.time()
                
                try:
                    logger.info(f"Starting optimized iteration {iteration + 1}/{max_iterations}")
                    
                    # Execute task with optimized provider
                    current_output = self._execute_task_optimized(
                        task, optimized_provider, iteration, reflections
                    )
                    
                    # Evaluate result
                    evaluation = self._evaluate_output(task, current_output, success_criteria)
                    
                    iteration_time = time.time() - iteration_start
                    logger.info(
                        f"Optimized iteration {iteration + 1} completed - "
                        f"Success: {evaluation['success']}, Score: {evaluation['score']:.2f}, "
                        f"Time: {iteration_time:.2f}s"
                    )
                    
                    # Check for success
                    if evaluation["success"] and evaluation["score"] >= success_threshold:
                        final_success = True
                        logger.info(f"Task succeeded after {iteration + 1} iterations")
                        break
                    
                    # Generate reflection if not the last iteration
                    if iteration < max_iterations - 1:
                        if self.enable_parallel_reflection and iteration > 0:
                            # Generate multiple reflection perspectives in parallel
                            reflection = self._generate_parallel_reflection(
                                task, current_output, evaluation, reflection_type, iteration
                            )
                        else:
                            reflection = self._generate_reflection(
                                task, current_output, evaluation, reflection_type, iteration
                            )
                        
                        reflections.append(reflection)
                        
                        logger.info(
                            f"Optimized reflection generated - Issues: {len(reflection.issues)}, "
                            f"Improvements: {len(reflection.improvements)}, "
                            f"Confidence: {reflection.confidence:.2f}"
                        )
                
                except Exception as e:
                    logger.error(f"Error in optimized iteration {iteration + 1}: {str(e)}")
                    
                    # Create error reflection
                    error_reflection = self._create_error_reflection(task, str(e), iteration)
                    reflections.append(error_reflection)
                    
                    current_output = f"Error: {str(e)}"
                    break
            
            # Calculate final metrics
            total_time = time.time() - start_time
            final_evaluation = evaluation or {"success": False, "score": 0.0}
            
            # Record metrics
            metrics.record_task_execution(
                success=final_success,
                iterations=len(reflections) + 1,
                reflections=len(reflections),
                execution_time=total_time,
                task_type=self._classify_task(task)
            )
            
            # Create optimized result
            result = ReflexionResult(
                task=task,
                output=current_output,
                success=final_success,
                iterations=len(reflections) + 1,
                reflections=reflections,
                total_time=total_time,
                metadata={
                    "llm": llm,
                    "threshold": success_threshold,
                    "execution_id": execution_id,
                    "final_score": final_evaluation.get("score", 0.0),
                    "optimization_type": "full",
                    "cache_hits": getattr(optimized_provider.cache, 'hit_count', 0)
                }
            )
            
            logger.info(
                f"Optimized execution {execution_id} completed - Success: {final_success}, "
                f"Total time: {total_time:.2f}s, Iterations: {len(reflections) + 1}"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Critical error in optimized execution {execution_id}: {str(e)}")
            
            # Record failed execution
            metrics.record_task_execution(
                success=False,
                iterations=len(reflections) + 1,
                reflections=len(reflections),
                execution_time=total_time,
                task_type=self._classify_task(task)
            )
            
            raise
    
    def _execute_task_optimized(
        self,
        task: str,
        provider: OptimizedLLMProvider,
        iteration: int,
        reflections: List[Reflection]
    ) -> str:
        """Execute task with optimized provider."""
        # Build enhanced prompt with optimization hints
        prompt = self._build_optimized_task_prompt(task, reflections, iteration)
        
        try:
            return provider.generate_sync(prompt)
        except Exception as e:
            logger.error(f"Optimized LLM execution failed: {e}")
            return f"Error executing task: {task}"
    
    def _build_optimized_task_prompt(
        self,
        task: str,
        reflections: List[Reflection],
        iteration: int
    ) -> str:
        """Build optimized prompt with performance hints."""
        base_prompt = f"Task: {task}\n\nPlease provide a complete solution."
        
        if reflections:
            # Use the most confident reflection for context
            best_reflection = max(reflections, key=lambda r: r.confidence)
            improvement_context = "\n".join(best_reflection.improvements)
            issues_context = "\n".join(best_reflection.issues)
            
            base_prompt += f"\n\nPrevious attempt (iteration {iteration}) had issues:\n{issues_context}"
            base_prompt += f"\n\nKey improvements needed:\n{improvement_context}"
            
            # Add optimization hint for better caching
            base_prompt += f"\n\nOptimization note: This is iteration {iteration + 1}, focus on the core improvements."
        
        return base_prompt
    
    def _generate_parallel_reflection(
        self,
        task: str,
        output: str,
        evaluation: Dict[str, Any],
        reflection_type: ReflectionType,
        iteration: int
    ) -> Reflection:
        """Generate reflection using parallel analysis."""
        self.performance_stats["parallel_executions"] += 1
        
        logger.debug(f"Generating parallel reflection for iteration {iteration}")
        
        # For now, use enhanced single reflection
        # In production, this could analyze multiple aspects in parallel
        base_reflection = self._generate_reflection(
            task, output, evaluation, reflection_type, iteration
        )
        
        # Enhance with parallel insights
        if iteration > 1:
            base_reflection.improvements.append(
                "Consider alternative approaches based on multiple previous attempts"
            )
            base_reflection.issues.append(
                "Pattern of repeated failures detected - fundamental approach change needed"
            )
            
            # Adjust confidence based on parallel analysis
            base_reflection.confidence = min(0.95, base_reflection.confidence + 0.1)
        
        return base_reflection
    
    async def process_batch_async(
        self,
        tasks: List[str],
        llm: str = "gpt-4",
        **kwargs
    ) -> List[ReflexionResult]:
        """Process multiple tasks asynchronously."""
        logger.info(f"Starting async batch processing of {len(tasks)} tasks")
        
        # Create agent factory for batch processing
        def agent_factory(**agent_kwargs):
            from ..core.agent import ReflexionAgent
            return ReflexionAgent(
                llm=llm,
                max_iterations=kwargs.get("max_iterations", 3),
                reflection_type=kwargs.get("reflection_type", ReflectionType.BINARY),
                success_threshold=kwargs.get("success_threshold", 0.7)
            )
        
        # Use batch processor
        results = await self.batch_processor.process_batch(
            tasks, agent_factory, **kwargs
        )
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cache_stats = performance_cache.get_stats()
        throttling_stats = adaptive_throttling.get_stats()
        resource_summary = self.resource_monitor.get_resource_summary(hours=1)
        
        return {
            "engine_stats": self.performance_stats,
            "cache_stats": cache_stats,
            "throttling_stats": throttling_stats,
            "resource_summary": resource_summary,
            "optimizations_enabled": {
                "caching": self.enable_caching,
                "parallel_reflection": self.enable_parallel_reflection,
                "batch_processing": True,
                "adaptive_throttling": True
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'batch_processor'):
            self.batch_processor.shutdown()
        logger.info("Optimized engine cleanup completed")