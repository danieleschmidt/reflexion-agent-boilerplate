"""Advanced performance optimization for reflexion agents."""

import asyncio
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Union
import json
import logging
from datetime import datetime, timedelta

from .types import ReflexionResult, Reflection
from .logging_config import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    avg_execution_time: float
    peak_execution_time: float
    success_rate: float
    avg_iterations: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_tasks_per_second: float
    cache_hit_rate: float
    error_rate: float
    optimization_score: float


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    enable_adaptive_batching: bool = True
    enable_memory_optimization: bool = True
    enable_prediction_optimization: bool = True
    max_concurrent_tasks: int = 10
    cache_size: int = 1000
    batch_size: int = 5
    optimization_interval: float = 60.0  # seconds


class ReflectionCache:
    """Intelligent caching system for reflection results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def _generate_key(self, task: str, llm: str, params: Dict[str, Any]) -> str:
        """Generate cache key for task and parameters."""
        key_data = {
            "task": task[:200],  # Truncate for reasonable key size
            "llm": llm,
            "reflection_type": params.get("reflection_type", "binary"),
            "threshold": params.get("success_threshold", 0.7),
            "max_iterations": params.get("max_iterations", 3)
        }
        return json.dumps(key_data, sort_keys=True)
    
    def get(self, task: str, llm: str, params: Dict[str, Any]) -> Optional[ReflexionResult]:
        """Get cached reflection result if available and valid."""
        with self.lock:
            key = self._generate_key(task, llm, params)
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    self.access_times[key] = time.time()
                    self.hit_count += 1
                    logger.debug(f"Cache hit for task: {task[:50]}...")
                    return entry["result"]
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.access_times[key]
            
            self.miss_count += 1
            logger.debug(f"Cache miss for task: {task[:50]}...")
            return None
    
    def put(self, task: str, llm: str, params: Dict[str, Any], result: ReflexionResult):
        """Store reflection result in cache."""
        with self.lock:
            key = self._generate_key(task, llm, params)
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = {
                "result": result,
                "timestamp": time.time()
            }
            self.access_times[key] = time.time()
            
            logger.debug(f"Cached result for task: {task[:50]}...")
    
    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[lru_key]
        del self.access_times[lru_key]
        logger.debug("Evicted LRU cache entry")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0


class AdaptiveBatcher:
    """Adaptive batching system for optimal task processing."""
    
    def __init__(self, initial_batch_size: int = 5, max_batch_size: int = 20):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 1
        self.performance_history = deque(maxlen=20)
        self.adaptation_rate = 0.1
        
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size based on performance history."""
        return self.current_batch_size
    
    def update_performance(self, batch_size: int, avg_time: float, success_rate: float):
        """Update batch size based on performance feedback."""
        efficiency_score = success_rate / max(avg_time, 0.1)  # Avoid division by zero
        
        self.performance_history.append({
            "batch_size": batch_size,
            "avg_time": avg_time,
            "success_rate": success_rate,
            "efficiency_score": efficiency_score,
            "timestamp": time.time()
        })
        
        # Adapt batch size based on recent performance
        if len(self.performance_history) >= 3:
            recent_scores = [entry["efficiency_score"] for entry in list(self.performance_history)[-3:]]
            recent_avg = sum(recent_scores) / len(recent_scores)
            
            # Increase batch size if performance is good and stable
            if recent_avg > 0.8 and self._is_performance_stable():
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * (1 + self.adaptation_rate))
                )
            # Decrease batch size if performance is poor
            elif recent_avg < 0.5:
                self.current_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * (1 - self.adaptation_rate))
                )
    
    def _is_performance_stable(self) -> bool:
        """Check if recent performance is stable."""
        if len(self.performance_history) < 3:
            return False
        
        recent_scores = [entry["efficiency_score"] for entry in list(self.performance_history)[-3:]]
        variance = sum((x - sum(recent_scores)/len(recent_scores))**2 for x in recent_scores) / len(recent_scores)
        
        return variance < 0.1  # Low variance indicates stability


class PredictiveOptimizer:
    """Predictive optimization for preemptive performance improvements."""
    
    def __init__(self):
        self.task_patterns = defaultdict(list)
        self.performance_predictions = {}
        self.optimization_suggestions = deque(maxlen=100)
        
    def analyze_task_pattern(self, task: str, result: ReflexionResult):
        """Analyze task patterns for predictive optimization."""
        task_signature = self._extract_task_signature(task)
        
        pattern_data = {
            "task_signature": task_signature,
            "execution_time": result.total_time,
            "iterations": result.iterations,
            "success": result.success,
            "reflection_count": len(result.reflections),
            "timestamp": datetime.now().isoformat()
        }
        
        self.task_patterns[task_signature].append(pattern_data)
        
        # Maintain reasonable history size
        if len(self.task_patterns[task_signature]) > 50:
            self.task_patterns[task_signature] = self.task_patterns[task_signature][-50:]
    
    def _extract_task_signature(self, task: str) -> str:
        """Extract meaningful signature from task for pattern matching."""
        # Simple keyword-based signature
        keywords = []
        task_lower = task.lower()
        
        # Task type indicators
        if any(word in task_lower for word in ['write', 'create', 'implement', 'build']):
            keywords.append('creation')
        if any(word in task_lower for word in ['analyze', 'review', 'examine', 'study']):
            keywords.append('analysis')
        if any(word in task_lower for word in ['fix', 'debug', 'correct', 'solve']):
            keywords.append('debugging')
        if any(word in task_lower for word in ['optimize', 'improve', 'enhance']):
            keywords.append('optimization')
        
        # Domain indicators
        if any(word in task_lower for word in ['python', 'code', 'function', 'algorithm']):
            keywords.append('programming')
        if any(word in task_lower for word in ['data', 'statistics', 'analysis']):
            keywords.append('data_science')
        if any(word in task_lower for word in ['web', 'api', 'server', 'database']):
            keywords.append('web_development')
        
        # Complexity indicators
        if len(task.split()) > 20:
            keywords.append('complex')
        elif len(task.split()) < 10:
            keywords.append('simple')
        else:
            keywords.append('medium')
        
        return '_'.join(sorted(keywords)) if keywords else 'general'
    
    def predict_performance(self, task: str) -> Dict[str, float]:
        """Predict performance metrics for a given task."""
        task_signature = self._extract_task_signature(task)
        
        if task_signature not in self.task_patterns:
            return {
                "predicted_time": 30.0,  # Default predictions
                "predicted_iterations": 2.5,
                "predicted_success_rate": 0.7,
                "confidence": 0.1
            }
        
        # Calculate predictions based on historical data
        historical_data = self.task_patterns[task_signature]
        
        if len(historical_data) < 3:
            confidence = 0.3
        else:
            confidence = min(0.9, 0.3 + (len(historical_data) * 0.05))
        
        avg_time = sum(entry["execution_time"] for entry in historical_data) / len(historical_data)
        avg_iterations = sum(entry["iterations"] for entry in historical_data) / len(historical_data)
        success_rate = sum(1 for entry in historical_data if entry["success"]) / len(historical_data)
        
        prediction = {
            "predicted_time": avg_time,
            "predicted_iterations": avg_iterations,
            "predicted_success_rate": success_rate,
            "confidence": confidence,
            "sample_size": len(historical_data)
        }
        
        self.performance_predictions[task_signature] = prediction
        return prediction
    
    def generate_optimization_suggestions(self, current_metrics: PerformanceMetrics) -> List[str]:
        """Generate optimization suggestions based on current performance."""
        suggestions = []
        
        # Analyze current performance issues
        if current_metrics.avg_execution_time > 60.0:
            suggestions.append("Consider enabling parallel processing for faster execution")
            suggestions.append("Increase cache size to improve response times")
        
        if current_metrics.success_rate < 0.7:
            suggestions.append("Increase max_iterations for better success rates")
            suggestions.append("Consider using ensemble or hierarchical reflexion algorithms")
        
        if current_metrics.cache_hit_rate < 0.3:
            suggestions.append("Increase cache TTL for better cache utilization")
            suggestions.append("Review task patterns for better caching strategies")
        
        if current_metrics.memory_usage_mb > 1024:  # > 1GB
            suggestions.append("Enable memory optimization to reduce memory usage")
            suggestions.append("Consider reducing cache size or batch size")
        
        if current_metrics.error_rate > 0.1:
            suggestions.append("Implement additional error handling and retry mechanisms")
            suggestions.append("Review task validation and preprocessing")
        
        # Store suggestions for analysis
        for suggestion in suggestions:
            self.optimization_suggestions.append({
                "suggestion": suggestion,
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics
            })
        
        return suggestions


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, strategy: OptimizationStrategy = None):
        self.strategy = strategy or OptimizationStrategy()
        self.cache = ReflectionCache(max_size=self.strategy.cache_size) if self.strategy.enable_caching else None
        self.batcher = AdaptiveBatcher(initial_batch_size=self.strategy.batch_size) if self.strategy.enable_adaptive_batching else None
        self.predictor = PredictiveOptimizer() if self.strategy.enable_prediction_optimization else None
        
        self.performance_history = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=self.strategy.max_concurrent_tasks) if self.strategy.enable_parallel_processing else None
        
        self.optimization_timer = None
        self.start_optimization_monitoring()
        
        logger.info(f"Performance optimizer initialized with strategy: {self.strategy}")
    
    def optimize_task_execution(self, task_func: Callable, task: str, **kwargs) -> ReflexionResult:
        """Optimize single task execution with caching and prediction."""
        start_time = time.time()
        
        # Try cache first if enabled
        if self.cache:
            cached_result = self.cache.get(task, kwargs.get('llm', 'gpt-4'), kwargs)
            if cached_result:
                logger.debug(f"Returning cached result for task: {task[:50]}...")
                return cached_result
        
        # Execute task with optimization
        try:
            result = task_func(task, **kwargs)
            
            # Cache successful results
            if self.cache and result.success:
                self.cache.put(task, kwargs.get('llm', 'gpt-4'), kwargs, result)
            
            # Update predictive models
            if self.predictor:
                self.predictor.analyze_task_pattern(task, result)
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_performance(task, result, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed after {execution_time:.2f}s: {e}")
            raise
    
    def optimize_batch_execution(self, task_func: Callable, tasks: List[str], **kwargs) -> List[ReflexionResult]:
        """Optimize batch task execution with parallel processing and adaptive batching."""
        if not self.strategy.enable_parallel_processing or not self.executor:
            # Fallback to sequential execution
            return [self.optimize_task_execution(task_func, task, **kwargs) for task in tasks]
        
        # Determine optimal batch size
        batch_size = self.batcher.get_optimal_batch_size() if self.batcher else self.strategy.batch_size
        
        results = []
        batch_start_time = time.time()
        
        # Process tasks in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = self._execute_batch_parallel(task_func, batch, **kwargs)
            results.extend(batch_results)
        
        # Update adaptive batcher
        if self.batcher:
            batch_time = time.time() - batch_start_time
            avg_time = batch_time / len(tasks)
            success_rate = sum(1 for r in results if r.success) / len(results)
            self.batcher.update_performance(batch_size, avg_time, success_rate)
        
        return results
    
    def _execute_batch_parallel(self, task_func: Callable, tasks: List[str], **kwargs) -> List[ReflexionResult]:
        """Execute batch of tasks in parallel."""
        futures = []
        
        for task in tasks:
            future = self.executor.submit(self.optimize_task_execution, task_func, task, **kwargs)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)  # 5-minute timeout per task
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel task execution failed: {e}")
                # Create error result
                error_result = self._create_error_result(str(e))
                results.append(error_result)
        
        return results
    
    def _create_error_result(self, error_message: str) -> ReflexionResult:
        """Create error result for failed tasks."""
        from .types import ReflexionResult
        
        return ReflexionResult(
            task="Error occurred during execution",
            output=f"Error: {error_message}",
            success=False,
            iterations=0,
            reflections=[],
            total_time=0.0,
            metadata={"error": True, "error_message": error_message}
        )
    
    def _record_performance(self, task: str, result: ReflexionResult, execution_time: float):
        """Record performance metrics for monitoring."""
        performance_record = {
            "task": task[:100],  # Truncate for storage efficiency
            "execution_time": execution_time,
            "success": result.success,
            "iterations": result.iterations,
            "reflection_count": len(result.reflections),
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance_history.append(performance_record)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if not self.performance_history:
            return PerformanceMetrics(
                avg_execution_time=0.0,
                peak_execution_time=0.0,
                success_rate=0.0,
                avg_iterations=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                throughput_tasks_per_second=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                optimization_score=0.0
            )
        
        recent_data = list(self.performance_history)[-100:]  # Last 100 records
        
        # Calculate metrics
        execution_times = [r["execution_time"] for r in recent_data]
        avg_execution_time = sum(execution_times) / len(execution_times)
        peak_execution_time = max(execution_times)
        
        success_count = sum(1 for r in recent_data if r["success"])
        success_rate = success_count / len(recent_data)
        
        avg_iterations = sum(r["iterations"] for r in recent_data) / len(recent_data)
        
        # Calculate throughput (tasks per second over last hour)
        recent_hour = datetime.now() - timedelta(hours=1)
        recent_tasks = [
            r for r in recent_data 
            if datetime.fromisoformat(r["timestamp"]) > recent_hour
        ]
        throughput = len(recent_tasks) / 3600.0 if recent_tasks else 0.0
        
        # Get cache stats
        cache_hit_rate = 0.0
        if self.cache:
            cache_stats = self.cache.get_stats()
            cache_hit_rate = cache_stats["hit_rate"]
        
        # Calculate optimization score (composite metric)
        optimization_score = (
            (success_rate * 0.4) +
            (min(1.0, 30.0 / max(avg_execution_time, 1.0)) * 0.3) +  # Prefer faster execution
            (cache_hit_rate * 0.2) +
            (min(1.0, throughput / 0.1) * 0.1)  # 0.1 tasks/sec as baseline
        )
        
        return PerformanceMetrics(
            avg_execution_time=avg_execution_time,
            peak_execution_time=peak_execution_time,
            success_rate=success_rate,
            avg_iterations=avg_iterations,
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            throughput_tasks_per_second=throughput,
            cache_hit_rate=cache_hit_rate,
            error_rate=1.0 - success_rate,
            optimization_score=optimization_score
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0  # psutil not available
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get current optimization suggestions."""
        if not self.predictor:
            return ["Enable predictive optimization for performance suggestions"]
        
        current_metrics = self.get_performance_metrics()
        return self.predictor.generate_optimization_suggestions(current_metrics)
    
    def start_optimization_monitoring(self):
        """Start periodic optimization monitoring."""
        def monitor():
            while True:
                try:
                    metrics = self.get_performance_metrics()
                    suggestions = self.get_optimization_suggestions()
                    
                    if metrics.optimization_score < 0.6:  # Poor performance threshold
                        logger.warning(f"Performance degradation detected. Score: {metrics.optimization_score:.2f}")
                        for suggestion in suggestions[:3]:  # Top 3 suggestions
                            logger.info(f"Optimization suggestion: {suggestion}")
                    
                    time.sleep(self.strategy.optimization_interval)
                    
                except Exception as e:
                    logger.error(f"Optimization monitoring error: {e}")
                    time.sleep(60)  # Retry after 1 minute
        
        if self.optimization_timer is None:
            self.optimization_timer = threading.Thread(target=monitor, daemon=True)
            self.optimization_timer.start()
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.cache:
            logger.info(f"Cache stats on shutdown: {self.cache.get_stats()}")
        
        logger.info("Performance optimizer shutdown completed")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()