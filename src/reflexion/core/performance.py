"""Performance optimization utilities for reflexion agents."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json

from .logging_config import logger


class PerformanceCache:
    """High-performance caching system for reflexion operations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, task: str, llm: str, params: Dict[str, Any]) -> str:
        """Generate cache key from task parameters."""
        # Create deterministic hash of task parameters
        key_data = {
            "task": task,
            "llm": llm,
            "params": sorted(params.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, task: str, llm: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(task, llm, params)
        
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        # Check TTL
        cache_entry = self.cache[key]
        age = time.time() - cache_entry["timestamp"]
        
        if age > self.ttl_seconds:
            self._evict_key(key)
            self.miss_count += 1
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        self.hit_count += 1
        
        logger.debug(f"Cache hit for task: {task[:50]}...")
        return cache_entry["data"]
    
    def put(self, task: str, llm: str, params: Dict[str, Any], data: Any):
        """Cache the result."""
        key = self._generate_key(task, llm, params)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        self.access_times[key] = time.time()
        
        logger.debug(f"Cached result for task: {task[:50]}...")
    
    def _evict_key(self, key: str):
        """Evict specific key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict_key(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Performance cache cleared")


class BatchProcessor:
    """Batch processing for multiple reflexion tasks."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        tasks: List[str],
        agent_factory: Callable,
        **agent_kwargs
    ) -> List[Any]:
        """Process multiple tasks concurrently."""
        if not tasks:
            return []
        
        logger.info(f"Processing batch of {len(tasks)} tasks with {self.max_workers} workers")
        start_time = time.time()
        
        # Split into batches
        batches = [tasks[i:i + self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} tasks)")
            
            # Create agents for this batch
            agents = [agent_factory(**agent_kwargs) for _ in batch]
            
            # Submit batch to thread pool
            futures = []
            for task, agent in zip(batch, agents):
                future = self.executor.submit(self._process_single_task, agent, task)
                futures.append(future)
            
            # Collect results as they complete
            batch_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per task
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch task failed: {str(e)}")
                    batch_results.append({"error": str(e), "success": False})
            
            all_results.extend(batch_results)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r.get("success", False))
        
        logger.info(
            f"Batch processing completed - {len(tasks)} tasks in {total_time:.2f}s, "
            f"Success rate: {success_count}/{len(tasks)} ({success_count/len(tasks):.1%})"
        )
        
        return all_results
    
    def _process_single_task(self, agent, task: str) -> Dict[str, Any]:
        """Process a single task."""
        try:
            result = agent.run(task)
            return {
                "task": task,
                "success": result.success,
                "output": result.output,
                "iterations": result.iterations,
                "total_time": result.total_time,
                "reflections": len(result.reflections)
            }
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return {
                "task": task,
                "success": False,
                "error": str(e),
                "iterations": 0,
                "total_time": 0,
                "reflections": 0
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the batch processor."""
        self.executor.shutdown(wait=wait)


class AdaptiveThrottling:
    """Adaptive throttling based on system performance and success rates."""
    
    def __init__(self):
        self.request_times: List[float] = []
        self.success_rates: List[float] = []
        self.current_delay = 0.0
        self.min_delay = 0.0
        self.max_delay = 2.0
        self.window_size = 100
        
        # Thresholds for adaptation
        self.high_latency_threshold = 5.0  # seconds
        self.low_success_threshold = 0.7   # 70%
        self.adaptation_factor = 1.2
    
    def record_request(self, duration: float, success: bool):
        """Record request metrics for adaptive throttling."""
        current_time = time.time()
        
        # Record timing
        self.request_times.append(duration)
        if len(self.request_times) > self.window_size:
            self.request_times = self.request_times[-self.window_size:]
        
        # Record success rate
        self.success_rates.append(1.0 if success else 0.0)
        if len(self.success_rates) > self.window_size:
            self.success_rates = self.success_rates[-self.window_size:]
        
        # Adapt throttling
        self._adapt_throttling()
    
    def _adapt_throttling(self):
        """Adapt throttling based on current metrics."""
        if len(self.request_times) < 10:  # Need minimum samples
            return
        
        avg_duration = sum(self.request_times) / len(self.request_times)
        avg_success = sum(self.success_rates) / len(self.success_rates)
        
        old_delay = self.current_delay
        
        # Increase delay if high latency or low success rate
        if avg_duration > self.high_latency_threshold or avg_success < self.low_success_threshold:
            self.current_delay = min(self.max_delay, self.current_delay * self.adaptation_factor)
        else:
            # Decrease delay if performance is good
            self.current_delay = max(self.min_delay, self.current_delay / self.adaptation_factor)
        
        if abs(self.current_delay - old_delay) > 0.1:
            logger.info(
                f"Adaptive throttling: {old_delay:.2f}s -> {self.current_delay:.2f}s "
                f"(avg_duration: {avg_duration:.2f}s, success_rate: {avg_success:.2%})"
            )
    
    async def throttle(self):
        """Apply current throttling delay."""
        if self.current_delay > 0:
            await asyncio.sleep(self.current_delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttling statistics."""
        if not self.request_times:
            return {"current_delay": self.current_delay, "sample_count": 0}
        
        return {
            "current_delay": self.current_delay,
            "sample_count": len(self.request_times),
            "avg_duration": sum(self.request_times) / len(self.request_times),
            "avg_success_rate": sum(self.success_rates) / len(self.success_rates),
            "min_delay": self.min_delay,
            "max_delay": self.max_delay
        }


class ResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "avg_response_time": 10.0
        }
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            import psutil
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
            }
        except ImportError:
            # Fallback metrics without psutil
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "process_count": 0,
                "note": "Limited metrics (psutil not available)"
            }
        
        # Add custom application metrics
        self._add_application_metrics(metrics)
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _add_application_metrics(self, metrics: Dict[str, Any]):
        """Add application-specific metrics."""
        # These would be populated by the reflexion system
        metrics.update({
            "active_agents": 0,  # Would be tracked by agent pool
            "cache_hit_rate": 0.0,  # Would be from performance cache
            "avg_response_time": 0.0,  # Would be from recent executions
            "error_rate": 0.0  # Would be from error tracking
        })
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds."""
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                logger.warning(
                    f"Resource alert: {metric} = {metrics[metric]:.1f} "
                    f"exceeds threshold {threshold:.1f}"
                )
    
    def get_resource_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified period"}
        
        # Calculate averages
        cpu_values = [m.get("cpu_percent", 0) for m in recent_metrics]
        memory_values = [m.get("memory_percent", 0) for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_percent": sum(memory_values) / len(memory_values),
            "max_memory_percent": max(memory_values),
            "latest_metrics": recent_metrics[-1] if recent_metrics else None
        }


def performance_profile(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            end_memory = 0
            try:
                end_memory = process.memory_info().rss
                memory_delta = end_memory - start_memory
            except:
                memory_delta = 0
            
            logger.debug(
                f"Performance profile - {func.__name__}: "
                f"{execution_time:.3f}s, memory_delta: {memory_delta/1024/1024:.1f}MB"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Performance profile - {func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise
    
    return wrapper


# Global performance instances
performance_cache = PerformanceCache()
adaptive_throttling = AdaptiveThrottling()
resource_monitor = ResourceMonitor()