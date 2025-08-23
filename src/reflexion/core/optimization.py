"""Advanced optimization and performance enhancement for reflexion systems."""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import hashlib
import pickle
from dataclasses import dataclass, field
from enum import Enum
import logging

from .types import ReflexionResult, Reflection


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    CACHING = "caching"
    PARALLEL_EXECUTION = "parallel_execution"
    RESULT_MEMOIZATION = "result_memoization"
    BATCH_PROCESSING = "batch_processing"
    LAZY_EVALUATION = "lazy_evaluation"
    PREFETCHING = "prefetching"
    COMPRESSION = "compression"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


@dataclass 
class OptimizationMetrics:
    """Metrics for optimization performance."""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    parallel_tasks: int = 0
    time_saved_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    compression_ratio: float = 1.0


class SmartCache:
    """Intelligent caching system with LRU eviction and TTL support."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,
        max_memory_mb: float = 100.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.current_memory = 0
        self.lock = threading.RLock()
        
        self.metrics = OptimizationMetrics()
        self.logger = logging.getLogger(__name__)
    
    def _hash_key(self, obj: Any) -> str:
        """Generate cache key from object."""
        if isinstance(obj, str):
            return hashlib.md5(obj.encode(), usedforsecurity=False).hexdigest()
        elif isinstance(obj, dict):
            # Sort keys for consistent hashing
            sorted_items = sorted(obj.items())
            return hashlib.md5(str(sorted_items).encode(), usedforsecurity=False).hexdigest()
        else:
            # Try to pickle for complex objects
            try:
                return hashlib.md5(pickle.dumps(obj), usedforsecurity=False).hexdigest()
            except:
                return hashlib.md5(str(obj).encode(), usedforsecurity=False).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            if isinstance(obj, (str, int, float, bool)):
                return len(str(obj).encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            else:
                # Use pickle size as approximation
                return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache."""
        cache_key = self._hash_key(key)
        
        with self.lock:
            if cache_key not in self.cache:
                self.metrics.cache_misses += 1
                return default
            
            entry = self.cache[cache_key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[cache_key]
                self.access_order.remove(cache_key)
                self.current_memory -= entry.size_bytes
                self.metrics.cache_misses += 1
                self.metrics.cache_evictions += 1
                return default
            
            # Update access tracking
            entry.access_count += 1
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            self.metrics.cache_hits += 1
            return entry.value
    
    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache."""
        cache_key = self._hash_key(key)
        size_bytes = self._estimate_size(value)
        
        with self.lock:
            # Check if we need to make space
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + size_bytes > self.max_memory_bytes):
                if not self._evict_lru():
                    break
            
            # Check if we can still fit
            if (len(self.cache) >= self.max_size or
                self.current_memory + size_bytes > self.max_memory_bytes):
                return False
            
            # Remove existing entry if updating
            if cache_key in self.cache:
                old_entry = self.cache[cache_key]
                self.current_memory -= old_entry.size_bytes
                self.access_order.remove(cache_key)
            
            # Add new entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            self.cache[cache_key] = entry
            self.access_order.append(cache_key)
            self.current_memory += size_bytes
            
            return True
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_order:
            return False
        
        lru_key = self.access_order.pop(0)
        if lru_key in self.cache:
            entry = self.cache[lru_key]
            del self.cache[lru_key]
            self.current_memory -= entry.size_bytes
            self.metrics.cache_evictions += 1
            return True
        
        return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = 0.0
            total_requests = self.metrics.cache_hits + self.metrics.cache_misses
            if total_requests > 0:
                hit_rate = self.metrics.cache_hits / total_requests
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "evictions": self.metrics.cache_evictions
            }


class ParallelExecutor:
    """Advanced parallel execution manager for reflexion tasks."""
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        chunk_size: int = 10
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        if use_processes:
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.process_pool = None
        
        self.metrics = OptimizationMetrics()
        self.logger = logging.getLogger(__name__)
    
    async def execute_parallel(
        self,
        tasks: List[Callable],
        task_args: List[Tuple] = None,
        use_processes: Optional[bool] = None
    ) -> List[Any]:
        """Execute tasks in parallel."""
        if not tasks:
            return []
        
        task_args = task_args or [() for _ in tasks]
        use_processes = use_processes or self.use_processes
        
        self.metrics.parallel_tasks += len(tasks)
        start_time = time.time()
        
        try:
            if use_processes and self.process_pool:
                # Use process pool for CPU-intensive tasks
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(self.process_pool, task, *args)
                    for task, args in zip(tasks, task_args)
                ]
            else:
                # Use thread pool for I/O-intensive tasks
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(self.thread_pool, task, *args)
                    for task, args in zip(tasks, task_args)
                ]
            
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            execution_time = time.time() - start_time
            self.metrics.time_saved_seconds += execution_time
            
            self.logger.info(f"Executed {len(tasks)} parallel tasks in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            raise
    
    def execute_batch(
        self,
        tasks: List[Callable],
        batch_size: Optional[int] = None
    ) -> List[List[Any]]:
        """Execute tasks in batches."""
        batch_size = batch_size or self.chunk_size
        batches = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = []
            
            for task in batch:
                try:
                    result = task()
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch task failed: {str(e)}")
                    batch_results.append(e)
            
            batches.append(batch_results)
        
        return batches
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ResultMemoizer:
    """Memoization system for reflexion results."""
    
    def __init__(self, cache: Optional[SmartCache] = None):
        self.cache = cache or SmartCache(max_size=500, default_ttl=1800)  # 30 min TTL
        self.logger = logging.getLogger(__name__)
    
    def memoize(
        self,
        func: Callable,
        key_generator: Optional[Callable] = None,
        ttl: Optional[float] = None
    ) -> Callable:
        """Decorator for memoizing function results."""
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = {"args": args, "kwargs": kwargs}
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for memoized function: {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            try:
                result = func(*args, **kwargs)
                self.cache.put(cache_key, result, ttl)
                self.logger.debug(f"Cached result for function: {func.__name__}")
                return result
            except Exception as e:
                self.logger.error(f"Memoized function failed: {str(e)}")
                raise
        
        return wrapper
    
    def memoize_async(
        self,
        func: Callable,
        key_generator: Optional[Callable] = None,
        ttl: Optional[float] = None
    ) -> Callable:
        """Async version of memoization decorator."""
        
        async def wrapper(*args, **kwargs):
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = {"args": args, "kwargs": kwargs}
            
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            try:
                result = await func(*args, **kwargs)
                self.cache.put(cache_key, result, ttl)
                return result
            except Exception as e:
                self.logger.error(f"Async memoized function failed: {str(e)}")
                raise
        
        return wrapper


class PrefetchManager:
    """Intelligent prefetching for commonly accessed data."""
    
    def __init__(self, cache: SmartCache):
        self.cache = cache
        self.access_patterns: Dict[str, List[float]] = {}  # key -> timestamps
        self.prefetch_rules: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def record_access(self, key: str):
        """Record access pattern for prefetch prediction."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(time.time())
        
        # Keep only recent accesses (last hour)
        one_hour_ago = time.time() - 3600
        self.access_patterns[key] = [
            timestamp for timestamp in self.access_patterns[key]
            if timestamp > one_hour_ago
        ]
    
    def predict_next_access(self, key: str) -> Optional[float]:
        """Predict when key will be accessed next."""
        if key not in self.access_patterns:
            return None
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return None
        
        # Simple prediction based on average interval
        intervals = [
            accesses[i] - accesses[i-1]
            for i in range(1, len(accesses))
        ]
        
        avg_interval = sum(intervals) / len(intervals)
        return accesses[-1] + avg_interval
    
    def should_prefetch(self, key: str) -> bool:
        """Determine if key should be prefetched."""
        if key not in self.access_patterns:
            return False
        
        accesses = self.access_patterns[key]
        if len(accesses) < 3:
            return False
        
        # Prefetch if frequently accessed (more than 3 times in last hour)
        return len(accesses) >= 3
    
    def add_prefetch_rule(self, rule: Callable[[str], bool]):
        """Add custom prefetch rule."""
        self.prefetch_rules.append(rule)
    
    async def run_prefetch_cycle(self, data_loader: Callable[[str], Any]):
        """Run prefetch cycle for predicted accesses."""
        prefetch_count = 0
        
        for key in list(self.access_patterns.keys()):
            should_prefetch = self.should_prefetch(key)
            
            # Apply custom rules
            for rule in self.prefetch_rules:
                if rule(key):
                    should_prefetch = True
                    break
            
            if should_prefetch and self.cache.get(key) is None:
                try:
                    data = await data_loader(key)
                    self.cache.put(key, data)
                    prefetch_count += 1
                except Exception as e:
                    self.logger.warning(f"Prefetch failed for {key}: {str(e)}")
        
        if prefetch_count > 0:
            self.logger.info(f"Prefetched {prefetch_count} items")


class OptimizationManager:
    """Comprehensive optimization management system."""
    
    def __init__(
        self,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        enable_memoization: bool = True,
        enable_prefetch: bool = True,
        cache_size: int = 1000,
        parallel_workers: int = 4
    ):
        self.strategies = set()
        
        # Initialize components based on configuration
        if enable_caching:
            self.cache = SmartCache(max_size=cache_size)
            self.strategies.add(OptimizationStrategy.CACHING)
        else:
            self.cache = None
        
        if enable_parallel:
            self.parallel_executor = ParallelExecutor(max_workers=parallel_workers)
            self.strategies.add(OptimizationStrategy.PARALLEL_EXECUTION)
        else:
            self.parallel_executor = None
        
        if enable_memoization and self.cache:
            self.memoizer = ResultMemoizer(self.cache)
            self.strategies.add(OptimizationStrategy.RESULT_MEMOIZATION)
        else:
            self.memoizer = None
        
        if enable_prefetch and self.cache:
            self.prefetch_manager = PrefetchManager(self.cache)
            self.strategies.add(OptimizationStrategy.PREFETCHING)
        else:
            self.prefetch_manager = None
        
        self.metrics = OptimizationMetrics()
        self.logger = logging.getLogger(__name__)
    
    def optimize_reflexion_result(self, result: ReflexionResult) -> ReflexionResult:
        """Apply optimizations to reflexion result."""
        if OptimizationStrategy.COMPRESSION in self.strategies:
            # Compress verbose reflections
            compressed_reflections = []
            for reflection in result.reflections:
                if len(reflection.improvements) > 5:
                    # Keep only top 3 improvements
                    compressed_reflection = Reflection(
                        task=reflection.task,
                        output=reflection.output,
                        success=reflection.success,
                        score=reflection.score,
                        issues=reflection.issues[:3],  # Top 3 issues
                        improvements=reflection.improvements[:3],  # Top 3 improvements
                        confidence=reflection.confidence,
                        timestamp=reflection.timestamp
                    )
                    compressed_reflections.append(compressed_reflection)
                else:
                    compressed_reflections.append(reflection)
            
            result.reflections = compressed_reflections
        
        return result
    
    async def optimize_batch_execution(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize batch task execution."""
        if not self.parallel_executor:
            # Sequential execution
            results = []
            for task in tasks:
                # Simulate task execution
                results.append({"task_id": task.get("id"), "result": "completed"})
            return results
        
        # Group tasks by similarity for better caching
        task_groups = self._group_similar_tasks(tasks)
        
        # Execute groups in parallel
        all_results = []
        for group in task_groups:
            if len(group) == 1:
                # Single task
                result = {"task_id": group[0].get("id"), "result": "completed"}
                all_results.append(result)
            else:
                # Parallel execution for group
                group_tasks = [
                    lambda t=task: {"task_id": t.get("id"), "result": "completed"}
                    for task in group
                ]
                
                group_results = await self.parallel_executor.execute_parallel(group_tasks)
                all_results.extend(group_results)
        
        return all_results
    
    def _group_similar_tasks(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar tasks for optimized execution."""
        # Simple grouping by task type or domain
        groups = {}
        
        for task in tasks:
            task_type = task.get("type", "default")
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(task)
        
        return list(groups.values())
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "strategies_enabled": [s.value for s in self.strategies],
            "metrics": {
                "cache_performance": self.cache.get_stats() if self.cache else {},
                "parallel_tasks": self.metrics.parallel_tasks,
                "time_saved_seconds": self.metrics.time_saved_seconds
            }
        }
        
        if self.parallel_executor:
            stats["parallel_executor"] = {
                "max_workers": self.parallel_executor.max_workers,
                "use_processes": self.parallel_executor.use_processes
            }
        
        return stats
    
    def clear_optimizations(self):
        """Clear all optimization caches and reset metrics."""
        if self.cache:
            self.cache.clear()
        
        self.metrics = OptimizationMetrics()
        self.logger.info("Optimization caches cleared")
    
    def shutdown(self):
        """Shutdown optimization components."""
        if self.parallel_executor:
            self.parallel_executor.shutdown()
        
        self.logger.info("Optimization manager shutdown completed")


# Global optimization manager instance
optimization_manager = OptimizationManager()