"""
Advanced Performance Optimization for Reflexion Agents.

Provides caching, concurrent processing, resource pooling, and auto-scaling.
"""

import asyncio
import time
import threading
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import weakref

from .exceptions import LLMError, ReflexionError


class PerformanceError(ReflexionError):
    """Performance-related errors."""
    def __init__(self, message: str, component: str = "unknown"):
        super().__init__(message)
        self.component = component


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


class SmartCache:
    """Intelligent cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.stats['misses'] += 1
                self.stats['size'] = len(self.cache)
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end of access order (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            ttl = ttl if ttl is not None else self.default_ttl
            
            # If key already exists, update it
            if key in self.cache:
                self.cache[key].value = value
                self.cache[key].last_accessed = current_time
                self.cache[key].access_count += 1
                self.cache[key].ttl = ttl
                
                # Move to end
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.stats['size'] = len(self.cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.popleft()
        if lru_key in self.cache:
            del self.cache[lru_key]
            self.stats['evictions'] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class TaskCache(SmartCache):
    """Specialized cache for reflexion tasks."""
    
    def __init__(self, max_size: int = 500):
        super().__init__(max_size, default_ttl=1800)  # 30 minutes default
    
    def get_task_key(self, task: str, llm: str, reflection_type: str) -> str:
        """Generate cache key for task."""
        # Normalize task text
        normalized_task = task.lower().strip()
        
        # Create deterministic key
        key_data = f"{normalized_task}|{llm}|{reflection_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_task_result(self, task: str, llm: str, reflection_type: str) -> Optional[Any]:
        """Get cached task result."""
        key = self.get_task_key(task, llm, reflection_type)
        return self.get(key)
    
    def cache_task_result(self, task: str, llm: str, reflection_type: str, 
                         result: Any, ttl: Optional[float] = None) -> None:
        """Cache task result."""
        key = self.get_task_key(task, llm, reflection_type)
        self.put(key, result, ttl)


class ConnectionPool:
    """Connection pool for LLM providers."""
    
    def __init__(self, factory: Callable, min_size: int = 2, max_size: int = 10):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.pool: deque = deque()
        self.active_connections = 0
        self._lock = threading.Lock()
        
        # Pre-populate with minimum connections
        self._populate_pool()
    
    def _populate_pool(self):
        """Populate pool with minimum connections."""
        for _ in range(self.min_size):
            try:
                conn = self.factory()
                self.pool.append(conn)
            except Exception as e:
                print(f"Failed to create connection: {e}")
    
    def get_connection(self):
        """Get connection from pool."""
        with self._lock:
            if self.pool:
                conn = self.pool.popleft()
                self.active_connections += 1
                return ConnectionWrapper(self, conn)
            
            elif self.active_connections < self.max_size:
                # Create new connection
                try:
                    conn = self.factory()
                    self.active_connections += 1
                    return ConnectionWrapper(self, conn)
                except Exception as e:
                    raise PerformanceError(f"Failed to create connection: {e}")
            
            else:
                raise PerformanceError("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self._lock:
            if len(self.pool) < self.max_size:
                self.pool.append(conn)
            self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in pool."""
        with self._lock:
            while self.pool:
                conn = self.pool.popleft()
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                except:
                    pass


class ConnectionWrapper:
    """Wrapper for pooled connections."""
    
    def __init__(self, pool: ConnectionPool, connection):
        self.pool = pool
        self.connection = connection
        self._released = False
    
    def __enter__(self):
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def release(self):
        """Release connection back to pool."""
        if not self._released:
            self.pool.return_connection(self.connection)
            self._released = True
    
    def __getattr__(self, name):
        return getattr(self.connection, name)


class ConcurrentExecutor:
    """High-performance concurrent execution manager."""
    
    def __init__(self, thread_workers: int = 4, process_workers: int = 2):
        self.thread_workers = thread_workers
        self.process_workers = process_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=process_workers)
        
        # Task queues
        self.task_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        
        # Performance metrics
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0,
            'queue_size': 0
        }
    
    async def execute_concurrent_tasks(self, tasks: List[Tuple[Callable, tuple, dict]], 
                                     max_concurrent: int = 5) -> List[Any]:
        """Execute multiple tasks concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_task(task_func, args, kwargs):
            async with semaphore:
                start_time = time.time()
                try:
                    # Run in thread pool for I/O bound tasks
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.thread_pool, 
                        lambda: task_func(*args, **kwargs)
                    )
                    
                    execution_time = time.time() - start_time
                    self.metrics['tasks_completed'] += 1
                    self._update_avg_time(execution_time)
                    
                    return result
                    
                except Exception as e:
                    self.metrics['tasks_failed'] += 1
                    return f"Error: {str(e)}"
        
        # Submit all tasks
        self.metrics['tasks_submitted'] += len(tasks)
        coroutines = [
            execute_task(func, args, kwargs) 
            for func, args, kwargs in tasks
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return results
    
    def _update_avg_time(self, new_time: float):
        """Update average execution time."""
        completed = self.metrics['tasks_completed']
        if completed == 1:
            self.metrics['avg_execution_time'] = new_time
        else:
            current_avg = self.metrics['avg_execution_time']
            self.metrics['avg_execution_time'] = (
                (current_avg * (completed - 1) + new_time) / completed
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_tasks = self.metrics['tasks_submitted']
        success_rate = (
            self.metrics['tasks_completed'] / total_tasks 
            if total_tasks > 0 else 0
        )
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'thread_pool_size': self.thread_workers,
            'process_pool_size': self.process_workers
        }
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AdaptiveThrottler:
    """Adaptive request throttling based on performance metrics."""
    
    def __init__(self, initial_rate: float = 1.0):
        self.current_rate = initial_rate
        self.target_success_rate = 0.95
        self.adjustment_factor = 0.1
        
        # Metrics window
        self.window_size = 100
        self.recent_results: deque = deque(maxlen=self.window_size)
        self.last_adjustment = time.time()
        self.adjustment_interval = 30  # seconds
        
        self._lock = threading.Lock()
    
    def record_result(self, success: bool, response_time: float):
        """Record task execution result."""
        with self._lock:
            self.recent_results.append({
                'success': success,
                'response_time': response_time,
                'timestamp': time.time()
            })
            
            # Check if we should adjust rate
            if time.time() - self.last_adjustment > self.adjustment_interval:
                self._adjust_rate()
    
    def _adjust_rate(self):
        """Adjust throttling rate based on recent performance."""
        if len(self.recent_results) < 10:
            return
        
        # Calculate success rate
        successes = sum(1 for r in self.recent_results if r['success'])
        success_rate = successes / len(self.recent_results)
        
        # Calculate average response time
        avg_response_time = sum(r['response_time'] for r in self.recent_results) / len(self.recent_results)
        
        # Adjust rate based on performance
        if success_rate < self.target_success_rate:
            # Decrease rate (slow down)
            self.current_rate *= (1 - self.adjustment_factor)
        elif success_rate > self.target_success_rate and avg_response_time < 2.0:
            # Increase rate (speed up)
            self.current_rate *= (1 + self.adjustment_factor)
        
        # Clamp rate to reasonable bounds
        self.current_rate = max(0.1, min(10.0, self.current_rate))
        self.last_adjustment = time.time()
    
    async def throttle(self):
        """Apply throttling delay."""
        delay = 1.0 / self.current_rate
        await asyncio.sleep(delay)
    
    def get_current_rate(self) -> float:
        """Get current throttling rate."""
        return self.current_rate


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.task_cache = TaskCache()
        self.executor = ConcurrentExecutor()
        self.throttler = AdaptiveThrottler()
        
        # Connection pools for different providers
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        # Performance monitoring
        self.start_time = time.time()
        self.optimization_stats = {
            'cache_enabled': True,
            'concurrent_execution': True,
            'adaptive_throttling': True,
            'connection_pooling': True
        }
    
    def register_connection_pool(self, provider: str, factory: Callable, 
                               min_size: int = 2, max_size: int = 10):
        """Register connection pool for provider."""
        self.connection_pools[provider] = ConnectionPool(factory, min_size, max_size)
    
    def get_connection(self, provider: str):
        """Get connection from pool."""
        if provider in self.connection_pools:
            return self.connection_pools[provider].get_connection()
        else:
            raise PerformanceError(f"No connection pool for provider: {provider}")
    
    async def execute_optimized(self, tasks: List[dict], max_concurrent: int = 5) -> List[Any]:
        """Execute tasks with full optimization."""
        optimized_tasks = []
        
        for task_data in tasks:
            task = task_data['task']
            llm = task_data['llm']
            reflection_type = task_data.get('reflection_type', 'binary')
            
            # Check cache first
            cached_result = self.task_cache.get_task_result(task, llm, reflection_type)
            if cached_result:
                optimized_tasks.append(('_return_cached', (cached_result,), {}))
            else:
                optimized_tasks.append((
                    '_execute_with_optimization',
                    (task_data,),
                    {}
                ))
        
        # Execute with concurrency
        results = await self.executor.execute_concurrent_tasks(
            optimized_tasks, max_concurrent
        )
        
        return results
    
    def _return_cached(self, result):
        """Return cached result."""
        return result
    
    def _execute_with_optimization(self, task_data):
        """Execute single task with optimization."""
        # This would integrate with the main reflexion engine
        # For now, return a mock result
        return {
            'task': task_data['task'],
            'success': True,
            'optimized': True,
            'timestamp': time.time()
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'optimization_features': self.optimization_stats,
            'cache_stats': self.task_cache.get_stats(),
            'executor_stats': self.executor.get_performance_stats(),
            'throttling_rate': self.throttler.get_current_rate(),
            'connection_pools': {
                name: {
                    'active_connections': pool.active_connections,
                    'pool_size': len(pool.pool),
                    'max_size': pool.max_size
                }
                for name, pool in self.connection_pools.items()
            }
        }
    
    def shutdown(self):
        """Shutdown all optimization components."""
        self.executor.shutdown()
        for pool in self.connection_pools.values():
            pool.close_all()


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()