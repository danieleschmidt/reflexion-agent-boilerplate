"""Advanced Performance Optimization System V2.0 for High-Scale Reflexion Operations.

This module implements enterprise-grade performance optimization including:
- Intelligent caching with adaptive eviction
- Connection pooling and resource management
- Concurrent execution with smart load balancing
- Memory optimization and garbage collection tuning
- Query optimization and batch processing
- Auto-scaling based on metrics
"""

import asyncio
import threading
import time
import statistics
import pickle
import hashlib
import gc
import sys
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import logging
import weakref
from functools import wraps, lru_cache
import json

logger = logging.getLogger(__name__)


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class PerformanceTier(Enum):
    """Performance optimization tiers."""
    BASIC = "basic"
    OPTIMIZED = "optimized"
    HIGH_PERFORMANCE = "high_performance"
    EXTREME = "extreme"


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation_name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    cache_hit: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    # Caching settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE
    
    # Concurrency settings
    max_workers: int = 8
    max_concurrent_operations: int = 100
    use_process_pool: bool = False
    
    # Memory optimization
    enable_gc_tuning: bool = True
    memory_threshold: float = 0.8  # 80% memory usage
    enable_memory_profiling: bool = False
    
    # Performance tier
    performance_tier: PerformanceTier = PerformanceTier.OPTIMIZED
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3
    min_workers: int = 2
    max_workers_limit: int = 32


class AdaptiveCache:
    """Advanced caching system with intelligent eviction and optimization."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE):
        self.max_size = max_size
        self.ttl = ttl
        self.policy = policy
        self._cache: OrderedDict = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._size_bytes = 0
        self._lock = threading.RLock()
        
        # Adaptive policy parameters
        self._access_pattern_history: deque = deque(maxlen=1000)
        self._optimal_policy = CacheEvictionPolicy.LRU
        self._policy_performance: Dict[CacheEvictionPolicy, float] = defaultdict(list)
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance tracking."""
        with self._lock:
            current_time = time.time()
            
            if key not in self._cache:
                self._miss_count += 1
                self._access_pattern_history.append(('miss', key, current_time))
                return None
            
            value, timestamp, size = self._cache[key]
            
            # Check TTL
            if current_time - timestamp > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                if key in self._access_counts:
                    del self._access_counts[key]
                self._size_bytes -= size
                self._miss_count += 1
                return None
            
            # Update access patterns
            self._access_counts[key] += 1
            self._access_times[key] = current_time
            self._hit_count += 1
            self._access_pattern_history.append(('hit', key, current_time))
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set item in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size = sys.getsizeof(pickle.dumps(value))
            except:
                size = sys.getsizeof(str(value))
            
            # Check if we need to evict
            while len(self._cache) >= self.max_size:
                self._evict_item()
            
            # Store item
            self._cache[key] = (value, current_time, size)
            self._access_times[key] = current_time
            self._access_counts[key] += 1
            self._size_bytes += size
            
            # Adapt eviction policy if needed
            if len(self._access_pattern_history) > 500:
                self._adapt_eviction_policy()
            
            return True
    
    def _evict_item(self):
        """Evict item based on current policy."""
        if not self._cache:
            return
        
        if self.policy == CacheEvictionPolicy.ADAPTIVE:
            policy = self._optimal_policy
        else:
            policy = self.policy
        
        key_to_evict = None
        
        if policy == CacheEvictionPolicy.LRU:
            # Least recently used (first item in OrderedDict)
            key_to_evict = next(iter(self._cache))
        
        elif policy == CacheEvictionPolicy.LFU:
            # Least frequently used
            min_count = min(self._access_counts[k] for k in self._cache.keys())
            candidates = [k for k in self._cache.keys() if self._access_counts[k] == min_count]
            key_to_evict = min(candidates, key=lambda k: self._access_times.get(k, 0))
        
        elif policy == CacheEvictionPolicy.FIFO:
            # First in, first out
            key_to_evict = next(iter(self._cache))
        
        if key_to_evict:
            _, _, size = self._cache[key_to_evict]
            del self._cache[key_to_evict]
            if key_to_evict in self._access_times:
                del self._access_times[key_to_evict]
            if key_to_evict in self._access_counts:
                del self._access_counts[key_to_evict]
            self._size_bytes -= size
    
    def _adapt_eviction_policy(self):
        """Adapt eviction policy based on access patterns."""
        if self.policy != CacheEvictionPolicy.ADAPTIVE:
            return
        
        # Analyze recent access patterns
        recent_accesses = list(self._access_pattern_history)[-200:]
        
        # Calculate hit rate for each potential policy
        policies_to_test = [CacheEvictionPolicy.LRU, CacheEvictionPolicy.LFU, CacheEvictionPolicy.FIFO]
        
        for policy in policies_to_test:
            # Simulate cache performance with this policy
            hit_rate = self._simulate_policy_performance(recent_accesses, policy)
            self._policy_performance[policy].append(hit_rate)
        
        # Choose best performing policy
        best_policy = max(policies_to_test, 
                         key=lambda p: statistics.mean(self._policy_performance[p][-10:]) 
                         if self._policy_performance[p] else 0)
        
        if best_policy != self._optimal_policy:
            logger.info(f"Cache adapting eviction policy from {self._optimal_policy} to {best_policy}")
            self._optimal_policy = best_policy
    
    def _simulate_policy_performance(self, access_history: List[Tuple], policy: CacheEvictionPolicy) -> float:
        """Simulate cache performance with given policy."""
        # Simplified simulation - in practice this would be more sophisticated
        hits = sum(1 for access_type, _, _ in access_history if access_type == 'hit')
        total = len(access_history)
        return hits / max(total, 1)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._access_times.clear()
            self._size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(total_requests, 1)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "size_bytes": self._size_bytes,
                "current_policy": self._optimal_policy.value if self.policy == CacheEvictionPolicy.ADAPTIVE else self.policy.value,
                "policy_performance": {
                    policy.value: statistics.mean(perfs[-5:]) if perfs else 0.0
                    for policy, perfs in self._policy_performance.items()
                }
            }


class ConnectionPool:
    """Advanced connection pooling with health monitoring."""
    
    def __init__(self, create_connection: Callable, max_connections: int = 10, max_idle_time: int = 300):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self._pool: deque = deque()
        self._active_connections: Set = set()
        self._connection_stats: Dict = defaultdict(lambda: {"uses": 0, "created": time.time(), "last_used": time.time()})
        self._lock = threading.RLock()
        
        # Health monitoring
        self._health_check_interval = 60  # 1 minute
        self._last_health_check = 0
    
    async def get_connection(self):
        """Get connection from pool or create new one."""
        with self._lock:
            # Perform health check if needed
            if time.time() - self._last_health_check > self._health_check_interval:
                await self._health_check()
            
            # Try to get from pool
            while self._pool:
                conn = self._pool.popleft()
                if self._is_connection_healthy(conn):
                    self._active_connections.add(conn)
                    self._connection_stats[id(conn)]["uses"] += 1
                    self._connection_stats[id(conn)]["last_used"] = time.time()
                    return conn
            
            # Create new connection if under limit
            if len(self._active_connections) < self.max_connections:
                conn = await self.create_connection()
                self._active_connections.add(conn)
                self._connection_stats[id(conn)]["uses"] = 1
                self._connection_stats[id(conn)]["created"] = time.time()
                self._connection_stats[id(conn)]["last_used"] = time.time()
                return conn
            
            # Pool is full, wait for connection to be returned
            # In a real implementation, this would use proper async waiting
            raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool."""
        with self._lock:
            if conn in self._active_connections:
                self._active_connections.remove(conn)
                
                # Only return to pool if not too old
                conn_stats = self._connection_stats[id(conn)]
                if time.time() - conn_stats["last_used"] < self.max_idle_time:
                    self._pool.append(conn)
                else:
                    # Close old connection
                    self._close_connection(conn)
    
    def _is_connection_healthy(self, conn) -> bool:
        """Check if connection is healthy."""
        # Implement connection health check logic
        return True
    
    async def _health_check(self):
        """Perform health check on pooled connections."""
        self._last_health_check = time.time()
        
        # Remove unhealthy connections
        healthy_connections = deque()
        while self._pool:
            conn = self._pool.popleft()
            if self._is_connection_healthy(conn):
                healthy_connections.append(conn)
            else:
                self._close_connection(conn)
        
        self._pool = healthy_connections
    
    def _close_connection(self, conn):
        """Close connection and clean up stats."""
        try:
            # Implement connection closing logic
            pass
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            if id(conn) in self._connection_stats:
                del self._connection_stats[id(conn)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            total_connections = len(self._pool) + len(self._active_connections)
            avg_uses = statistics.mean([stats["uses"] for stats in self._connection_stats.values()]) if self._connection_stats else 0
            
            return {
                "pooled_connections": len(self._pool),
                "active_connections": len(self._active_connections),
                "total_connections": total_connections,
                "max_connections": self.max_connections,
                "utilization": len(self._active_connections) / max(self.max_connections, 1),
                "average_uses_per_connection": avg_uses
            }


class AdvancedPerformanceOptimizer:
    """Advanced performance optimization system for production workloads."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.metrics: deque = deque(maxlen=10000)
        self.caches: Dict[str, AdaptiveCache] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.performance_tier = self.config.performance_tier
        
        # Auto-scaling
        self.current_workers = self.config.max_workers
        self.load_history: deque = deque(maxlen=100)
        self.scaling_decisions: List[Dict] = []
        
        # Memory optimization
        if self.config.enable_gc_tuning:
            self._tune_garbage_collection()
        
        # Initialize thread pools
        self._initialize_executors()
        
        # Performance monitoring
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.bottlenecks: List[Dict] = []
        
        logger.info(f"Performance optimizer initialized with tier: {self.performance_tier.value}")
    
    def _initialize_executors(self):
        """Initialize thread and process executors."""
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="perf_optimizer"
        )
        
        if self.config.use_process_pool:
            self.process_pool = ProcessPoolExecutor(
                max_workers=max(2, self.current_workers // 2)
            )
    
    def _tune_garbage_collection(self):
        """Optimize garbage collection settings."""
        # Set aggressive garbage collection thresholds for better memory management
        gc.set_threshold(700, 10, 10)
        
        # Enable automatic garbage collection
        gc.enable()
        
        logger.info("Garbage collection tuning applied")
    
    def get_cache(self, cache_name: str) -> AdaptiveCache:
        """Get or create cache instance."""
        if cache_name not in self.caches:
            self.caches[cache_name] = AdaptiveCache(
                max_size=self.config.cache_size,
                ttl=self.config.cache_ttl,
                policy=self.config.eviction_policy
            )
        return self.caches[cache_name]
    
    async def cached_execution(self, cache_name: str, cache_key: str, operation: Callable, *args, **kwargs):
        """Execute operation with caching support."""
        cache = self.get_cache(cache_name)
        
        # Try cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute operation
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, operation, *args, **kwargs
                )
            
            # Cache result
            cache.set(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            self._record_performance_metric(cache_name, duration, cache_hit=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Cached execution failed for {cache_name}: {e}")
            raise
    
    async def concurrent_execution(self, operations: List[Tuple[Callable, tuple, dict]], max_concurrency: Optional[int] = None) -> List[Any]:
        """Execute multiple operations concurrently with optimal resource allocation."""
        if not operations:
            return []
        
        max_concurrent = min(
            max_concurrency or self.config.max_concurrent_operations,
            len(operations),
            self.current_workers
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def execute_with_semaphore(operation, args, kwargs):
            async with semaphore:
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation(*args, **kwargs)
                    else:
                        # Use appropriate executor
                        executor = self.process_pool if self.config.use_process_pool else self.thread_pool
                        result = await asyncio.get_event_loop().run_in_executor(
                            executor, lambda: operation(*args, **kwargs)
                        )
                    
                    duration = time.time() - start_time
                    self._record_performance_metric("concurrent_execution", duration)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Concurrent operation failed: {e}")
                    return None
        
        # Execute all operations concurrently
        tasks = [
            execute_with_semaphore(op, args, kwargs)
            for op, args, kwargs in operations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update load metrics for auto-scaling
        load = len(operations) / max_concurrent
        self.load_history.append(load)
        
        # Trigger auto-scaling if enabled
        if self.config.enable_auto_scaling:
            await self._check_auto_scaling()
        
        return results
    
    async def batch_processing(self, items: List[Any], processor: Callable, batch_size: Optional[int] = None) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(items))
        
        results = []
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        batch_operations = [
            (processor, (batch,), {})
            for batch in batches
        ]
        
        batch_results = await self.concurrent_execution(batch_operations)
        
        # Flatten results
        for batch_result in batch_results:
            if batch_result is not None and isinstance(batch_result, list):
                results.extend(batch_result)
            elif batch_result is not None:
                results.append(batch_result)
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on system resources."""
        # Consider available workers, memory constraints, and historical performance
        base_batch_size = max(1, total_items // (self.current_workers * 2))
        
        # Adjust based on performance tier
        if self.performance_tier == PerformanceTier.EXTREME:
            return min(base_batch_size, 50)
        elif self.performance_tier == PerformanceTier.HIGH_PERFORMANCE:
            return min(base_batch_size, 100)
        elif self.performance_tier == PerformanceTier.OPTIMIZED:
            return min(base_batch_size, 200)
        else:
            return min(base_batch_size, 500)
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed based on current load."""
        if len(self.load_history) < 10:
            return
        
        recent_load = statistics.mean(list(self.load_history)[-10:])
        
        # Scale up if load is consistently high
        if (recent_load > self.config.scale_up_threshold and 
            self.current_workers < self.config.max_workers_limit):
            
            new_workers = min(self.current_workers + 2, self.config.max_workers_limit)
            await self._scale_workers(new_workers, "scale_up", recent_load)
        
        # Scale down if load is consistently low
        elif (recent_load < self.config.scale_down_threshold and 
              self.current_workers > self.config.min_workers):
            
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            await self._scale_workers(new_workers, "scale_down", recent_load)
    
    async def _scale_workers(self, new_worker_count: int, reason: str, load: float):
        """Scale worker threads/processes."""
        old_workers = self.current_workers
        self.current_workers = new_worker_count
        
        # Reinitialize executors with new worker count
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        self._initialize_executors()
        
        # Record scaling decision
        scaling_decision = {
            "timestamp": datetime.now(),
            "reason": reason,
            "old_workers": old_workers,
            "new_workers": new_worker_count,
            "load": load
        }
        self.scaling_decisions.append(scaling_decision)
        
        logger.info(f"Auto-scaled workers: {old_workers} -> {new_worker_count} ({reason}, load: {load:.2f})")
    
    def _record_performance_metric(self, operation_name: str, duration: float, cache_hit: bool = False, memory_usage: float = 0.0):
        """Record performance metrics for analysis."""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # Would be measured in real implementation
            cache_hit=cache_hit
        )
        
        self.metrics.append(metric)
        self.operation_stats[operation_name].append(duration)
        
        # Detect bottlenecks
        self._detect_bottlenecks(operation_name, duration)
    
    def _detect_bottlenecks(self, operation_name: str, duration: float):
        """Detect performance bottlenecks."""
        stats = self.operation_stats[operation_name]
        if len(stats) < 10:
            return
        
        # Calculate performance statistics
        recent_stats = stats[-10:]
        avg_duration = statistics.mean(recent_stats)
        p95_duration = statistics.quantiles(recent_stats, n=20)[18]  # 95th percentile
        
        # Flag as bottleneck if significantly slow
        if duration > avg_duration * 2 or duration > p95_duration:
            bottleneck = {
                "operation": operation_name,
                "duration": duration,
                "avg_duration": avg_duration,
                "p95_duration": p95_duration,
                "timestamp": datetime.now(),
                "severity": "high" if duration > avg_duration * 3 else "medium"
            }
            
            self.bottlenecks.append(bottleneck)
            logger.warning(f"Performance bottleneck detected: {operation_name} took {duration:.2f}s (avg: {avg_duration:.2f}s)")
    
    def optimize_memory(self):
        """Perform memory optimization."""
        if self.config.enable_memory_profiling:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            if memory_percent > self.config.memory_threshold:
                logger.info(f"High memory usage detected ({memory_percent:.1%}), performing optimization")
                
                # Clear old cache entries
                for cache in self.caches.values():
                    cache.clear()
                
                # Force garbage collection
                gc.collect()
                
                # Reduce concurrent operations if needed
                if memory_percent > 0.9:
                    self.config.max_concurrent_operations = max(10, self.config.max_concurrent_operations // 2)
                    logger.info(f"Reduced max concurrent operations to {self.config.max_concurrent_operations}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics)[-100:]  # Last 100 operations
        
        # Calculate overall statistics
        durations = [m.duration for m in recent_metrics]
        avg_duration = statistics.mean(durations) if durations else 0
        p50_duration = statistics.median(durations) if durations else 0
        p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations) if durations else 0
        
        # Cache statistics
        cache_stats = {}
        for name, cache in self.caches.items():
            cache_stats[name] = cache.get_stats()
        
        # Connection pool statistics
        pool_stats = {}
        for name, pool in self.connection_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Recent bottlenecks
        recent_bottlenecks = [b for b in self.bottlenecks if datetime.now() - b["timestamp"] < timedelta(hours=1)]
        
        # Auto-scaling history
        recent_scaling = [s for s in self.scaling_decisions if datetime.now() - s["timestamp"] < timedelta(hours=24)]
        
        return {
            "performance_tier": self.performance_tier.value,
            "current_workers": self.current_workers,
            "metrics": {
                "total_operations": len(self.metrics),
                "avg_duration": avg_duration,
                "p50_duration": p50_duration,
                "p95_duration": p95_duration,
                "operations_by_type": dict(defaultdict(int, {
                    m.operation_name: sum(1 for m2 in recent_metrics if m2.operation_name == m.operation_name)
                    for m in recent_metrics
                }))
            },
            "caching": {
                "total_caches": len(self.caches),
                "cache_details": cache_stats,
                "overall_hit_rate": statistics.mean([
                    stats["hit_rate"] for stats in cache_stats.values()
                ]) if cache_stats else 0.0
            },
            "connection_pools": {
                "total_pools": len(self.connection_pools),
                "pool_details": pool_stats
            },
            "bottlenecks": {
                "recent_count": len(recent_bottlenecks),
                "top_bottlenecks": sorted(recent_bottlenecks, key=lambda x: x["duration"], reverse=True)[:5]
            },
            "auto_scaling": {
                "enabled": self.config.enable_auto_scaling,
                "recent_scaling_events": len(recent_scaling),
                "current_load": statistics.mean(list(self.load_history)[-10:]) if self.load_history else 0.0,
                "scaling_history": recent_scaling[-10:]  # Last 10 scaling events
            },
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current performance."""
        recommendations = []
        
        # Check cache performance
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            if stats["hit_rate"] < 0.5:
                recommendations.append(f"Consider increasing cache size for {name} (current hit rate: {stats['hit_rate']:.1%})")
        
        # Check for consistent bottlenecks
        recent_bottlenecks = [b for b in self.bottlenecks if datetime.now() - b["timestamp"] < timedelta(hours=1)]
        bottleneck_operations = defaultdict(int)
        for bottleneck in recent_bottlenecks:
            bottleneck_operations[bottleneck["operation"]] += 1
        
        for operation, count in bottleneck_operations.items():
            if count >= 3:
                recommendations.append(f"Optimize {operation} operation - frequent bottleneck ({count} occurrences)")
        
        # Check memory usage
        if self.config.enable_memory_profiling:
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    recommendations.append(f"High memory usage ({memory_percent:.1f}%) - consider memory optimization")
            except ImportError:
                pass
        
        # Check scaling patterns
        recent_scaling = [s for s in self.scaling_decisions if datetime.now() - s["timestamp"] < timedelta(hours=6)]
        scale_ups = sum(1 for s in recent_scaling if s["reason"] == "scale_up")
        scale_downs = sum(1 for s in recent_scaling if s["reason"] == "scale_down")
        
        if scale_ups > scale_downs * 2:
            recommendations.append("Consider increasing base worker count - frequent scale-up events")
        elif scale_downs > scale_ups * 2:
            recommendations.append("Consider decreasing base worker count - frequent scale-down events")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown performance optimizer and clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Performance optimizer shutdown completed")


# Global performance optimizer instance
performance_optimizer = AdvancedPerformanceOptimizer()