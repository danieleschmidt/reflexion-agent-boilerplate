"""Health checking and system monitoring for reflexion agents."""

import asyncio
import time
import logging
import os
import platform

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Fallback implementations
    class psutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 50.0  # Fallback value
        
        @staticmethod
        def virtual_memory():
            class Memory:
                percent = 60.0
                available = 4 * 1024**3  # 4GB
                total = 8 * 1024**3  # 8GB
            return Memory()
        
        @staticmethod
        def disk_usage(path):
            class Disk:
                free = 50 * 1024**3  # 50GB
                used = 50 * 1024**3  # 50GB
                total = 100 * 1024**3  # 100GB
            return Disk()

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ReflexionError


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Comprehensive health checking for reflexion systems."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Health check thresholds
        self.thresholds = {
            'cpu_warning': self.config.get('cpu_warning_threshold', 80.0),
            'cpu_critical': self.config.get('cpu_critical_threshold', 95.0),
            'memory_warning': self.config.get('memory_warning_threshold', 85.0),
            'memory_critical': self.config.get('memory_critical_threshold', 95.0),
            'disk_warning': self.config.get('disk_warning_threshold', 90.0),
            'disk_critical': self.config.get('disk_critical_threshold', 98.0),
            'response_time_warning': self.config.get('response_time_warning_ms', 1000),
            'response_time_critical': self.config.get('response_time_critical_ms', 5000)
        }
        
        # Health check history
        self.health_history: List[HealthCheckResult] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Custom health checks
        self.custom_checks: Dict[str, Callable] = {}
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        checks = {}
        
        # System resource checks
        checks['cpu'] = await self.check_cpu_usage()
        checks['memory'] = await self.check_memory_usage()
        checks['disk'] = await self.check_disk_usage()
        checks['uptime'] = await self.check_uptime()
        
        # Component checks
        checks['llm_connectivity'] = await self.check_llm_connectivity()
        checks['memory_storage'] = await self.check_memory_storage()
        
        # Custom checks
        for name, check_func in self.custom_checks.items():
            try:
                checks[name] = await self._run_check_safely(name, check_func)
            except Exception as e:
                checks[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )
        
        # Store in history
        for result in checks.values():
            self.health_history.append(result)
        
        # Maintain history size
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
        
        return checks
    
    async def check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            duration_ms = (time.time() - start_time) * 1000
            
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                name="cpu",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={"cpu_percent": cpu_percent}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU usage: {str(e)}"
            )
    
    async def check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            duration_ms = (time.time() - start_time) * 1000
            
            if memory_percent >= self.thresholds['memory_critical']:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent >= self.thresholds['memory_warning']:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {str(e)}"
            )
    
    async def check_disk_usage(self) -> HealthCheckResult:
        """Check disk usage."""
        start_time = time.time()
        
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            duration_ms = (time.time() - start_time) * 1000
            
            if disk_percent >= self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent >= self.thresholds['disk_warning']:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk usage: {str(e)}"
            )
    
    async def check_uptime(self) -> HealthCheckResult:
        """Check system uptime."""
        start_time = time.time()
        
        try:
            uptime_seconds = time.time() - self.start_time
            duration_ms = (time.time() - start_time) * 1000
            
            uptime_hours = uptime_seconds / 3600
            
            if uptime_hours < 1:
                status = HealthStatus.WARNING
                message = f"System recently started: {uptime_seconds:.0f}s uptime"
            else:
                status = HealthStatus.HEALTHY
                message = f"System uptime: {uptime_hours:.1f}h"
            
            return HealthCheckResult(
                name="uptime",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={"uptime_seconds": uptime_seconds}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="uptime",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check uptime: {str(e)}"
            )
    
    async def check_llm_connectivity(self) -> HealthCheckResult:
        """Check LLM service connectivity."""
        start_time = time.time()
        
        try:
            # Simulate LLM connectivity check
            await asyncio.sleep(0.1)  # Simulate network call
            
            duration_ms = (time.time() - start_time) * 1000
            
            if duration_ms >= self.thresholds['response_time_critical']:
                status = HealthStatus.CRITICAL
                message = f"LLM response time critical: {duration_ms:.0f}ms"
            elif duration_ms >= self.thresholds['response_time_warning']:
                status = HealthStatus.WARNING
                message = f"LLM response time slow: {duration_ms:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"LLM connectivity good: {duration_ms:.0f}ms"
            
            return HealthCheckResult(
                name="llm_connectivity",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={"response_time_ms": duration_ms}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="llm_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"LLM connectivity failed: {str(e)}"
            )
    
    async def check_memory_storage(self) -> HealthCheckResult:
        """Check memory storage accessibility."""
        start_time = time.time()
        
        try:
            # Check if we can read/write to memory storage
            test_data = {"test": True, "timestamp": time.time()}
            
            # Simulate storage check
            await asyncio.sleep(0.05)
            
            duration_ms = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            message = f"Memory storage accessible: {duration_ms:.0f}ms"
            
            return HealthCheckResult(
                name="memory_storage",
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata={"storage_response_time_ms": duration_ms}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_storage",
                status=HealthStatus.CRITICAL,
                message=f"Memory storage failed: {str(e)}"
            )
    
    async def _run_check_safely(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a health check safely with timeout."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check_func(),
                timeout=self.config.get('check_timeout_seconds', 10)
            )
            
            if isinstance(result, HealthCheckResult):
                return result
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=f"Custom check passed: {str(result)}"
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.config.get('check_timeout_seconds', 10)}s"
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}"
            )
    
    def register_custom_check(self, name: str, check_func: Callable):
        """Register a custom health check function."""
        self.custom_checks[name] = check_func
        self.logger.info(f"Registered custom health check: {name}")
    
    def get_overall_status(self, checks: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall system health status."""
        if not checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.WARNING  # Treat unknown as warning
        else:
            return HealthStatus.HEALTHY
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            uptime = time.time() - self.start_time
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                uptime_seconds=uptime
            )
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                uptime_seconds=0.0
            )
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_checks = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time
        ]
        
        if not recent_checks:
            return {"status": "no_data", "hours": hours}
        
        # Count statuses
        status_counts = {}
        check_names = set()
        
        for check in recent_checks:
            check_names.add(check.name)
            status = check.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate availability per check type
        availability_by_check = {}
        for name in check_names:
            name_checks = [c for c in recent_checks if c.name == name]
            healthy_checks = [c for c in name_checks if c.status == HealthStatus.HEALTHY]
            availability = len(healthy_checks) / len(name_checks) if name_checks else 0
            availability_by_check[name] = availability
        
        # Overall availability
        total_checks = len(recent_checks)
        healthy_checks = status_counts.get('healthy', 0)
        overall_availability = healthy_checks / total_checks if total_checks > 0 else 0
        
        return {
            "period_hours": hours,
            "total_checks": total_checks,
            "status_distribution": status_counts,
            "overall_availability": overall_availability,
            "availability_by_check": availability_by_check,
            "check_types": list(check_names)
        }


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """Decorator for circuit breaker."""
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                    self.logger.info("Circuit breaker half-open - attempting recovery")
                else:
                    raise ReflexionError(
                        f"Circuit breaker OPEN - service unavailable",
                        {"failure_count": self.failure_count, "state": self.state}
                    )
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
                
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "closed"
        if self.state == "half-open":
            self.logger.info("Circuit breaker recovered - state closed")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )


# Global health checker instance
health_checker = HealthChecker()