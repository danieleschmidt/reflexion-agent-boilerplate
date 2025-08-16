"""
Advanced Resilience Engine v4.0
Comprehensive error recovery, monitoring, and self-healing capabilities
"""

import asyncio
import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from .types import ReflectionType, ReflexionResult
from .autonomous_sdlc_engine import QualityMetrics


class FailureType(Enum):
    """Types of system failures"""
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADE = "cascade"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_DEPENDENCY = "external_dependency"
    LOGIC_ERROR = "logic_error"
    SECURITY_BREACH = "security_breach"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""
    RETRY_EXPONENTIAL = "retry_exponential"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD_ISOLATION = "bulkhead_isolation"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER_SWITCH = "failover_switch"
    ROLLBACK_TRANSACTION = "rollback_transaction"
    RESOURCE_SCALING = "resource_scaling"
    SELF_HEALING = "self_healing"


class SeverityLevel(Enum):
    """Severity levels for incidents"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass
class FailureEvent:
    """Represents a system failure event"""
    id: str
    timestamp: datetime
    failure_type: FailureType
    severity: SeverityLevel
    component: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_strategy: Optional[RecoveryStrategy] = None
    impact_scope: Set[str] = field(default_factory=set)


@dataclass
class RecoveryAction:
    """Represents a recovery action"""
    strategy: RecoveryStrategy
    target_component: str
    action_function: Callable
    prerequisites: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_attempts: int = 3
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    name: str
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    last_failure_time: Optional[datetime] = None
    success_threshold: int = 3
    consecutive_successes: int = 0


class HealthMetrics:
    """System health metrics collection"""
    
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.thresholds = {
            'response_time': 1.0,  # seconds
            'error_rate': 0.05,    # 5%
            'cpu_usage': 0.80,     # 80%
            'memory_usage': 0.85,  # 85%
            'disk_usage': 0.90     # 90%
        }
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Keep only last 1000 measurements
        if len(self.metrics[name]) >= 1000:
            self.metrics[name].popleft()
        
        self.metrics[name].append((timestamp, value))
    
    def get_average(self, name: str, window_minutes: int = 5) -> float:
        """Get average metric value over time window"""
        if name not in self.metrics:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            value for timestamp, value in self.metrics[name]
            if timestamp >= cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def is_healthy(self, name: str) -> bool:
        """Check if metric is within healthy threshold"""
        if name not in self.thresholds:
            return True
        
        current_value = self.get_average(name, 1)  # 1 minute window
        return current_value <= self.thresholds[name]


class AdvancedResilienceEngine:
    """
    Advanced Resilience Engine with intelligent error recovery,
    self-healing capabilities, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        max_concurrent_recoveries: int = 5,
        failure_analysis_window: int = 300,  # 5 minutes
        enable_self_healing: bool = True
    ):
        self.max_concurrent_recoveries = max_concurrent_recoveries
        self.failure_analysis_window = failure_analysis_window
        self.enable_self_healing = enable_self_healing
        
        # State management
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_history: List[FailureEvent] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_actions: Dict[RecoveryStrategy, RecoveryAction] = {}
        self.health_metrics = HealthMetrics()
        
        # Concurrency control
        self.recovery_semaphore = asyncio.Semaphore(max_concurrent_recoveries)
        self.recovery_tasks: Set[asyncio.Task] = set()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize recovery strategies
        self._initialize_recovery_actions()
    
    async def initialize(self):
        """Initialize the resilience engine"""
        self.logger.info("🛡️ Initializing Advanced Resilience Engine")
        
        # Start continuous monitoring
        if not self.monitoring_active:
            await self.start_monitoring()
        
        # Initialize circuit breakers for critical components
        await self._initialize_circuit_breakers()
        
        self.logger.info("✅ Resilience Engine initialized")
    
    async def start_monitoring(self):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._continuous_monitoring())
        self.logger.info("📊 Started continuous monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("🛑 Stopped monitoring")
    
    async def handle_failure(
        self,
        component: str,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> FailureEvent:
        """Handle a system failure with intelligent recovery"""
        
        # Create failure event
        failure_event = FailureEvent(
            id=f"{component}_{int(time.time())}_{id(error)}",
            timestamp=datetime.now(),
            failure_type=await self._classify_failure(error, context or {}),
            severity=await self._assess_severity(error, component, context or {}),
            component=component,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        self.active_failures[failure_event.id] = failure_event
        self.failure_history.append(failure_event)
        
        self.logger.error(
            f"🚨 Failure detected in {component}: {failure_event.error_message}"
        )
        
        # Trigger recovery process
        recovery_task = asyncio.create_task(
            self._execute_recovery_process(failure_event)
        )
        self.recovery_tasks.add(recovery_task)
        
        # Clean up completed tasks
        recovery_task.add_done_callback(
            lambda t: self.recovery_tasks.discard(t)
        )
        
        return failure_event
    
    async def _execute_recovery_process(self, failure_event: FailureEvent):
        """Execute intelligent recovery process"""
        async with self.recovery_semaphore:
            try:
                # Analyze failure patterns
                failure_analysis = await self._analyze_failure_patterns(failure_event)
                
                # Select recovery strategy
                recovery_strategy = await self._select_recovery_strategy(
                    failure_event, failure_analysis
                )
                
                # Execute recovery
                recovery_success = await self._execute_recovery_strategy(
                    failure_event, recovery_strategy
                )
                
                if recovery_success:
                    failure_event.resolved = True
                    failure_event.resolution_strategy = recovery_strategy
                    del self.active_failures[failure_event.id]
                    
                    self.logger.info(
                        f"✅ Recovery successful for {failure_event.component} "
                        f"using {recovery_strategy.value}"
                    )
                else:
                    # Escalate to advanced recovery
                    await self._escalate_recovery(failure_event)
                
            except Exception as recovery_error:
                self.logger.error(
                    f"❌ Recovery process failed: {recovery_error}"
                )
                await self._handle_recovery_failure(failure_event, recovery_error)
    
    async def _classify_failure(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> FailureType:
        """Classify failure type using intelligent analysis"""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Network/connection errors
        if any(keyword in error_message for keyword in 
               ['timeout', 'connection', 'network', 'unreachable']):
            return FailureType.EXTERNAL_DEPENDENCY
        
        # Resource exhaustion
        if any(keyword in error_message for keyword in 
               ['memory', 'disk', 'space', 'limit', 'quota']):
            return FailureType.RESOURCE_EXHAUSTION
        
        # Performance issues
        if any(keyword in error_message for keyword in 
               ['slow', 'performance', 'latency', 'response time']):
            return FailureType.PERFORMANCE_DEGRADATION
        
        # Security issues
        if any(keyword in error_message for keyword in 
               ['unauthorized', 'forbidden', 'authentication', 'permission']):
            return FailureType.SECURITY_BREACH
        
        # Check if it's a cascade failure
        if len(self.active_failures) > 3:
            return FailureType.CASCADE
        
        # Check if it's persistent
        recent_failures = [
            f for f in self.failure_history
            if f.component == context.get('component') and
            f.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        if len(recent_failures) > 2:
            return FailureType.PERSISTENT
        
        return FailureType.TRANSIENT
    
    async def _assess_severity(
        self, 
        error: Exception, 
        component: str, 
        context: Dict[str, Any]
    ) -> SeverityLevel:
        """Assess failure severity"""
        
        # Critical components
        critical_components = ['database', 'auth', 'core_engine', 'security']
        if any(critical in component.lower() for critical in critical_components):
            return SeverityLevel.CRITICAL
        
        # High severity for security issues
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in 
               ['security', 'breach', 'unauthorized', 'attack']):
            return SeverityLevel.HIGH
        
        # Medium severity for performance issues
        if any(keyword in error_message for keyword in 
               ['slow', 'timeout', 'performance']):
            return SeverityLevel.MEDIUM
        
        # Check cascade potential
        if len(self.active_failures) > 1:
            return SeverityLevel.HIGH
        
        return SeverityLevel.LOW
    
    async def _analyze_failure_patterns(self, failure_event: FailureEvent) -> Dict[str, Any]:
        """Analyze failure patterns to inform recovery strategy"""
        
        # Analyze recent failures in same component
        component_failures = [
            f for f in self.failure_history
            if f.component == failure_event.component and
            f.timestamp > datetime.now() - timedelta(seconds=self.failure_analysis_window)
        ]
        
        # Analyze system-wide failure patterns
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp > datetime.now() - timedelta(seconds=self.failure_analysis_window)
        ]
        
        failure_rate = len(recent_failures) / (self.failure_analysis_window / 60)  # per minute
        
        analysis = {
            'component_failure_count': len(component_failures),
            'system_failure_rate': failure_rate,
            'is_cascade': len(self.active_failures) > 2,
            'recurring_pattern': len(component_failures) > 2,
            'failure_types': [f.failure_type for f in recent_failures],
            'affected_components': list(set(f.component for f in recent_failures))
        }
        
        return analysis
    
    async def _select_recovery_strategy(
        self, 
        failure_event: FailureEvent, 
        analysis: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Select optimal recovery strategy based on failure analysis"""
        
        # Strategy selection logic
        if failure_event.failure_type == FailureType.TRANSIENT:
            return RecoveryStrategy.RETRY_EXPONENTIAL
        
        elif failure_event.failure_type == FailureType.EXTERNAL_DEPENDENCY:
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        elif failure_event.failure_type == FailureType.CASCADE:
            return RecoveryStrategy.BULKHEAD_ISOLATION
        
        elif failure_event.failure_type == FailureType.RESOURCE_EXHAUSTION:
            return RecoveryStrategy.RESOURCE_SCALING
        
        elif failure_event.failure_type == FailureType.PERFORMANCE_DEGRADATION:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        elif failure_event.failure_type == FailureType.PERSISTENT:
            if self.enable_self_healing:
                return RecoveryStrategy.SELF_HEALING
            else:
                return RecoveryStrategy.ROLLBACK_TRANSACTION
        
        elif failure_event.failure_type == FailureType.SECURITY_BREACH:
            return RecoveryStrategy.BULKHEAD_ISOLATION
        
        # Default strategy
        return RecoveryStrategy.RETRY_EXPONENTIAL
    
    async def _execute_recovery_strategy(
        self, 
        failure_event: FailureEvent, 
        strategy: RecoveryStrategy
    ) -> bool:
        """Execute specific recovery strategy"""
        
        self.logger.info(f"🔧 Executing {strategy.value} for {failure_event.component}")
        
        recovery_functions = {
            RecoveryStrategy.RETRY_EXPONENTIAL: self._retry_with_exponential_backoff,
            RecoveryStrategy.CIRCUIT_BREAKER: self._activate_circuit_breaker,
            RecoveryStrategy.BULKHEAD_ISOLATION: self._isolate_component,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._enable_graceful_degradation,
            RecoveryStrategy.FAILOVER_SWITCH: self._execute_failover,
            RecoveryStrategy.ROLLBACK_TRANSACTION: self._rollback_transaction,
            RecoveryStrategy.RESOURCE_SCALING: self._scale_resources,
            RecoveryStrategy.SELF_HEALING: self._self_heal_component
        }
        
        recovery_function = recovery_functions.get(strategy)
        if not recovery_function:
            self.logger.error(f"Unknown recovery strategy: {strategy}")
            return False
        
        try:
            return await recovery_function(failure_event)
        except Exception as e:
            self.logger.error(f"Recovery strategy {strategy.value} failed: {e}")
            return False
    
    async def _retry_with_exponential_backoff(self, failure_event: FailureEvent) -> bool:
        """Implement exponential backoff retry strategy"""
        max_attempts = 5
        base_delay = 1.0
        
        for attempt in range(max_attempts):
            try:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
                # Simulate retry logic - in real implementation, this would
                # re-execute the failed operation
                success = await self._simulate_component_recovery(failure_event.component)
                
                if success:
                    self.logger.info(f"✅ Retry successful after {attempt + 1} attempts")
                    return True
                
            except Exception as e:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        return False
    
    async def _activate_circuit_breaker(self, failure_event: FailureEvent) -> bool:
        """Activate circuit breaker for component"""
        component = failure_event.component
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState(name=component)
        
        breaker = self.circuit_breakers[component]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "OPEN"
            self.logger.info(f"🔴 Circuit breaker OPEN for {component}")
            
            # Schedule recovery check
            asyncio.create_task(self._schedule_circuit_breaker_recovery(component))
            
            return True
        
        return False
    
    async def _schedule_circuit_breaker_recovery(self, component: str):
        """Schedule circuit breaker recovery attempt"""
        breaker = self.circuit_breakers[component]
        
        await asyncio.sleep(breaker.recovery_timeout)
        
        if breaker.state == "OPEN":
            breaker.state = "HALF_OPEN"
            self.logger.info(f"🟡 Circuit breaker HALF_OPEN for {component}")
            
            # Test if component is recovered
            try:
                success = await self._simulate_component_recovery(component)
                if success:
                    breaker.state = "CLOSED"
                    breaker.failure_count = 0
                    breaker.consecutive_successes = 0
                    self.logger.info(f"🟢 Circuit breaker CLOSED for {component}")
                else:
                    breaker.state = "OPEN"
                    # Schedule another recovery attempt
                    asyncio.create_task(self._schedule_circuit_breaker_recovery(component))
            except Exception as e:
                breaker.state = "OPEN"
                self.logger.error(f"Circuit breaker recovery test failed: {e}")
    
    async def _isolate_component(self, failure_event: FailureEvent) -> bool:
        """Isolate failing component to prevent cascade failures"""
        component = failure_event.component
        
        self.logger.info(f"🔒 Isolating component: {component}")
        
        # Implement bulkhead isolation
        # In real implementation, this would:
        # 1. Stop all requests to the component
        # 2. Drain existing connections
        # 3. Redirect traffic to healthy components
        
        await asyncio.sleep(0.1)  # Simulate isolation process
        return True
    
    async def _enable_graceful_degradation(self, failure_event: FailureEvent) -> bool:
        """Enable graceful degradation mode"""
        component = failure_event.component
        
        self.logger.info(f"📉 Enabling graceful degradation for: {component}")
        
        # Implement graceful degradation
        # In real implementation, this would:
        # 1. Reduce feature set
        # 2. Use cached data
        # 3. Return simplified responses
        
        await asyncio.sleep(0.1)  # Simulate degradation setup
        return True
    
    async def _execute_failover(self, failure_event: FailureEvent) -> bool:
        """Execute failover to backup systems"""
        component = failure_event.component
        
        self.logger.info(f"🔄 Executing failover for: {component}")
        
        # Implement failover logic
        await asyncio.sleep(0.1)  # Simulate failover
        return True
    
    async def _rollback_transaction(self, failure_event: FailureEvent) -> bool:
        """Rollback failed transaction or deployment"""
        component = failure_event.component
        
        self.logger.info(f"⏪ Rolling back transaction for: {component}")
        
        # Implement rollback logic
        await asyncio.sleep(0.1)  # Simulate rollback
        return True
    
    async def _scale_resources(self, failure_event: FailureEvent) -> bool:
        """Scale resources to handle increased load"""
        component = failure_event.component
        
        self.logger.info(f"📈 Scaling resources for: {component}")
        
        # Implement resource scaling
        await asyncio.sleep(0.1)  # Simulate scaling
        return True
    
    async def _self_heal_component(self, failure_event: FailureEvent) -> bool:
        """Self-heal component using AI-driven recovery"""
        component = failure_event.component
        
        self.logger.info(f"🤖 Self-healing component: {component}")
        
        # Implement AI-driven self-healing
        # In real implementation, this would:
        # 1. Analyze failure patterns
        # 2. Apply learned fixes
        # 3. Validate recovery
        
        await asyncio.sleep(0.2)  # Simulate self-healing process
        return True
    
    async def _simulate_component_recovery(self, component: str) -> bool:
        """Simulate component recovery for testing"""
        # In real implementation, this would test actual component health
        await asyncio.sleep(0.05)
        return True  # Assume recovery successful for simulation
    
    async def _escalate_recovery(self, failure_event: FailureEvent):
        """Escalate recovery to higher-level strategies"""
        self.logger.warning(f"🚨 Escalating recovery for {failure_event.component}")
        
        # Try more aggressive recovery strategies
        escalation_strategies = [
            RecoveryStrategy.BULKHEAD_ISOLATION,
            RecoveryStrategy.ROLLBACK_TRANSACTION,
            RecoveryStrategy.FAILOVER_SWITCH
        ]
        
        for strategy in escalation_strategies:
            if strategy != failure_event.resolution_strategy:
                success = await self._execute_recovery_strategy(failure_event, strategy)
                if success:
                    failure_event.resolved = True
                    failure_event.resolution_strategy = strategy
                    return
        
        # If all strategies fail, mark as unrecoverable
        failure_event.severity = SeverityLevel.CATASTROPHIC
        self.logger.critical(f"💥 Unrecoverable failure in {failure_event.component}")
    
    async def _handle_recovery_failure(self, failure_event: FailureEvent, recovery_error: Exception):
        """Handle recovery process failure"""
        self.logger.error(
            f"Recovery process failed for {failure_event.component}: {recovery_error}"
        )
        
        # Record recovery failure
        failure_event.recovery_attempts.append({
            'timestamp': datetime.now(),
            'error': str(recovery_error),
            'strategy_attempted': failure_event.resolution_strategy.value if failure_event.resolution_strategy else 'unknown'
        })
    
    async def _continuous_monitoring(self):
        """Continuous system monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_health_metrics()
                await self._detect_anomalies()
                await self._cleanup_old_failures()
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _collect_health_metrics(self):
        """Collect system health metrics"""
        # Simulate metric collection
        import random
        
        self.health_metrics.record_metric('response_time', random.uniform(0.1, 2.0))
        self.health_metrics.record_metric('error_rate', random.uniform(0.01, 0.1))
        self.health_metrics.record_metric('cpu_usage', random.uniform(0.3, 0.9))
        self.health_metrics.record_metric('memory_usage', random.uniform(0.4, 0.8))
    
    async def _detect_anomalies(self):
        """Detect system anomalies from metrics"""
        unhealthy_metrics = []
        
        for metric_name in self.health_metrics.thresholds:
            if not self.health_metrics.is_healthy(metric_name):
                unhealthy_metrics.append(metric_name)
        
        if unhealthy_metrics:
            self.logger.warning(f"🔍 Anomalies detected in metrics: {unhealthy_metrics}")
            
            # Trigger preventive measures
            await self._trigger_preventive_measures(unhealthy_metrics)
    
    async def _trigger_preventive_measures(self, unhealthy_metrics: List[str]):
        """Trigger preventive measures for unhealthy metrics"""
        for metric in unhealthy_metrics:
            if metric == 'response_time':
                await self._optimize_performance()
            elif metric == 'error_rate':
                await self._enhance_error_handling()
            elif metric in ['cpu_usage', 'memory_usage']:
                await self._optimize_resource_usage()
    
    async def _cleanup_old_failures(self):
        """Clean up old failure records"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Keep only recent failures in history
        self.failure_history = [
            f for f in self.failure_history
            if f.timestamp > cutoff_time
        ]
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical components"""
        critical_components = [
            'database', 'auth_service', 'core_engine', 
            'memory_store', 'llm_provider', 'security_validator'
        ]
        
        for component in critical_components:
            self.circuit_breakers[component] = CircuitBreakerState(
                name=component,
                failure_threshold=3,
                recovery_timeout=30.0
            )
    
    def _initialize_recovery_actions(self):
        """Initialize recovery action mappings"""
        # In a real implementation, these would be actual recovery functions
        pass
    
    # Placeholder methods for preventive measures
    async def _optimize_performance(self):
        """Optimize system performance"""
        self.logger.info("🚀 Optimizing performance")
    
    async def _enhance_error_handling(self):
        """Enhance error handling"""
        self.logger.info("🛡️ Enhancing error handling")
    
    async def _optimize_resource_usage(self):
        """Optimize resource usage"""
        self.logger.info("💾 Optimizing resource usage")
    
    @asynccontextmanager
    async def resilient_operation(self, component: str, operation_name: str = "operation"):
        """Context manager for resilient operations"""
        try:
            # Check circuit breaker
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                if breaker.state == "OPEN":
                    raise Exception(f"Circuit breaker is OPEN for {component}")
            
            yield
            
            # Record success
            if component in self.circuit_breakers:
                breaker = self.circuit_breakers[component]
                if breaker.state == "HALF_OPEN":
                    breaker.consecutive_successes += 1
                    if breaker.consecutive_successes >= breaker.success_threshold:
                        breaker.state = "CLOSED"
                        breaker.failure_count = 0
                        breaker.consecutive_successes = 0
            
        except Exception as e:
            # Handle failure
            await self.handle_failure(component, e, {'operation': operation_name})
            raise
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        active_failure_count = len(self.active_failures)
        total_failures_24h = len([
            f for f in self.failure_history
            if f.timestamp > datetime.now() - timedelta(hours=24)
        ])
        
        circuit_breaker_status = {
            name: breaker.state 
            for name, breaker in self.circuit_breakers.items()
        }
        
        health_status = {
            metric: self.health_metrics.is_healthy(metric)
            for metric in self.health_metrics.thresholds
        }
        
        return {
            "resilience_engine_report": {
                "timestamp": datetime.now().isoformat(),
                "active_failures": active_failure_count,
                "total_failures_24h": total_failures_24h,
                "circuit_breaker_status": circuit_breaker_status,
                "health_metrics_status": health_status,
                "recovery_tasks_running": len(self.recovery_tasks),
                "monitoring_active": self.monitoring_active,
                "overall_health": "HEALTHY" if active_failure_count == 0 else "DEGRADED"
            }
        }


# Global resilience functions
async def create_resilient_system(
    max_concurrent_recoveries: int = 5,
    enable_self_healing: bool = True
) -> AdvancedResilienceEngine:
    """Create and initialize resilient system"""
    engine = AdvancedResilienceEngine(
        max_concurrent_recoveries=max_concurrent_recoveries,
        enable_self_healing=enable_self_healing
    )
    await engine.initialize()
    return engine


async def with_resilience(
    component: str,
    operation: Callable,
    resilience_engine: AdvancedResilienceEngine,
    *args,
    **kwargs
):
    """Execute operation with resilience protection"""
    async with resilience_engine.resilient_operation(component):
        return await operation(*args, **kwargs)