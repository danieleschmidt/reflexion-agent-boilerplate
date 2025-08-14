"""
Comprehensive tests for advanced error recovery system.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.reflexion.core.advanced_error_recovery import (
    AdvancedErrorRecoveryManager, ErrorContext, ErrorSeverity, RecoveryStrategy,
    CircuitBreaker, AdaptiveTimeout, SelfHealingSystem, RecoveryMetrics
)
from src.reflexion.core.exceptions import (
    LLMError, ValidationError, TimeoutError, SecurityError, RetryableError
)


class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={"key": "value"},
            timestamp=datetime.now()
        )
        
        assert context.error == error
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.context == {"key": "value"}
        assert context.recovery_attempts == 0
    
    def test_is_recoverable_within_attempts(self):
        """Test recoverable check within attempt limits."""
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now(),
            max_recovery_attempts=3
        )
        
        assert context.is_recoverable() is True
        
        context.recovery_attempts = 2
        assert context.is_recoverable() is True
        
        context.recovery_attempts = 3
        assert context.is_recoverable() is False
    
    def test_is_recoverable_critical_severity(self):
        """Test recoverable check for critical errors."""
        error = ValueError("Critical error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.CRITICAL,
            context={},
            timestamp=datetime.now()
        )
        
        assert context.is_recoverable() is False
    
    def test_is_recoverable_security_error(self):
        """Test recoverable check for security errors."""
        error = SecurityError("Security violation", "test_context", "test_input")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        assert context.is_recoverable() is False
    
    def test_get_error_signature(self):
        """Test error signature generation."""
        error = ValueError("Test error message")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        signature = context.get_error_signature()
        
        assert "ValueError:" in signature
        assert isinstance(signature, str)


class TestRecoveryMetrics:
    """Test recovery metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = RecoveryMetrics()
        
        assert metrics.total_errors == 0
        assert metrics.recoverable_errors == 0
        assert metrics.successful_recoveries == 0
        assert metrics.recovery_time_total == 0.0
    
    def test_add_error(self):
        """Test adding error to metrics."""
        metrics = RecoveryMetrics()
        error = ValueError("Test")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        metrics.add_error(context)
        
        assert metrics.total_errors == 1
        assert metrics.recoverable_errors == 1
        
        signature = context.get_error_signature()
        assert metrics.error_patterns[signature] == 1
    
    def test_add_recovery(self):
        """Test adding recovery to metrics."""
        metrics = RecoveryMetrics()
        
        metrics.add_recovery(RecoveryStrategy.RETRY, 1.5, True)
        
        assert metrics.recovery_strategies_used[RecoveryStrategy.RETRY] == 1
        assert metrics.recovery_time_total == 1.5
        assert metrics.successful_recoveries == 1
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        metrics = RecoveryMetrics()
        
        # No recoverable errors
        assert metrics.get_success_rate() == 0.0
        
        # Some recoverable errors
        metrics.recoverable_errors = 5
        metrics.successful_recoveries = 3
        
        assert metrics.get_success_rate() == 0.6
    
    def test_get_avg_recovery_time(self):
        """Test average recovery time calculation."""
        metrics = RecoveryMetrics()
        
        # No attempts
        assert metrics.get_avg_recovery_time() == 0.0
        
        # Some attempts
        metrics.recovery_strategies_used[RecoveryStrategy.RETRY] = 2
        metrics.recovery_strategies_used[RecoveryStrategy.FALLBACK] = 1
        metrics.recovery_time_total = 6.0
        
        assert metrics.get_avg_recovery_time() == 2.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        return CircuitBreaker(
            failure_threshold=3,
            timeout_duration=60.0,
            recovery_timeout=300.0
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state."""
        async def mock_function():
            return "success"
        
        result = await circuit_breaker.call(mock_function)
        
        assert result == "success"
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self, circuit_breaker):
        """Test circuit breaker failure counting."""
        async def failing_function():
            raise ValueError("Function failed")
        
        # First few failures should not open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_function)
            
            assert circuit_breaker.state == "closed"
            assert circuit_breaker.failure_count == i + 1
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker in open state."""
        # Force circuit breaker to open
        circuit_breaker.state = "open"
        circuit_breaker.last_failure_time = datetime.now()
        
        async def mock_function():
            return "success"
        
        with pytest.raises(Exception) as exc_info:
            await circuit_breaker.call(mock_function)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success(self, circuit_breaker):
        """Test circuit breaker half-open state success."""
        # Set up half-open state
        circuit_breaker.state = "half_open"
        
        async def mock_function():
            return "success"
        
        result = await circuit_breaker.call(mock_function)
        
        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    def test_should_attempt_reset(self, circuit_breaker):
        """Test should attempt reset logic."""
        # No previous failure
        assert circuit_breaker._should_attempt_reset() is False
        
        # Recent failure
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=100)
        assert circuit_breaker._should_attempt_reset() is False
        
        # Old failure
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=400)
        assert circuit_breaker._should_attempt_reset() is True


class TestAdaptiveTimeout:
    """Test adaptive timeout functionality."""
    
    @pytest.fixture
    def adaptive_timeout(self):
        """Create adaptive timeout for testing."""
        return AdaptiveTimeout(initial_timeout=10.0, max_timeout=60.0)
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_success(self, adaptive_timeout):
        """Test adaptive timeout with successful execution."""
        async def fast_function():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await adaptive_timeout.execute_with_timeout(fast_function)
        
        assert result == "success"
        assert len(adaptive_timeout.success_times) == 1
        assert adaptive_timeout.success_times[0] > 0.1
    
    @pytest.mark.asyncio
    async def test_adaptive_timeout_failure(self, adaptive_timeout):
        """Test adaptive timeout with timeout."""
        async def slow_function():
            await asyncio.sleep(20)  # Longer than initial timeout
            return "success"
        
        with pytest.raises(TimeoutError):
            await adaptive_timeout.execute_with_timeout(slow_function)
        
        assert adaptive_timeout.timeout_count == 1
        assert adaptive_timeout.current_timeout > adaptive_timeout.initial_timeout
    
    def test_adapt_timeout_increase(self, adaptive_timeout):
        """Test timeout adaptation after successful executions."""
        # Add some execution times
        adaptive_timeout.success_times.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        
        original_timeout = adaptive_timeout.current_timeout
        adaptive_timeout._adapt_timeout()
        
        # Timeout should be adjusted based on execution patterns
        assert adaptive_timeout.current_timeout >= adaptive_timeout.initial_timeout


class TestSelfHealingSystem:
    """Test self-healing system functionality."""
    
    @pytest.fixture
    def self_healing(self):
        """Create self-healing system for testing."""
        return SelfHealingSystem()
    
    def test_register_healer(self, self_healing):
        """Test registering custom healer."""
        def custom_healer(error_context):
            return True
        
        self_healing.register_healer("custom_pattern", custom_healer)
        
        assert "custom_pattern" in self_healing.healing_strategies
        assert self_healing.healing_strategies["custom_pattern"] == custom_healer
    
    @pytest.mark.asyncio
    async def test_attempt_healing_success(self, self_healing):
        """Test successful healing attempt."""
        error = ConnectionError("Network error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        # Mock healing strategy
        with patch.object(self_healing, '_heal_connection_failure', return_value=True) as mock_heal:
            result = await self_healing.attempt_healing(context)
            
            assert result is True
            mock_heal.assert_called_once()
            assert len(self_healing.healing_history) == 1
            assert self_healing.healing_history[0]["success"] is True
    
    @pytest.mark.asyncio
    async def test_attempt_healing_failure(self, self_healing):
        """Test failed healing attempt."""
        error = ValueError("Generic error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        # No matching healing pattern
        result = await self_healing.attempt_healing(context)
        
        assert result is False
    
    def test_identify_healing_pattern_memory(self, self_healing):
        """Test identifying memory pressure pattern."""
        error = MemoryError("Out of memory")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        pattern = self_healing._identify_healing_pattern(context)
        assert pattern == "memory_pressure"
    
    def test_identify_healing_pattern_connection(self, self_healing):
        """Test identifying connection failure pattern."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        pattern = self_healing._identify_healing_pattern(context)
        assert pattern == "connection_failure"
    
    def test_identify_healing_pattern_timeout(self, self_healing):
        """Test identifying timeout pattern."""
        error = TimeoutError("Operation timed out", "test_context", 30.0)
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        pattern = self_healing._identify_healing_pattern(context)
        assert pattern == "timeout_cascade"
    
    @pytest.mark.asyncio
    async def test_heal_memory_pressure(self, self_healing):
        """Test memory pressure healing."""
        error = MemoryError("Out of memory")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        result = await self_healing._heal_memory_pressure(context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_heal_connection_failure(self, self_healing):
        """Test connection failure healing."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await self_healing._heal_connection_failure(context)
            assert result is True
    
    def test_get_healing_report(self, self_healing):
        """Test healing report generation."""
        # Add some healing history
        self_healing.healing_history.append({
            "pattern": "memory_pressure",
            "error_signature": "MemoryError:123",
            "success": True,
            "healing_time": 0.5,
            "timestamp": datetime.now().isoformat()
        })
        
        self_healing.healing_history.append({
            "pattern": "connection_failure", 
            "error_signature": "ConnectionError:456",
            "success": False,
            "healing_time": 1.0,
            "timestamp": datetime.now().isoformat()
        })
        
        report = self_healing.get_healing_report()
        
        assert "overall_statistics" in report
        assert "pattern_effectiveness" in report
        assert "recent_healing_activity" in report
        
        assert report["overall_statistics"]["total_healing_attempts"] == 2
        assert report["overall_statistics"]["successful_healings"] == 1


class TestAdvancedErrorRecoveryManager:
    """Test advanced error recovery manager functionality."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create recovery manager for testing."""
        return AdvancedErrorRecoveryManager()
    
    def test_assess_error_severity(self, recovery_manager):
        """Test error severity assessment."""
        # Security error
        security_error = SecurityError("Security violation", "test", "input")
        severity = recovery_manager._assess_error_severity(security_error)
        assert severity == ErrorSeverity.CRITICAL
        
        # LLM error
        llm_error = LLMError("LLM failed", "gpt-4", "generate", {})
        severity = recovery_manager._assess_error_severity(llm_error)
        assert severity == ErrorSeverity.HIGH
        
        # Validation error
        validation_error = ValidationError("Invalid input", [], [])
        severity = recovery_manager._assess_error_severity(validation_error)
        assert severity == ErrorSeverity.MEDIUM
        
        # Generic error
        generic_error = ValueError("Generic error")
        severity = recovery_manager._assess_error_severity(generic_error)
        assert severity == ErrorSeverity.LOW
    
    def test_select_recovery_strategy_first_attempt(self, recovery_manager):
        """Test recovery strategy selection for first attempt."""
        timeout_error = TimeoutError("Timeout", "test", 30.0)
        context = ErrorContext(
            error=timeout_error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now(),
            recovery_attempts=0
        )
        
        strategy = recovery_manager._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.ADAPTIVE_TIMEOUT
        
        connection_error = ConnectionError("Connection failed")
        context.error = connection_error
        
        strategy = recovery_manager._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.RETRY
    
    def test_select_recovery_strategy_second_attempt(self, recovery_manager):
        """Test recovery strategy selection for second attempt."""
        error = LLMError("LLM failed", "gpt-4", "generate", {})
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now(),
            recovery_attempts=1
        )
        
        strategy = recovery_manager._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.CIRCUIT_BREAKER
    
    def test_select_recovery_strategy_final_attempt(self, recovery_manager):
        """Test recovery strategy selection for final attempts."""
        error = ValueError("Generic error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now(),
            recovery_attempts=2
        )
        
        strategy = recovery_manager._select_recovery_strategy(context)
        assert strategy == RecoveryStrategy.GRACEFUL_DEGRADATION
    
    @pytest.mark.asyncio
    async def test_retry_strategy(self, recovery_manager):
        """Test retry recovery strategy."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now(),
            recovery_attempts=1
        )
        
        async def mock_operation():
            return "success"
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await recovery_manager._retry_strategy(context, mock_operation)
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_fallback_strategy(self, recovery_manager):
        """Test fallback recovery strategy."""
        error = ValueError("Operation failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        async def mock_operation():
            return "primary"
        
        result = await recovery_manager._fallback_strategy(context, mock_operation)
        
        assert "error" in result
        assert result["fallback"] is True
        assert "original_error" in result
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_with_custom_fallback(self, recovery_manager):
        """Test fallback strategy with custom fallback operation."""
        error = ValueError("Operation failed")
        
        async def fallback_operation():
            return "fallback_result"
        
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={"fallback_operation": fallback_operation},
            timestamp=datetime.now()
        )
        
        async def mock_operation():
            return "primary"
        
        result = await recovery_manager._fallback_strategy(context, mock_operation)
        assert result == "fallback_result"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_strategy(self, recovery_manager):
        """Test graceful degradation strategy."""
        error = ValueError("Operation failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.HIGH,
            context={},
            timestamp=datetime.now()
        )
        
        async def mock_operation():
            return "primary"
        
        result = await recovery_manager._graceful_degradation_strategy(context, mock_operation)
        
        assert result["degraded"] is True
        assert "limited_functionality" in result
        assert result["limited_functionality"] is True
    
    @pytest.mark.asyncio
    async def test_self_healing_strategy_success(self, recovery_manager):
        """Test self-healing strategy with successful healing."""
        error = ConnectionError("Connection failed")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        async def mock_operation():
            return "success"
        
        with patch.object(recovery_manager.self_healing, 'attempt_healing', new_callable=AsyncMock, return_value=True):
            result = await recovery_manager._self_healing_strategy(context, mock_operation)
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_self_healing_strategy_failure(self, recovery_manager):
        """Test self-healing strategy with failed healing."""
        error = ValueError("Generic error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        
        async def mock_operation():
            return "success"
        
        with patch.object(recovery_manager.self_healing, 'attempt_healing', new_callable=AsyncMock, return_value=False):
            with pytest.raises(ValueError):
                await recovery_manager._self_healing_strategy(context, mock_operation)
    
    @pytest.mark.asyncio
    async def test_handle_error_recovery_success(self, recovery_manager):
        """Test successful error handling and recovery."""
        error = ConnectionError("Connection failed")
        
        async def mock_operation():
            return "success"
        
        with patch.object(recovery_manager, '_attempt_recovery', new_callable=AsyncMock, return_value="recovered"):
            result = await recovery_manager.handle_error(
                error=error,
                operation=mock_operation,
                context={}
            )
            
            assert result == "recovered"
            assert recovery_manager.recovery_metrics.total_errors == 1
    
    @pytest.mark.asyncio
    async def test_handle_error_non_recoverable(self, recovery_manager):
        """Test handling non-recoverable errors."""
        error = SecurityError("Security violation", "test", "input")
        
        async def mock_operation():
            return "success"
        
        with pytest.raises(SecurityError):
            await recovery_manager.handle_error(
                error=error,
                operation=mock_operation,
                context={}
            )
        
        assert recovery_manager.recovery_metrics.total_errors == 1
    
    def test_get_recovery_report(self, recovery_manager):
        """Test recovery report generation."""
        # Add some metrics
        recovery_manager.recovery_metrics.total_errors = 10
        recovery_manager.recovery_metrics.successful_recoveries = 7
        recovery_manager.recovery_metrics.recovery_strategies_used[RecoveryStrategy.RETRY] = 5
        
        # Add error history
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            context={},
            timestamp=datetime.now()
        )
        recovery_manager.error_history.append(context)
        
        report = recovery_manager.get_recovery_report()
        
        assert "recovery_metrics" in report
        assert "strategy_usage" in report
        assert "circuit_breaker" in report
        assert "adaptive_timeout" in report
        assert "self_healing" in report
        assert "recent_errors" in report
        
        assert report["recovery_metrics"]["total_errors"] == 10
        assert report["recovery_metrics"]["successful_recoveries"] == 7


if __name__ == "__main__":
    pytest.main([__file__])