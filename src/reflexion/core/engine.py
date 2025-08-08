"""Core reflexion engine implementation with advanced resilience."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .types import Reflection, ReflectionType, ReflexionResult
from .validation import validator
from .exceptions import (
    LLMError, ReflectionError, ValidationError, TimeoutError,
    RetryableError, SecurityError
)
from .logging_config import logger, metrics
from .health import health_checker, HealthStatus
from .retry import default_retry_manager, RetryConfig, with_retry
from .resilience import resilience_manager, ResiliencePattern


class LLMProvider:
    """Robust LLM provider interface with error handling and retries."""
    
    def __init__(self, model: str = "gpt-4", max_retries: int = 3, timeout: float = 30.0):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Validate model
        validation_result = validator.validate_llm_config(model)
        if not validation_result.is_valid:
            raise ValidationError(
                f"Invalid LLM model: {model}",
                validation_result.errors,
                validation_result.warnings
            )
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"LLM Config Warning: {warning}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM with robust error handling."""
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Validate and sanitize prompt
                validation_result = validator.validate_task(prompt)
                if not validation_result.is_valid:
                    raise SecurityError(
                        "Prompt failed security validation",
                        ", ".join(validation_result.errors),
                        prompt
                    )
                
                sanitized_prompt = validation_result.sanitized_input
                
                # Simulate API call with timeout
                result = await asyncio.wait_for(
                    self._simulate_llm_call(sanitized_prompt, **kwargs),
                    timeout=self.timeout
                )
                
                # Sanitize output
                result = validator.sanitize_output(result)
                
                execution_time = time.time() - start_time
                self.logger.info(
                    f"LLM generation successful - Model: {self.model}, "
                    f"Time: {execution_time:.2f}s, Attempt: {attempt + 1}"
                )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"LLM generation timeout after {self.timeout}s"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1}/{self.max_retries})")
                
                if attempt == self.max_retries - 1:
                    raise TimeoutError(error_msg, "llm_generation", self.timeout)
                
            except SecurityError:
                # Security errors should not be retried
                raise
                
            except Exception as e:
                error_msg = f"LLM generation failed: {str(e)}"
                self.logger.error(f"{error_msg} (attempt {attempt + 1}/{self.max_retries})")
                
                if attempt == self.max_retries - 1:
                    raise LLMError(error_msg, self.model, "generate", {"original_error": str(e)})
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous version with error handling."""
        try:
            # Validate and sanitize prompt
            validation_result = validator.validate_task(prompt)
            if not validation_result.is_valid:
                raise SecurityError(
                    "Prompt failed security validation",
                    ", ".join(validation_result.errors),
                    prompt
                )
            
            sanitized_prompt = validation_result.sanitized_input
            
            # Simple synchronous simulation
            if "error" in sanitized_prompt.lower():
                result = "Error: Could not complete the task due to unclear requirements."
            else:
                result = f"Task completed: {sanitized_prompt[:100]}..."
            
            # Sanitize output
            result = validator.sanitize_output(result)
            
            self.logger.info(f"LLM sync generation successful - Model: {self.model}")
            return result
            
        except SecurityError:
            raise
        except Exception as e:
            error_msg = f"LLM sync generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise LLMError(error_msg, self.model, "generate_sync", {"original_error": str(e)})
    
    async def _simulate_llm_call(self, prompt: str, **kwargs) -> str:
        """Simulate LLM API call."""
        # In production, this would make actual API calls
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Simulate occasional failures for testing
        import random
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated API failure")
        
        if "error" in prompt.lower():
            return "Error: Could not complete the task due to unclear requirements."
        
        return f"Task completed: {prompt[:100]}..."


class ReflexionEngine:
    """Core engine for reflexion-based task execution with advanced resilience."""

    def __init__(self, **config):
        """Initialize reflexion engine with enhanced resilience configuration."""
        self.config = config
        self.logger = logger
        self.metrics = metrics
        
        # Initialize resilience components
        self.health_checker = health_checker
        self.retry_manager = default_retry_manager
        self.resilience_manager = resilience_manager
        
        # Configure health monitoring
        if config.get('enable_health_checks', True):
            self.health_checker.register_custom_check(
                'reflexion_engine',
                self._health_check_callback
            )
        
        # Configure retry behavior
        self.retry_config = RetryConfig(
            max_attempts=config.get('max_retry_attempts', 3),
            base_delay=config.get('retry_base_delay', 1.0),
            max_delay=config.get('retry_max_delay', 30.0)
        )
        
        # Configuration validation
        self._validate_config(config)
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate engine configuration."""
        # Add any engine-specific configuration validation here
        self.logger.info(f"ReflexionEngine initialized with config: {list(config.keys())}")
    
    async def _health_check_callback(self):
        """Custom health check for reflexion engine."""
        from .health import HealthCheckResult, HealthStatus
        
        try:
            # Test basic functionality
            test_result = self._execute_task(
                "test health check",
                "gpt-4",
                0,
                []
            )
            
            if "Error" in test_result:
                return HealthCheckResult(
                    name="reflexion_engine",
                    status=HealthStatus.WARNING,
                    message="Engine responding but with errors"
                )
            else:
                return HealthCheckResult(
                    name="reflexion_engine", 
                    status=HealthStatus.HEALTHY,
                    message="Engine functioning normally"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name="reflexion_engine",
                status=HealthStatus.CRITICAL,
                message=f"Engine health check failed: {str(e)}"
            )

    async def execute_with_reflexion_async(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float,
        success_criteria: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Execute task with comprehensive reflexion loop and resilience patterns."""
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Apply resilience patterns
        try:
            return await self.resilience_manager.execute_with_resilience(
                self._execute_reflexion_core,
                f"reflexion_task_{execution_id}",
                patterns=[
                    ResiliencePattern.CIRCUIT_BREAKER,
                    ResiliencePattern.TIMEOUT,
                    ResiliencePattern.RATE_LIMITING
                ],
                fallback=self._fallback_execution,
                task=task,
                llm=llm,
                max_iterations=max_iterations,
                reflection_type=reflection_type,
                success_threshold=success_threshold,
                success_criteria=success_criteria,
                execution_id=execution_id,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Resilient execution failed for {execution_id}: {str(e)}")
            # Fallback to synchronous execution
            return self._execute_reflexion_core(
                task, llm, max_iterations, reflection_type, 
                success_threshold, success_criteria, execution_id, **kwargs
            )
    
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
        """Synchronous wrapper for reflexion execution with basic resilience."""
        try:
            # Try async execution with resilience if event loop available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create new task
                future = asyncio.ensure_future(
                    self.execute_with_reflexion_async(
                        task, llm, max_iterations, reflection_type,
                        success_threshold, success_criteria, **kwargs
                    )
                )
                return asyncio.get_event_loop().run_until_complete(future)
            else:
                return asyncio.run(
                    self.execute_with_reflexion_async(
                        task, llm, max_iterations, reflection_type,
                        success_threshold, success_criteria, **kwargs
                    )
                )
        except RuntimeError:
            # No event loop available, fall back to synchronous execution
            execution_id = f"exec_{int(time.time() * 1000)}"
            return self._execute_reflexion_core(
                task, llm, max_iterations, reflection_type,
                success_threshold, success_criteria, execution_id, **kwargs
            )

    def _execute_reflexion_core(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float,
        success_criteria: Optional[str] = None,
        execution_id: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Core reflexion execution with comprehensive error handling."""
        execution_id = execution_id or f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        reflections: List[Reflection] = []
        current_output = ""
        final_success = False
        evaluation = None
        
        self.logger.info(
            f"Starting reflexion execution {execution_id} - Task: {task[:50]}..., "
            f"LLM: {llm}, Max iterations: {max_iterations}"
        )
        
        try:
            # Validate all inputs
            self._validate_execution_params(
                task, llm, max_iterations, reflection_type, success_threshold
            )
            
            # Main execution loop with error handling
            for iteration in range(max_iterations):
                iteration_start = time.time()
                
                try:
                    self.logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
                    
                    # Execute task
                    current_output = self._execute_task(task, llm, iteration, reflections)
                    
                    # Evaluate result
                    evaluation = self._evaluate_output(task, current_output, success_criteria)
                    
                    iteration_time = time.time() - iteration_start
                    self.logger.info(
                        f"Iteration {iteration + 1} completed - Success: {evaluation['success']}, "
                        f"Score: {evaluation['score']:.2f}, Time: {iteration_time:.2f}s"
                    )
                    
                    # Check for success
                    if evaluation["success"] and evaluation["score"] >= success_threshold:
                        final_success = True
                        self.logger.info(f"Task succeeded after {iteration + 1} iterations")
                        break
                    
                    # Generate reflection if not the last iteration
                    if iteration < max_iterations - 1:
                        reflection = self._generate_reflection(
                            task, current_output, evaluation, reflection_type, iteration
                        )
                        reflections.append(reflection)
                        
                        self.logger.info(
                            f"Reflection generated - Issues: {len(reflection.issues)}, "
                            f"Improvements: {len(reflection.improvements)}, "
                            f"Confidence: {reflection.confidence:.2f}"
                        )
                
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                    
                    # Create error reflection
                    error_reflection = self._create_error_reflection(task, str(e), iteration)
                    reflections.append(error_reflection)
                    
                    # Don't break on retryable errors unless max iterations reached
                    if isinstance(e, RetryableError) and e.can_retry and iteration < max_iterations - 1:
                        continue
                    else:
                        current_output = f"Error: {str(e)}"
                        break
            
            # Calculate final metrics
            total_time = time.time() - start_time
            final_evaluation = evaluation or {"success": False, "score": 0.0}
            
            # Record metrics
            self.metrics.record_task_execution(
                success=final_success,
                iterations=len(reflections) + 1,
                reflections=len(reflections),
                execution_time=total_time,
                task_type=self._classify_task(task)
            )
            
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
                    "final_score": final_evaluation.get("score", 0.0)
                }
            )
            
            self.logger.info(
                f"Execution {execution_id} completed - Success: {final_success}, "
                f"Total time: {total_time:.2f}s, Iterations: {len(reflections) + 1}"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"Critical error in execution {execution_id}: {str(e)}")
            
            # Record failed execution in metrics
            self.metrics.record_task_execution(
                success=False,
                iterations=len(reflections) + 1,
                reflections=len(reflections),
                execution_time=total_time,
                task_type=self._classify_task(task)
            )
            
            # Re-raise the exception
            raise
    
    def _validate_execution_params(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float
    ):
        """Validate execution parameters."""
        # Validate task
        task_validation = validator.validate_task(task)
        if not task_validation.is_valid:
            raise ValidationError(
                "Task validation failed",
                task_validation.errors,
                task_validation.warnings
            )
        
        # Validate LLM config
        llm_validation = validator.validate_llm_config(llm)
        if not llm_validation.is_valid:
            raise ValidationError(
                "LLM configuration validation failed",
                llm_validation.errors,
                llm_validation.warnings
            )
        
        # Validate reflexion params
        params_validation = validator.validate_reflexion_params(
            max_iterations, success_threshold, reflection_type.value
        )
        if not params_validation.is_valid:
            raise ValidationError(
                "Reflexion parameters validation failed",
                params_validation.errors,
                params_validation.warnings
            )
        
        # Log warnings
        for validation in [task_validation, llm_validation, params_validation]:
            for warning in validation.warnings:
                self.logger.warning(f"Validation warning: {warning}")

    def _execute_task(self, task: str, llm: str, iteration: int, reflections: List[Reflection]) -> str:
        """Execute task with LLM."""
        provider = LLMProvider(llm)
        
        # Build prompt with reflection context if available
        prompt = self._build_task_prompt(task, reflections, iteration)
        
        try:
            return provider.generate_sync(prompt)
        except Exception as e:
            logging.error(f"LLM execution failed: {e}")
            return f"Error executing task: {task}"
    
    def _build_task_prompt(self, task: str, reflections: List[Reflection], iteration: int) -> str:
        """Build enhanced prompt with reflection context."""
        from ..prompts import ReflectionPrompts, PromptDomain
        
        base_prompt = f"Task: {task}\n\nPlease provide a complete solution."
        
        if reflections:
            latest_reflection = reflections[-1]
            improvement_context = "\n".join(latest_reflection.improvements)
            issues_context = "\n".join(latest_reflection.issues)
            
            # Determine domain from task content
            domain = self._infer_domain_from_task(task)
            
            # Use domain-specific improvement prompts
            improvement_prompt = ReflectionPrompts.build_improvement_prompt(
                domain, task, latest_reflection.issues, latest_reflection.improvements
            )
            
            base_prompt += f"\n\nPrevious attempt (iteration {iteration}) failed with issues:\n{issues_context}"
            base_prompt += f"\n\nPlease improve by:\n{improvement_context}"
            base_prompt += f"\n\nDomain-specific guidance:\n{improvement_prompt}"
        
        return base_prompt
    
    def _infer_domain_from_task(self, task: str) -> 'PromptDomain':
        """Infer the domain from task content."""
        from ..prompts import PromptDomain
        
        task_lower = task.lower()
        
        # Software engineering keywords
        code_keywords = ['code', 'function', 'class', 'implement', 'debug', 'algorithm', 'program', 'script']
        if any(keyword in task_lower for keyword in code_keywords):
            return PromptDomain.SOFTWARE_ENGINEERING
        
        # Data analysis keywords
        data_keywords = ['data', 'analyze', 'statistics', 'chart', 'graph', 'dataset', 'analysis']
        if any(keyword in task_lower for keyword in data_keywords):
            return PromptDomain.DATA_ANALYSIS
        
        # Creative writing keywords
        creative_keywords = ['write', 'story', 'creative', 'narrative', 'poem', 'article', 'content']
        if any(keyword in task_lower for keyword in creative_keywords):
            return PromptDomain.CREATIVE_WRITING
        
        # Research keywords
        research_keywords = ['research', 'investigate', 'study', 'explore', 'examine', 'review']
        if any(keyword in task_lower for keyword in research_keywords):
            return PromptDomain.RESEARCH
        
        return PromptDomain.GENERAL

    def _evaluate_output(self, task: str, output: str, criteria: Optional[str]) -> Dict[str, Any]:
        """Evaluate task output using multiple heuristics."""
        success_indicators = [
            len(output) > 30,
            "completed" in output.lower() or "solution" in output.lower(),
            not output.lower().startswith("error"),
            "task" in output.lower()
        ]
        
        success_count = sum(success_indicators)
        success = success_count >= 3
        score = success_count / len(success_indicators)
        
        # Apply custom criteria if provided
        if criteria and success:
            criteria_met = any(criterion.lower() in output.lower() 
                             for criterion in criteria.split(","))
            if not criteria_met:
                success = False
                score *= 0.5
        
        return {
            "success": success,
            "score": score,
            "details": {
                "length": len(output),
                "indicators_met": success_count,
                "criteria_met": criteria is None or success,
                "has_error": "error" in output.lower()
            }
        }

    def _generate_reflection(
        self, task: str, output: str, evaluation: Dict[str, Any], 
        reflection_type: ReflectionType, iteration: int
    ) -> Reflection:
        """Generate comprehensive reflection with enhanced analysis."""
        try:
            issues = []
            improvements = []
            
            details = evaluation.get("details", {})
            
            # Analyze specific failure modes
            if details.get("has_error", False):
                issues.append("Output contains error messages")
                improvements.append("Address the underlying error conditions")
                improvements.append("Provide error handling and validation")
            
            if details.get("length", 0) < 30:
                issues.append("Output is too brief and lacks detail")
                improvements.append("Provide more comprehensive explanation")
                improvements.append("Include step-by-step implementation details")
            
            if not details.get("criteria_met", True):
                issues.append("Custom success criteria not satisfied")
                improvements.append("Ensure all specified requirements are addressed")
            
            if details.get("indicators_met", 0) < 2:
                issues.append("Output lacks clear completion indicators")
                improvements.append("Clearly state when task is completed")
                improvements.append("Use explicit success language")
            
            # Add iteration-specific insights
            if iteration > 0:
                issues.append(f"Previous attempt (iteration {iteration}) also failed")
                improvements.append("Try a fundamentally different approach")
                improvements.append("Consider breaking down the task into smaller steps")
            
            # Reflection type-specific analysis
            if reflection_type == ReflectionType.STRUCTURED:
                # Add structured analysis dimensions
                issues.append("Multi-dimensional analysis needed")
                improvements.append("Consider correctness, efficiency, and maintainability")
            
            # Calculate confidence based on multiple factors
            base_confidence = 0.4
            issue_factor = min(len(issues) * 0.1, 0.3)
            improvement_factor = min(len(improvements) * 0.08, 0.2)
            iteration_penalty = min(iteration * 0.05, 0.15)
            
            confidence = min(0.95, base_confidence + issue_factor + improvement_factor - iteration_penalty)
            
            return Reflection(
                task=task,
                output=output,
                success=evaluation["success"],
                score=evaluation["score"],
                issues=issues,
                improvements=improvements,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating reflection: {str(e)}")
            raise ReflectionError(
                f"Failed to generate reflection: {str(e)}",
                task, iteration, {"original_error": str(e)}
            )
    
    def _create_error_reflection(self, task: str, error: str, iteration: int) -> Reflection:
        """Create reflection for error scenarios."""
        return Reflection(
            task=task,
            output=f"Error occurred: {error}",
            success=False,
            score=0.0,
            issues=[f"Execution error: {error}", "System failure during processing"],
            improvements=[
                "Investigate and fix the underlying system issue",
                "Add proper error handling and recovery mechanisms",
                "Consider alternative approaches or fallback strategies"
            ],
            confidence=0.8,  # High confidence in error identification
            timestamp=datetime.now().isoformat()
        )
    
    def _classify_task(self, task: str) -> str:
        """Classify task type for metrics."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['write', 'create', 'implement', 'build']):
            return 'creation'
        elif any(word in task_lower for word in ['analyze', 'review', 'examine', 'evaluate']):
            return 'analysis'
        elif any(word in task_lower for word in ['fix', 'debug', 'correct', 'repair']):
            return 'debugging'
        elif any(word in task_lower for word in ['optimize', 'improve', 'enhance', 'refactor']):
            return 'optimization'
        else:
            return 'general'
    
    def _fallback_execution(
        self,
        task: str,
        llm: str,
        max_iterations: int,
        reflection_type: ReflectionType,
        success_threshold: float,
        success_criteria: Optional[str] = None,
        execution_id: Optional[str] = None,
        **kwargs
    ) -> ReflexionResult:
        """Fallback execution with reduced functionality but guaranteed response."""
        execution_id = execution_id or f"fallback_{int(time.time() * 1000)}"
        start_time = time.time()
        
        self.logger.warning(f"Using fallback execution for {execution_id}")
        
        try:
            # Simplified execution with minimal reflexion
            simple_output = f"Fallback response for task: {task[:100]}. " \
                          f"Due to system constraints, providing simplified response."
            
            # Create minimal reflection
            fallback_reflection = Reflection(
                task=task,
                output=simple_output,
                success=False,  # Mark as not fully successful
                score=0.4,  # Low score to indicate fallback
                issues=["System constraints triggered fallback mode"],
                improvements=["Retry when system resources are available"],
                confidence=0.3,  # Low confidence in fallback response
                timestamp=datetime.now().isoformat()
            )
            
            result = ReflexionResult(
                task=task,
                output=simple_output,
                success=False,
                iterations=1,
                reflections=[fallback_reflection],
                total_time=time.time() - start_time,
                metadata={
                    "execution_mode": "fallback",
                    "execution_id": execution_id,
                    "reason": "system_constraints"
                }
            )
            
            self.logger.info(f"Fallback execution {execution_id} completed")
            return result
            
        except Exception as e:
            # Even fallback failed, return minimal error response
            self.logger.error(f"Fallback execution also failed: {str(e)}")
            
            return ReflexionResult(
                task=task,
                output=f"System error: Unable to process task due to {str(e)}",
                success=False,
                iterations=0,
                reflections=[],
                total_time=time.time() - start_time,
                metadata={
                    "execution_mode": "error_fallback",
                    "execution_id": execution_id,
                    "error": str(e)
                }
            )
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        try:
            # Run health checks
            health_results = await self.health_checker.run_all_checks()
            overall_status = self.health_checker.get_overall_status(health_results)
            
            # Get metrics
            system_metrics = self.health_checker.get_system_metrics()
            resilience_metrics = self.resilience_manager.get_resilience_metrics()
            retry_stats = self.retry_manager.get_retry_stats()
            
            return {
                "overall_status": overall_status.value,
                "health_checks": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "duration_ms": result.duration_ms
                    }
                    for name, result in health_results.items()
                },
                "system_metrics": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_percent,
                    "uptime_seconds": system_metrics.uptime_seconds
                },
                "resilience_metrics": resilience_metrics,
                "retry_stats": retry_stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system health: {str(e)}")
            return {
                "overall_status": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }