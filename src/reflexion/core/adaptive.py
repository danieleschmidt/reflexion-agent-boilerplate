"""Self-adaptive performance optimization and machine learning-driven improvements."""

import asyncio
import time
import threading
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta

from .types import ReflexionResult
from .optimization import OptimizationManager
from .scaling import ScalingManager


class AdaptiveStrategy(Enum):
    """Adaptive optimization strategies."""
    PERFORMANCE_TUNING = "performance_tuning"
    RESOURCE_ALLOCATION = "resource_allocation"
    CACHING_OPTIMIZATION = "caching_optimization"
    LOAD_PREDICTION = "load_prediction"
    FAILURE_PREDICTION = "failure_prediction"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class PerformanceMetric:
    """Performance metric with historical context."""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    trend: Optional[str] = None  # "improving", "degrading", "stable"
    significance: float = 0.0  # Statistical significance of changes


@dataclass
class AdaptationDecision:
    """Record of adaptive optimization decision."""
    timestamp: float
    strategy: AdaptiveStrategy
    decision: str
    confidence: float
    expected_improvement: float
    applied: bool = False
    actual_improvement: Optional[float] = None
    rollback_trigger: Optional[str] = None


class PredictiveModel:
    """Simple predictive model for performance optimization."""
    
    def __init__(self, window_size: int = 100, min_samples: int = 20):
        self.window_size = window_size
        self.min_samples = min_samples
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.models: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_sample(self, metric_name: str, value: float, features: Dict[str, float] = None):
        """Add sample to training data."""
        sample = {
            "timestamp": time.time(),
            "value": value,
            "features": features or {}
        }
        self.historical_data[metric_name].append(sample)
        
        # Update model if we have enough samples
        if len(self.historical_data[metric_name]) >= self.min_samples:
            self._update_model(metric_name)
    
    def predict(self, metric_name: str, features: Dict[str, float] = None, horizon: int = 1) -> Optional[float]:
        """Predict future metric value."""
        if metric_name not in self.models:
            return None
        
        model = self.models[metric_name]
        if model["type"] == "linear_trend":
            return self._predict_linear_trend(model, horizon)
        elif model["type"] == "moving_average":
            return self._predict_moving_average(model, horizon)
        elif model["type"] == "exponential_smoothing":
            return self._predict_exponential_smoothing(model, horizon)
        
        return None
    
    def _update_model(self, metric_name: str):
        """Update predictive model for metric."""
        data = list(self.historical_data[metric_name])
        if len(data) < self.min_samples:
            return
        
        values = [sample["value"] for sample in data]
        timestamps = [sample["timestamp"] for sample in data]
        
        # Choose best model based on data characteristics
        model_performance = {}
        
        # Test linear trend model
        trend_model = self._fit_linear_trend(timestamps, values)
        trend_error = self._calculate_prediction_error(trend_model, timestamps[-10:], values[-10:])
        model_performance["linear_trend"] = {"model": trend_model, "error": trend_error}
        
        # Test moving average model
        ma_model = self._fit_moving_average(values)
        ma_error = self._calculate_prediction_error(ma_model, timestamps[-10:], values[-10:])
        model_performance["moving_average"] = {"model": ma_model, "error": ma_error}
        
        # Test exponential smoothing
        exp_model = self._fit_exponential_smoothing(values)
        exp_error = self._calculate_prediction_error(exp_model, timestamps[-10:], values[-10:])
        model_performance["exponential_smoothing"] = {"model": exp_model, "error": exp_error}
        
        # Select best model
        best_model_type = min(model_performance.keys(), key=lambda k: model_performance[k]["error"])
        self.models[metric_name] = model_performance[best_model_type]["model"]
        self.models[metric_name]["type"] = best_model_type
        
        self.logger.debug(f"Updated model for {metric_name}: {best_model_type} (error: {model_performance[best_model_type]['error']:.4f})")
    
    def _fit_linear_trend(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """Fit linear trend model."""
        if len(timestamps) < 2:
            return {"type": "linear_trend", "slope": 0, "intercept": values[0] if values else 0}
        
        # Simple linear regression
        x = np.array(timestamps)
        y = np.array(values)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        return {
            "type": "linear_trend",
            "slope": slope,
            "intercept": intercept,
            "last_timestamp": timestamps[-1]
        }
    
    def _fit_moving_average(self, values: List[float], window: int = 10) -> Dict[str, Any]:
        """Fit moving average model."""
        if len(values) < window:
            window = len(values)
        
        recent_values = values[-window:]
        avg = np.mean(recent_values)
        std = np.std(recent_values) if len(recent_values) > 1 else 0
        
        return {
            "type": "moving_average",
            "average": avg,
            "std": std,
            "window": window
        }
    
    def _fit_exponential_smoothing(self, values: List[float], alpha: float = 0.3) -> Dict[str, Any]:
        """Fit exponential smoothing model."""
        if len(values) < 2:
            return {"type": "exponential_smoothing", "level": values[0] if values else 0, "alpha": alpha}
        
        level = values[0]
        for value in values[1:]:
            level = alpha * value + (1 - alpha) * level
        
        return {
            "type": "exponential_smoothing",
            "level": level,
            "alpha": alpha
        }
    
    def _predict_linear_trend(self, model: Dict[str, Any], horizon: int) -> float:
        """Predict using linear trend model."""
        future_timestamp = model["last_timestamp"] + horizon * 60  # 1 minute intervals
        return model["slope"] * future_timestamp + model["intercept"]
    
    def _predict_moving_average(self, model: Dict[str, Any], horizon: int) -> float:
        """Predict using moving average model."""
        return model["average"]  # Moving average assumes stability
    
    def _predict_exponential_smoothing(self, model: Dict[str, Any], horizon: int) -> float:
        """Predict using exponential smoothing model."""
        return model["level"]  # Simple exponential smoothing for level prediction
    
    def _calculate_prediction_error(self, model: Dict[str, Any], test_timestamps: List[float], test_values: List[float]) -> float:
        """Calculate prediction error for model evaluation."""
        if not test_values:
            return float('inf')
        
        predictions = []
        for i, timestamp in enumerate(test_timestamps):
            if model["type"] == "linear_trend":
                pred = model["slope"] * timestamp + model["intercept"]
            elif model["type"] == "moving_average":
                pred = model["average"]
            elif model["type"] == "exponential_smoothing":
                pred = model["level"]
            else:
                pred = 0
            
            predictions.append(pred)
        
        # Calculate mean absolute error
        errors = [abs(pred - actual) for pred, actual in zip(predictions, test_values)]
        return np.mean(errors) if errors else float('inf')


class AdaptiveOptimizer:
    """Self-adaptive performance optimization system with ML-driven insights."""
    
    def __init__(
        self,
        optimization_manager: OptimizationManager,
        scaling_manager: ScalingManager,
        adaptation_interval: float = 60.0,  # seconds
        sensitivity: float = 0.1  # threshold for triggering adaptations
    ):
        self.optimization_manager = optimization_manager
        self.scaling_manager = scaling_manager
        self.adaptation_interval = adaptation_interval
        self.sensitivity = sensitivity
        
        # Predictive models for different metrics
        self.predictive_model = PredictiveModel(window_size=200, min_samples=30)
        
        # Historical performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.adaptation_history: List[AdaptationDecision] = []
        
        # Current optimization state
        self.current_optimizations: Dict[str, Any] = {}
        self.adaptation_counters: Dict[AdaptiveStrategy, int] = defaultdict(int)
        
        # Adaptive parameters
        self.cache_hit_rate_target = 0.8
        self.response_time_target = 1.0  # seconds
        self.resource_utilization_target = 0.7
        self.error_rate_threshold = 0.05
        
        # Control loop state
        self.adaptation_active = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def record_performance_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """Record performance metric for adaptive optimization."""
        timestamp = time.time()
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Add to history
        self.performance_history[name].append(metric)
        
        # Update predictive model
        features = self._extract_features(metadata or {})
        self.predictive_model.add_sample(name, value, features)
        
        # Calculate trend
        metric.trend = self._calculate_trend(name)
        metric.significance = self._calculate_significance(name)
    
    def _extract_features(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from metadata."""
        features = {}
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, bool):
                features[key] = 1.0 if value else 0.0
        
        # Add time-based features
        now = datetime.now()
        features["hour_of_day"] = now.hour
        features["day_of_week"] = now.weekday()
        
        return features
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate performance trend for metric."""
        if metric_name not in self.performance_history:
            return "stable"
        
        history = list(self.performance_history[metric_name])
        if len(history) < 10:
            return "stable"
        
        # Compare recent vs older values
        recent_values = [m.value for m in history[-5:]]
        older_values = [m.value for m in history[-10:-5]]
        
        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values)
        
        change_ratio = (recent_avg - older_avg) / abs(older_avg) if older_avg != 0 else 0
        
        if change_ratio > self.sensitivity:
            return "improving" if metric_name in ["cache_hit_rate", "success_rate"] else "degrading"
        elif change_ratio < -self.sensitivity:
            return "degrading" if metric_name in ["cache_hit_rate", "success_rate"] else "improving"
        else:
            return "stable"
    
    def _calculate_significance(self, metric_name: str) -> float:
        """Calculate statistical significance of recent changes."""
        if metric_name not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history[metric_name])
        if len(history) < 20:
            return 0.0
        
        # Simple t-test like significance calculation
        recent_values = [m.value for m in history[-10:]]
        older_values = [m.value for m in history[-20:-10]]
        
        recent_mean = np.mean(recent_values)
        older_mean = np.mean(older_values)
        
        recent_std = np.std(recent_values) if len(recent_values) > 1 else 0
        older_std = np.std(older_values) if len(older_values) > 1 else 0
        
        pooled_std = np.sqrt((recent_std**2 + older_std**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        t_stat = abs(recent_mean - older_mean) / (pooled_std * np.sqrt(2/10))
        
        # Convert to rough significance score (0-1)
        return min(1.0, t_stat / 3.0)
    
    def _adaptation_loop(self):
        """Main adaptation control loop."""
        while self.adaptation_active:
            try:
                self._analyze_performance()
                self._make_adaptations()
                self._validate_adaptations()
                time.sleep(self.adaptation_interval)
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {str(e)}")
                time.sleep(self.adaptation_interval * 2)  # Back off on error
    
    def _analyze_performance(self):
        """Analyze current performance and identify optimization opportunities."""
        current_stats = {}
        
        # Collect current performance metrics
        if self.optimization_manager.cache:
            cache_stats = self.optimization_manager.cache.get_stats()
            current_stats["cache_hit_rate"] = cache_stats["hit_rate"]
            current_stats["cache_memory_usage"] = cache_stats["memory_usage_mb"]
            
            self.record_performance_metric("cache_hit_rate", cache_stats["hit_rate"])
            self.record_performance_metric("cache_memory_usage", cache_stats["memory_usage_mb"])
        
        # Collect scaling metrics
        scaling_status = self.scaling_manager.get_scaling_status()
        if "metrics" in scaling_status:
            metrics = scaling_status["metrics"]
            current_stats["cpu_utilization"] = metrics.get("cpu_utilization", 0)
            current_stats["queue_depth"] = metrics.get("queue_depth", 0)
            
            self.record_performance_metric("cpu_utilization", metrics.get("cpu_utilization", 0))
            self.record_performance_metric("queue_depth", metrics.get("queue_depth", 0))
    
    def _make_adaptations(self):
        """Make adaptive optimizations based on performance analysis."""
        adaptations_made = []
        
        # Cache optimization adaptations
        if self._should_adapt_cache():
            adaptation = self._adapt_cache_settings()
            if adaptation:
                adaptations_made.append(adaptation)
        
        # Resource allocation adaptations
        if self._should_adapt_resources():
            adaptation = self._adapt_resource_allocation()
            if adaptation:
                adaptations_made.append(adaptation)
        
        # Predictive adaptations
        predictive_adaptations = self._make_predictive_adaptations()
        adaptations_made.extend(predictive_adaptations)
        
        # Record all adaptations
        for adaptation in adaptations_made:
            self.adaptation_history.append(adaptation)
            self.adaptation_counters[adaptation.strategy] += 1
            
            self.logger.info(f"Applied adaptation: {adaptation.strategy.value} - {adaptation.decision}")
    
    def _should_adapt_cache(self) -> bool:
        """Determine if cache settings should be adapted."""
        if "cache_hit_rate" not in self.performance_history:
            return False
        
        recent_metrics = list(self.performance_history["cache_hit_rate"])[-5:]
        if not recent_metrics:
            return False
        
        avg_hit_rate = np.mean([m.value for m in recent_metrics])
        return avg_hit_rate < self.cache_hit_rate_target
    
    def _adapt_cache_settings(self) -> Optional[AdaptationDecision]:
        """Adapt cache settings based on performance."""
        if not self.optimization_manager.cache:
            return None
        
        cache_stats = self.optimization_manager.cache.get_stats()
        current_hit_rate = cache_stats["hit_rate"]
        
        decision = None
        confidence = 0.7
        expected_improvement = 0.0
        
        if current_hit_rate < self.cache_hit_rate_target:
            if cache_stats["memory_usage_mb"] < cache_stats["max_memory_mb"] * 0.8:
                # Increase cache size
                new_size = min(
                    self.optimization_manager.cache.max_size * 1.5,
                    self.optimization_manager.cache.max_size + 200
                )
                self.optimization_manager.cache.max_size = int(new_size)
                
                decision = f"Increased cache size to {int(new_size)}"
                expected_improvement = 0.1  # 10% hit rate improvement expected
            else:
                # Adjust TTL to reduce memory pressure
                new_ttl = self.optimization_manager.cache.default_ttl * 0.8
                self.optimization_manager.cache.default_ttl = new_ttl
                
                decision = f"Reduced cache TTL to {new_ttl:.0f}s"
                expected_improvement = 0.05  # 5% improvement expected
        
        if decision:
            return AdaptationDecision(
                timestamp=time.time(),
                strategy=AdaptiveStrategy.CACHING_OPTIMIZATION,
                decision=decision,
                confidence=confidence,
                expected_improvement=expected_improvement,
                applied=True
            )
        
        return None
    
    def _should_adapt_resources(self) -> bool:
        """Determine if resource allocation should be adapted."""
        if "cpu_utilization" not in self.performance_history:
            return False
        
        recent_metrics = list(self.performance_history["cpu_utilization"])[-5:]
        if not recent_metrics:
            return False
        
        avg_utilization = np.mean([m.value for m in recent_metrics])
        return abs(avg_utilization - self.resource_utilization_target) > 0.2
    
    def _adapt_resource_allocation(self) -> Optional[AdaptationDecision]:
        """Adapt resource allocation based on utilization."""
        recent_cpu = list(self.performance_history["cpu_utilization"])[-5:]
        if not recent_cpu:
            return None
        
        avg_cpu = np.mean([m.value for m in recent_cpu])
        
        decision = None
        confidence = 0.8
        expected_improvement = 0.0
        
        if avg_cpu > self.resource_utilization_target + 0.2:
            # High utilization - need more resources
            scaling_status = self.scaling_manager.get_scaling_status()
            current_workers = scaling_status["workers"]["available_workers"]
            
            if current_workers < self.scaling_manager.auto_scaler.max_workers:
                decision = "Recommended scale-up due to high CPU utilization"
                expected_improvement = 0.3  # 30% improvement expected
        
        elif avg_cpu < self.resource_utilization_target - 0.2:
            # Low utilization - can reduce resources
            scaling_status = self.scaling_manager.get_scaling_status()
            current_workers = scaling_status["workers"]["available_workers"]
            
            if current_workers > self.scaling_manager.auto_scaler.min_workers:
                decision = "Recommended scale-down due to low CPU utilization"
                expected_improvement = 0.1  # 10% cost improvement expected
        
        if decision:
            return AdaptationDecision(
                timestamp=time.time(),
                strategy=AdaptiveStrategy.RESOURCE_ALLOCATION,
                decision=decision,
                confidence=confidence,
                expected_improvement=expected_improvement,
                applied=False  # Scaling manager will handle actual scaling
            )
        
        return None
    
    def _make_predictive_adaptations(self) -> List[AdaptationDecision]:
        """Make adaptations based on predictive models."""
        adaptations = []
        
        # Predict cache hit rate
        predicted_hit_rate = self.predictive_model.predict("cache_hit_rate", horizon=5)
        if predicted_hit_rate and predicted_hit_rate < self.cache_hit_rate_target:
            adaptation = AdaptationDecision(
                timestamp=time.time(),
                strategy=AdaptiveStrategy.LOAD_PREDICTION,
                decision=f"Predicted cache hit rate decline to {predicted_hit_rate:.2f}",
                confidence=0.6,
                expected_improvement=0.0,  # Preventive measure
                applied=False
            )
            adaptations.append(adaptation)
        
        # Predict resource needs
        predicted_cpu = self.predictive_model.predict("cpu_utilization", horizon=3)
        if predicted_cpu and predicted_cpu > 0.9:
            adaptation = AdaptationDecision(
                timestamp=time.time(),
                strategy=AdaptiveStrategy.LOAD_PREDICTION,
                decision=f"Predicted high CPU utilization: {predicted_cpu:.2f}",
                confidence=0.7,
                expected_improvement=0.0,  # Preventive measure
                applied=False
            )
            adaptations.append(adaptation)
        
        return adaptations
    
    def _validate_adaptations(self):
        """Validate previous adaptations and rollback if necessary."""
        current_time = time.time()
        
        # Check adaptations made in the last hour
        recent_adaptations = [
            a for a in self.adaptation_history
            if current_time - a.timestamp < 3600 and a.applied and a.actual_improvement is None
        ]
        
        for adaptation in recent_adaptations:
            # Measure actual improvement
            actual_improvement = self._measure_adaptation_impact(adaptation)
            adaptation.actual_improvement = actual_improvement
            
            # Check if rollback is needed
            if (actual_improvement < adaptation.expected_improvement * 0.5 or 
                actual_improvement < -0.1):  # Negative improvement
                
                self._rollback_adaptation(adaptation)
                adaptation.rollback_trigger = "Poor performance impact"
                
                self.logger.warning(f"Rolled back adaptation: {adaptation.decision}")
    
    def _measure_adaptation_impact(self, adaptation: AdaptationDecision) -> float:
        """Measure actual impact of an adaptation."""
        adaptation_time = adaptation.timestamp
        
        # Define measurement window around adaptation
        before_window = (adaptation_time - 1800, adaptation_time)  # 30 min before
        after_window = (adaptation_time + 300, adaptation_time + 1800)  # 5-30 min after
        
        # Measure relevant metric based on adaptation strategy
        if adaptation.strategy == AdaptiveStrategy.CACHING_OPTIMIZATION:
            return self._measure_cache_improvement(before_window, after_window)
        elif adaptation.strategy == AdaptiveStrategy.RESOURCE_ALLOCATION:
            return self._measure_resource_improvement(before_window, after_window)
        
        return 0.0
    
    def _measure_cache_improvement(self, before_window: Tuple[float, float], after_window: Tuple[float, float]) -> float:
        """Measure cache performance improvement."""
        if "cache_hit_rate" not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history["cache_hit_rate"])
        
        # Get metrics in each window
        before_metrics = [
            m.value for m in history
            if before_window[0] <= m.timestamp <= before_window[1]
        ]
        
        after_metrics = [
            m.value for m in history
            if after_window[0] <= m.timestamp <= after_window[1]
        ]
        
        if not before_metrics or not after_metrics:
            return 0.0
        
        before_avg = np.mean(before_metrics)
        after_avg = np.mean(after_metrics)
        
        return after_avg - before_avg
    
    def _measure_resource_improvement(self, before_window: Tuple[float, float], after_window: Tuple[float, float]) -> float:
        """Measure resource utilization improvement."""
        if "cpu_utilization" not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history["cpu_utilization"])
        
        before_metrics = [
            m.value for m in history
            if before_window[0] <= m.timestamp <= before_window[1]
        ]
        
        after_metrics = [
            m.value for m in history
            if after_window[0] <= m.timestamp <= after_window[1]
        ]
        
        if not before_metrics or not after_metrics:
            return 0.0
        
        before_avg = np.mean(before_metrics)
        after_avg = np.mean(after_metrics)
        
        # For CPU utilization, getting closer to target is improvement
        before_distance = abs(before_avg - self.resource_utilization_target)
        after_distance = abs(after_avg - self.resource_utilization_target)
        
        return before_distance - after_distance
    
    def _rollback_adaptation(self, adaptation: AdaptationDecision):
        """Rollback a failed adaptation."""
        if adaptation.strategy == AdaptiveStrategy.CACHING_OPTIMIZATION:
            if "Increased cache size" in adaptation.decision:
                # Revert cache size increase
                if self.optimization_manager.cache:
                    old_size = int(self.optimization_manager.cache.max_size / 1.5)
                    self.optimization_manager.cache.max_size = old_size
            elif "Reduced cache TTL" in adaptation.decision:
                # Revert TTL reduction
                if self.optimization_manager.cache:
                    old_ttl = self.optimization_manager.cache.default_ttl / 0.8
                    self.optimization_manager.cache.default_ttl = old_ttl
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Generate comprehensive adaptation report."""
        current_time = time.time()
        
        # Recent adaptations (last 24 hours)
        recent_adaptations = [
            a for a in self.adaptation_history
            if current_time - a.timestamp < 86400
        ]
        
        # Calculate success rate
        successful_adaptations = [
            a for a in recent_adaptations
            if a.actual_improvement is not None and a.actual_improvement > 0
        ]
        
        success_rate = len(successful_adaptations) / len(recent_adaptations) if recent_adaptations else 0
        
        # Strategy effectiveness
        strategy_stats = {}
        for strategy in AdaptiveStrategy:
            strategy_adaptations = [a for a in recent_adaptations if a.strategy == strategy]
            if strategy_adaptations:
                successful = [a for a in strategy_adaptations if a.actual_improvement and a.actual_improvement > 0]
                strategy_stats[strategy.value] = {
                    "total": len(strategy_adaptations),
                    "successful": len(successful),
                    "success_rate": len(successful) / len(strategy_adaptations),
                    "avg_improvement": np.mean([a.actual_improvement for a in successful if a.actual_improvement])
                }
        
        return {
            "period_summary": {
                "total_adaptations": len(recent_adaptations),
                "successful_adaptations": len(successful_adaptations),
                "success_rate": success_rate,
                "rollbacks": len([a for a in recent_adaptations if a.rollback_trigger])
            },
            "strategy_effectiveness": strategy_stats,
            "current_targets": {
                "cache_hit_rate": self.cache_hit_rate_target,
                "response_time": self.response_time_target,
                "resource_utilization": self.resource_utilization_target,
                "error_rate": self.error_rate_threshold
            },
            "predictive_insights": self._get_predictive_insights(),
            "recent_adaptations": [
                {
                    "timestamp": a.timestamp,
                    "strategy": a.strategy.value,
                    "decision": a.decision,
                    "confidence": a.confidence,
                    "expected_improvement": a.expected_improvement,
                    "actual_improvement": a.actual_improvement,
                    "rollback": a.rollback_trigger is not None
                }
                for a in recent_adaptations[-10:]  # Last 10 adaptations
            ]
        }
    
    def _get_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights for upcoming performance."""
        insights = {}
        
        # Cache predictions
        cache_prediction = self.predictive_model.predict("cache_hit_rate", horizon=10)
        if cache_prediction:
            insights["cache_hit_rate"] = {
                "predicted_value": cache_prediction,
                "trend": "declining" if cache_prediction < self.cache_hit_rate_target else "stable",
                "recommendation": "increase cache size" if cache_prediction < 0.7 else "maintain current settings"
            }
        
        # Resource predictions
        cpu_prediction = self.predictive_model.predict("cpu_utilization", horizon=10)
        if cpu_prediction:
            insights["cpu_utilization"] = {
                "predicted_value": cpu_prediction,
                "trend": "increasing" if cpu_prediction > self.resource_utilization_target else "stable",
                "recommendation": "scale up" if cpu_prediction > 0.8 else "maintain current resources"
            }
        
        return insights
    
    def shutdown(self):
        """Shutdown adaptive optimization system."""
        self.adaptation_active = False
        if self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=5.0)
        
        self.logger.info("Adaptive optimizer shutdown completed")


# Global adaptive optimizer instance (initialized later with dependencies)
adaptive_optimizer: Optional[AdaptiveOptimizer] = None


def initialize_adaptive_optimizer(optimization_manager: OptimizationManager, scaling_manager: ScalingManager):
    """Initialize global adaptive optimizer with dependencies."""
    global adaptive_optimizer
    adaptive_optimizer = AdaptiveOptimizer(optimization_manager, scaling_manager)
    return adaptive_optimizer