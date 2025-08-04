"""Logging configuration for reflexion agents."""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ReflexionLogger:
    """Centralized logging configuration for reflexion systems."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True
    ):
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger("reflexion")
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


class ReflexionMetrics:
    """Metrics collection for reflexion operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("reflexion.metrics")
        self.metrics = {
            "tasks_executed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_iterations": 0,
            "total_reflections": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        }
        self.session_start = datetime.now()
    
    def record_task_execution(
        self,
        success: bool,
        iterations: int,
        reflections: int,
        execution_time: float,
        task_type: Optional[str] = None
    ):
        """Record metrics for a task execution."""
        self.metrics["tasks_executed"] += 1
        
        if success:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        
        self.metrics["total_iterations"] += iterations
        self.metrics["total_reflections"] += reflections
        self.metrics["total_execution_time"] += execution_time
        
        # Update derived metrics
        self.metrics["average_execution_time"] = (
            self.metrics["total_execution_time"] / self.metrics["tasks_executed"]
        )
        self.metrics["success_rate"] = (
            self.metrics["successful_tasks"] / self.metrics["tasks_executed"]
        )
        
        # Log the execution
        self.logger.info(
            f"Task executed - Success: {success}, Iterations: {iterations}, "
            f"Reflections: {reflections}, Time: {execution_time:.2f}s, "
            f"Type: {task_type or 'unknown'}"
        )
        
        # Log aggregate metrics periodically
        if self.metrics["tasks_executed"] % 10 == 0:
            self.log_aggregate_metrics()
    
    def log_aggregate_metrics(self):
        """Log aggregate metrics."""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        self.logger.info(
            f"Aggregate Metrics - Tasks: {self.metrics['tasks_executed']}, "
            f"Success Rate: {self.metrics['success_rate']:.2%}, "
            f"Avg Time: {self.metrics['average_execution_time']:.2f}s, "
            f"Session Duration: {session_duration:.0f}s"
        )
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], (int, float)):
                self.metrics[key] = 0 if isinstance(self.metrics[key], int) else 0.0
        
        self.session_start = datetime.now()
        self.logger.info("Metrics reset")


# Global logging and metrics instances
default_logger = ReflexionLogger(
    log_level="INFO",
    log_file="./logs/reflexion.log",
    enable_console=True
)

logger = default_logger.get_logger()
metrics = ReflexionMetrics(logger)


def get_logger(name: str = "reflexion") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> logging.Logger:
    """Configure logging with custom settings."""
    reflexion_logger = ReflexionLogger(
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console
    )
    return reflexion_logger.get_logger()