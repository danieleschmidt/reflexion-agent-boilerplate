"""Security utilities for reflexion agents."""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .exceptions import SecurityError
from .logging_config import logger


class SecurityManager:
    """Centralized security management for reflexion systems."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_patterns: Set[str] = set()
        self.security_events: List[Dict[str, Any]] = []
        
        # Load default security patterns
        self._load_security_patterns()
    
    def _load_security_patterns(self):
        """Load default security patterns to block."""
        self.blocked_patterns.update([
            # Code execution patterns
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            
            # File system access
            r'open\s*\(',
            r'file\s*\(',
            r'\.read\s*\(',
            r'\.write\s*\(',
            
            # Network access
            r'urllib\.',
            r'requests\.',
            r'socket\.',
            r'http\.',
            
            # Injection patterns
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'data:text\/html',
            
            # Shell commands
            r'\|\s*sh',
            r'\|\s*bash',
            r'\|\s*zsh',
            r'rm\s+-rf',
            r'sudo\s+',
        ])
    
    def validate_input_security(self, input_text: str, context: str = "unknown") -> bool:
        """Validate input for security issues."""
        import re
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                self._log_security_event(
                    event_type="blocked_pattern",
                    details={
                        "pattern": pattern,
                        "context": context,
                        "input_sample": input_text[:100]
                    }
                )
                raise SecurityError(
                    f"Input contains blocked security pattern: {pattern}",
                    pattern,
                    input_text
                )
        
        return True
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Initialize or clean old requests
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests outside the window
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if req_time > window_start
        ]
        
        # Check current rate
        current_requests = len(self.rate_limits[identifier])
        
        if current_requests >= max_requests:
            self._log_security_event(
                event_type="rate_limit_exceeded",
                details={
                    "identifier": identifier,
                    "current_requests": current_requests,
                    "max_requests": max_requests,
                    "window_minutes": window_minutes
                }
            )
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a secure API key with metadata."""
        key = secrets.token_urlsafe(32)
        
        self.api_keys[key] = {
            "name": name,
            "permissions": permissions or ["read"],
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        logger.info(f"API key generated for: {name}")
        return key
    
    def create_secure_hash(self, data: str, salt: Optional[str] = None) -> tuple:
        """Create secure hash with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hash_obj.hex(), salt
    
    def verify_secure_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify secure hash."""
        computed_hash, _ = self.create_secure_hash(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary and statistics."""
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "active_api_keys": len([k for k, v in self.api_keys.items() if v["active"]]),
            "total_api_keys": len(self.api_keys),
            "blocked_patterns": len(self.blocked_patterns),
            "security_events_24h": len(recent_events),
            "rate_limited_identifiers": len(self.rate_limits),
            "recent_event_types": list(set(event["event_type"] for event in recent_events))
        }
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only recent events to prevent memory growth
        cutoff_time = datetime.now() - timedelta(days=7)
        self.security_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]
        
        logger.warning(f"Security event: {event_type} - {details}")


class HealthChecker:
    """Health check utilities for reflexion systems."""
    
    def __init__(self):
        self.checks = {}
        self.last_results = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("log_directory", self._check_log_directory)
    
    def register_check(self, name: str, check_function):
        """Register a health check function."""
        self.checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "pass" if check_result["healthy"] else "fail",
                    "message": check_result["message"],
                    "details": check_result.get("details", {})
                }
                
                if not check_result["healthy"]:
                    results["overall_status"] = "unhealthy"
                    
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "details": {}
                }
                results["overall_status"] = "unhealthy"
        
        self.last_results = results
        return results
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            threshold = 90  # 90% usage threshold
            usage_percent = memory.percent
            
            return {
                "healthy": usage_percent < threshold,
                "message": f"Memory usage: {usage_percent:.1f}%",
                "details": {
                    "usage_percent": usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            }
        except ImportError:
            return {
                "healthy": True,
                "message": "Memory check skipped (psutil not available)",
                "details": {}
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            usage_percent = (used / total) * 100
            threshold = 90  # 90% usage threshold
            
            return {
                "healthy": usage_percent < threshold,
                "message": f"Disk usage: {usage_percent:.1f}%",
                "details": {
                    "usage_percent": usage_percent,
                    "free_gb": free / (1024**3),
                    "total_gb": total / (1024**3)
                }
            }
        except Exception:
            return {
                "healthy": True,
                "message": "Disk check skipped (unable to determine usage)",
                "details": {}
            }
    
    def _check_log_directory(self) -> Dict[str, Any]:
        """Check log directory accessibility."""
        from pathlib import Path
        
        log_dir = Path("./logs")
        
        try:
            log_dir.mkdir(exist_ok=True)
            test_file = log_dir / "health_check.tmp"
            
            # Test write access
            test_file.write_text("health check")
            content = test_file.read_text()
            test_file.unlink()
            
            return {
                "healthy": content == "health check",
                "message": "Log directory accessible",
                "details": {
                    "path": str(log_dir.absolute()),
                    "writable": True
                }
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Log directory not accessible: {str(e)}",
                "details": {
                    "path": str(log_dir.absolute()),
                    "error": str(e)
                }
            }


# Global instances
security_manager = SecurityManager()
health_checker = HealthChecker()