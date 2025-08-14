#!/usr/bin/env python3
"""
Production health check script for Reflexion Agent.
"""

import sys
import json
import time
import logging
from typing import Dict, Any
from datetime import datetime

# Suppress unnecessary logging
logging.disable(logging.WARNING)

def check_import_health() -> Dict[str, Any]:
    """Check if core modules can be imported."""
    try:
        # Test critical imports
        from src.reflexion.core.quantum_reflexion_agent import QuantumReflexionAgent
        from src.reflexion.core.security_validator import security_validator
        from src.reflexion.core.intelligent_monitoring import intelligent_monitor
        from src.reflexion.core.autonomous_scaling_engine import autonomous_scaling_engine
        
        return {
            "status": "healthy",
            "message": "All core modules imported successfully",
            "details": {
                "quantum_agent": "available",
                "security_validator": "available",
                "monitoring": "available",
                "scaling_engine": "available"
            }
        }
    except ImportError as e:
        return {
            "status": "unhealthy",
            "message": f"Import error: {str(e)}",
            "details": {"error": str(e)}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Unexpected error during imports: {str(e)}",
            "details": {"error": str(e)}
        }

def check_system_resources() -> Dict[str, Any]:
    """Check system resource availability."""
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Determine health status
        issues = []
        if memory_percent > 90:
            issues.append(f"High memory usage: {memory_percent:.1f}%")
        
        if cpu_percent > 95:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if disk_percent > 90:
            issues.append(f"High disk usage: {disk_percent:.1f}%")
        
        status = "unhealthy" if issues else "healthy"
        message = "; ".join(issues) if issues else "System resources within normal ranges"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "memory_percent": round(memory_percent, 1),
                "cpu_percent": round(cpu_percent, 1),
                "disk_percent": round(disk_percent, 1),
                "memory_available_gb": round(memory.available / (1024**3), 2)
            }
        }
    
    except ImportError:
        return {
            "status": "degraded",
            "message": "psutil not available, cannot check system resources",
            "details": {"psutil": "unavailable"}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Error checking system resources: {str(e)}",
            "details": {"error": str(e)}
        }

def check_configuration() -> Dict[str, Any]:
    """Check configuration validity."""
    try:
        import os
        
        # Check critical environment variables
        required_vars = []
        optional_vars = [
            "ENVIRONMENT", "LOG_LEVEL", "WORKERS", 
            "ENABLE_QUANTUM_ENHANCEMENT", "SECURITY_STRICT_MODE"
        ]
        
        config_status = {}
        issues = []
        
        # Check required variables
        for var in required_vars:
            if var not in os.environ:
                issues.append(f"Missing required environment variable: {var}")
                config_status[var] = "missing"
            else:
                config_status[var] = "present"
        
        # Check optional variables
        for var in optional_vars:
            config_status[var] = "present" if var in os.environ else "default"
        
        # Check runtime config file
        runtime_config_path = "/app/runtime_config.json"
        if os.path.exists(runtime_config_path):
            try:
                with open(runtime_config_path, 'r') as f:
                    config_data = json.load(f)
                config_status["runtime_config"] = "valid"
                config_status["environment"] = config_data.get("environment", "unknown")
            except json.JSONDecodeError:
                issues.append("Invalid runtime configuration JSON")
                config_status["runtime_config"] = "invalid"
        else:
            config_status["runtime_config"] = "missing"
        
        status = "unhealthy" if issues else "healthy"
        message = "; ".join(issues) if issues else "Configuration is valid"
        
        return {
            "status": status,
            "message": message,
            "details": config_status
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Error checking configuration: {str(e)}",
            "details": {"error": str(e)}
        }

def check_application_health() -> Dict[str, Any]:
    """Check application-specific health."""
    try:
        from src.reflexion.core.intelligent_monitoring import intelligent_monitor
        
        # Get system health from monitoring
        health_report = intelligent_monitor.get_system_health()
        
        status = "healthy"
        if health_report["overall_health_score"] < 0.3:
            status = "unhealthy"
        elif health_report["overall_health_score"] < 0.6:
            status = "degraded"
        
        return {
            "status": status,
            "message": f"Application health score: {health_report['overall_health_score']:.2f}",
            "details": {
                "health_score": health_report["overall_health_score"],
                "health_status": health_report["health_status"],
                "metrics_tracked": health_report["metrics_tracked"],
                "active_alerts": health_report["active_alerts"]
            }
        }
    
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Could not check application health: {str(e)}",
            "details": {"error": str(e)}
        }

def run_health_checks() -> Dict[str, Any]:
    """Run comprehensive health checks."""
    start_time = time.time()
    
    # Run individual health checks
    checks = {
        "imports": check_import_health(),
        "system_resources": check_system_resources(),
        "configuration": check_configuration(),
        "application": check_application_health()
    }
    
    # Determine overall status
    statuses = [check["status"] for check in checks.values()]
    
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Count issues
    issues = []
    for check_name, check_result in checks.items():
        if check_result["status"] != "healthy":
            issues.append(f"{check_name}: {check_result['message']}")
    
    execution_time = time.time() - start_time
    
    return {
        "overall_status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "execution_time_ms": round(execution_time * 1000, 2),
        "summary": {
            "total_checks": len(checks),
            "healthy_checks": sum(1 for s in statuses if s == "healthy"),
            "degraded_checks": sum(1 for s in statuses if s == "degraded"),
            "unhealthy_checks": sum(1 for s in statuses if s == "unhealthy")
        },
        "issues": issues,
        "detailed_checks": checks
    }

def main():
    """Main health check function."""
    try:
        # Run health checks
        health_result = run_health_checks()
        
        # Output results as JSON for structured logging
        print(json.dumps(health_result, indent=2))
        
        # Return appropriate exit code
        if health_result["overall_status"] == "healthy":
            sys.exit(0)
        elif health_result["overall_status"] == "degraded":
            sys.exit(1)  # Warning status
        else:
            sys.exit(2)  # Critical status
    
    except Exception as e:
        # Fallback error handling
        error_result = {
            "overall_status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
            "message": "Health check script encountered an error"
        }
        
        print(json.dumps(error_result, indent=2))
        sys.exit(2)

if __name__ == "__main__":
    main()