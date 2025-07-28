#!/usr/bin/env python3
"""
Health check script for Reflexion Agent service.
Can be used for Docker health checks, load balancer probes, and monitoring.
"""

import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional
import requests
import psutil


def check_service_health(base_url: str = "http://localhost:8000", timeout: int = 10) -> Dict[str, Any]:
    """Check the health of the Reflexion service."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    try:
        # Check main health endpoint
        response = requests.get(f"{base_url}/health", timeout=timeout)
        health_status["checks"]["api"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "status_code": response.status_code
        }
        
        # Check readiness endpoint
        response = requests.get(f"{base_url}/ready", timeout=timeout)
        health_status["checks"]["readiness"] = {
            "status": "ready" if response.status_code == 200 else "not_ready",
            "response_time": response.elapsed.total_seconds()
        }
        
        # Check metrics endpoint
        response = requests.get(f"{base_url}/metrics", timeout=timeout)
        health_status["checks"]["metrics"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds()
        }
        
    except requests.exceptions.RequestException as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_status


def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }


def check_database_connection(db_url: Optional[str] = None) -> Dict[str, Any]:
    """Check database connectivity."""
    if not db_url:
        return {"status": "skipped", "reason": "no_db_url"}
    
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        return {"status": "healthy", "connection": "successful"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_redis_connection(redis_url: Optional[str] = None) -> Dict[str, Any]:
    """Check Redis connectivity."""
    if not redis_url:
        return {"status": "skipped", "reason": "no_redis_url"}
    
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        return {"status": "healthy", "connection": "successful"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Reflexion Agent Health Check")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout")
    parser.add_argument("--db-url", help="Database URL")
    parser.add_argument("--redis-url", help="Redis URL")
    parser.add_argument("--detailed", action="store_true", help="Include detailed checks")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Perform health checks
    health_status = check_service_health(args.url, args.timeout)
    
    if args.detailed:
        health_status["system"] = check_system_resources()
        health_status["database"] = check_database_connection(args.db_url)
        health_status["redis"] = check_redis_connection(args.redis_url)
    
    # Output results
    if args.format == "json":
        import json
        print(json.dumps(health_status, indent=2))
    else:
        print(f"Service Status: {health_status['status'].upper()}")
        print(f"Timestamp: {time.ctime(health_status['timestamp'])}")
        
        for check_name, check_result in health_status.get("checks", {}).items():
            status = check_result.get("status", "unknown")
            print(f"  {check_name}: {status.upper()}")
            if "response_time" in check_result:
                print(f"    Response time: {check_result['response_time']:.3f}s")
            if "error" in check_result:
                print(f"    Error: {check_result['error']}")
        
        if args.detailed:
            if "system" in health_status:
                sys_info = health_status["system"]
                print(f"\nSystem Resources:")
                print(f"  CPU: {sys_info['cpu_percent']}%")
                print(f"  Memory: {sys_info['memory_percent']}%")
                print(f"  Disk: {sys_info['disk_percent']}%")
            
            for service in ["database", "redis"]:
                if service in health_status:
                    service_status = health_status[service]["status"]
                    print(f"\n{service.capitalize()}: {service_status.upper()}")
    
    # Exit with appropriate code
    overall_healthy = health_status["status"] == "healthy"
    if args.detailed:
        # Also check external services
        db_healthy = health_status.get("database", {}).get("status") != "unhealthy"
        redis_healthy = health_status.get("redis", {}).get("status") != "unhealthy"
        overall_healthy = overall_healthy and db_healthy and redis_healthy
    
    sys.exit(0 if overall_healthy else 1)


if __name__ == "__main__":
    main()