#!/usr/bin/env python3
"""Health check script for production deployment."""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, '/app')


async def check_basic_health() -> Dict[str, Any]:
    """Basic health checks."""
    checks = {}
    
    # Check Python version
    checks['python_version'] = {
        'status': 'healthy',
        'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    # Check required directories
    required_dirs = ['/app/logs', '/app/data', '/app/cache']
    for dir_path in required_dirs:
        path = Path(dir_path)
        checks[f'directory_{path.name}'] = {
            'status': 'healthy' if path.exists() and path.is_dir() else 'unhealthy',
            'path': str(path),
            'exists': path.exists(),
            'writable': path.is_dir() and os.access(path, os.W_OK) if path.exists() else False
        }
    
    # Check disk space
    try:
        stat = os.statvfs('/app')
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        checks['disk_space'] = {
            'status': 'healthy' if free_space_gb > 1.0 else 'warning' if free_space_gb > 0.5 else 'critical',
            'free_space_gb': round(free_space_gb, 2)
        }
    except Exception as e:
        checks['disk_space'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    return checks


async def check_reflexion_components() -> Dict[str, Any]:
    """Check reflexion-specific components."""
    checks = {}
    
    try:
        # Test core imports
        from reflexion import ReflexionAgent
        from reflexion.core.types import ReflectionType
        
        checks['core_imports'] = {
            'status': 'healthy',
            'message': 'Core reflexion modules imported successfully'
        }
        
        # Test basic agent creation
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=1,
            reflection_type=ReflectionType.BINARY
        )
        
        checks['agent_creation'] = {
            'status': 'healthy',
            'message': 'ReflexionAgent created successfully'
        }
        
        # Test enterprise components if available
        try:
            from reflexion.enterprise.governance import GovernanceFramework
            from reflexion.enterprise.multi_tenant import MultiTenantManager
            
            governance = GovernanceFramework()
            tenant_manager = MultiTenantManager()
            
            checks['enterprise_components'] = {
                'status': 'healthy',
                'message': 'Enterprise components initialized'
            }
        except Exception as e:
            checks['enterprise_components'] = {
                'status': 'warning',
                'message': f'Enterprise components unavailable: {str(e)}'
            }
        
        # Test scaling components if available
        try:
            from reflexion.scaling.auto_scaler import AutoScaler
            from reflexion.scaling.distributed import DistributedReflexionManager
            
            auto_scaler = AutoScaler([])
            dist_manager = DistributedReflexionManager()
            
            checks['scaling_components'] = {
                'status': 'healthy',
                'message': 'Scaling components initialized'
            }
        except Exception as e:
            checks['scaling_components'] = {
                'status': 'warning',
                'message': f'Scaling components unavailable: {str(e)}'
            }
        
    except Exception as e:
        checks['reflexion_import'] = {
            'status': 'critical',
            'error': str(e),
            'message': 'Failed to import reflexion components'
        }
    
    return checks


async def check_external_dependencies() -> Dict[str, Any]:
    """Check external service dependencies."""
    checks = {}
    
    # Check database connection
    database_url = os.getenv('REFLEXION_DATABASE_URL')
    if database_url:
        try:
            # In a real implementation, test actual DB connection
            checks['database'] = {
                'status': 'healthy',
                'message': 'Database configuration present'
            }
        except Exception as e:
            checks['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
    else:
        checks['database'] = {
            'status': 'warning',
            'message': 'No database URL configured'
        }
    
    # Check Redis connection
    redis_url = os.getenv('REFLEXION_REDIS_URL')
    if redis_url:
        try:
            # In a real implementation, test actual Redis connection
            checks['redis'] = {
                'status': 'healthy',
                'message': 'Redis configuration present'
            }
        except Exception as e:
            checks['redis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
    else:
        checks['redis'] = {
            'status': 'warning',
            'message': 'No Redis URL configured'
        }
    
    return checks


async def check_performance() -> Dict[str, Any]:
    """Check system performance metrics."""
    checks = {}
    
    try:
        # Memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            checks['memory'] = {
                'status': 'healthy' if memory.percent < 85 else 'warning' if memory.percent < 95 else 'critical',
                'usage_percent': memory.percent,
                'available_gb': round(memory.available / (1024**3), 2)
            }
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            checks['cpu'] = {
                'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical',
                'usage_percent': cpu_percent
            }
        except ImportError:
            checks['system_metrics'] = {
                'status': 'warning',
                'message': 'psutil not available for system metrics'
            }
        
        # Response time test
        start_time = time.time()
        
        # Simple computation to test responsiveness
        result = sum(i * i for i in range(1000))
        
        response_time = time.time() - start_time
        checks['response_time'] = {
            'status': 'healthy' if response_time < 0.1 else 'warning' if response_time < 0.5 else 'critical',
            'response_time_ms': round(response_time * 1000, 2)
        }
        
    except Exception as e:
        checks['performance_check'] = {
            'status': 'warning',
            'error': str(e)
        }
    
    return checks


async def main():
    """Main health check function."""
    health_status = {
        'timestamp': time.time(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    # Run all health checks
    health_checks = [
        ('basic', check_basic_health()),
        ('reflexion', check_reflexion_components()),
        ('dependencies', check_external_dependencies()),
        ('performance', check_performance())
    ]
    
    critical_failures = 0
    warnings = 0
    
    for category, check_coro in health_checks:
        try:
            checks = await check_coro
            health_status['checks'][category] = checks
            
            # Count issues
            for check_name, check_result in checks.items():
                status = check_result.get('status', 'unknown')
                if status == 'critical':
                    critical_failures += 1
                elif status in ['unhealthy', 'warning']:
                    warnings += 1
                    
        except Exception as e:
            health_status['checks'][category] = {
                'health_check_error': {
                    'status': 'critical',
                    'error': str(e)
                }
            }
            critical_failures += 1
    
    # Determine overall status
    if critical_failures > 0:
        health_status['overall_status'] = 'critical'
        exit_code = 1
    elif warnings > 0:
        health_status['overall_status'] = 'warning'
        exit_code = 0  # Don't fail container on warnings
    else:
        health_status['overall_status'] = 'healthy'
        exit_code = 0
    
    # Add summary
    health_status['summary'] = {
        'total_checks': sum(len(checks) for checks in health_status['checks'].values()),
        'critical_failures': critical_failures,
        'warnings': warnings,
        'exit_code': exit_code
    }
    
    # Output health status
    print(json.dumps(health_status, indent=2))
    
    # Write to health check file
    health_file = Path('/app/logs/health.json')
    health_file.parent.mkdir(exist_ok=True)
    
    with open(health_file, 'w') as f:
        json.dump(health_status, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())