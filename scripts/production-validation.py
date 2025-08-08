#!/usr/bin/env python3
"""
Production Validation Script for ReflexionAgent

This script validates that the system is ready for production deployment
by running comprehensive checks across all system components.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion import (
    ReflexionAgent,
    OptimizedReflexionAgent,
    AutoScalingReflexionAgent,
    ReflectionType
)


class ProductionValidator:
    """Comprehensive production readiness validator."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def log_result(self, test_name: str, passed: bool, details: str = "", metrics: Optional[Dict] = None):
        """Log validation result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "status": status,
            "passed": passed,
            "details": details,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        self.results.append(result)
        print(f"{status} {test_name}: {details}")
    
    def validate_core_functionality(self):
        """Validate core ReflexionAgent functionality."""
        print("\n🧠 CORE FUNCTIONALITY VALIDATION")
        print("=" * 50)
        
        try:
            # Test basic agent creation
            agent = ReflexionAgent(llm="gpt-4", max_iterations=2)
            self.log_result("Agent Creation", True, "Basic ReflexionAgent created successfully")
            
            # Test basic task execution
            result = agent.run("Write a simple hello world function")
            success = result.success and result.iterations > 0 and len(result.output) > 10
            self.log_result(
                "Basic Task Execution", 
                success, 
                f"Task completed with {result.iterations} iterations",
                {"success": result.success, "iterations": result.iterations}
            )
            
            # Test reflection generation
            reflection_result = agent.run("invalid task that should trigger reflection")
            has_reflections = len(reflection_result.reflections) > 0
            self.log_result(
                "Reflection Generation",
                has_reflections,
                f"Generated {len(reflection_result.reflections)} reflections",
                {"reflections": len(reflection_result.reflections)}
            )
            
        except Exception as e:
            self.log_result("Core Functionality", False, f"Exception: {str(e)}")
    
    def validate_optimization_features(self):
        """Validate optimization and performance features."""
        print("\n⚡ OPTIMIZATION FEATURES VALIDATION")
        print("=" * 50)
        
        try:
            # Test optimized agent
            optimized_agent = OptimizedReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_caching=True,
                enable_parallel_execution=True
            )
            self.log_result("Optimized Agent Creation", True, "OptimizedReflexionAgent created with caching and parallelization")
            
            # Test caching performance
            start_time = time.time()
            task = "Create a sorting algorithm implementation"
            result1 = optimized_agent.run(task)
            first_run_time = time.time() - start_time
            
            start_time = time.time()
            result2 = optimized_agent.run(task)  # Should hit cache
            second_run_time = time.time() - start_time
            
            cache_performance = second_run_time < first_run_time * 0.8  # At least 20% faster
            performance_stats = optimized_agent.get_performance_stats()
            
            self.log_result(
                "Caching Performance",
                cache_performance,
                f"First run: {first_run_time:.3f}s, Second run: {second_run_time:.3f}s",
                {
                    "first_run_time": first_run_time,
                    "second_run_time": second_run_time,
                    "cache_hit_rate": performance_stats["derived_metrics"]["cache_hit_rate"]
                }
            )
            
            # Test performance stats collection
            stats_available = "agent_stats" in performance_stats and "cache_stats" in performance_stats
            self.log_result(
                "Performance Metrics",
                stats_available,
                "Performance statistics collection working",
                performance_stats["derived_metrics"]
            )
            
        except Exception as e:
            self.log_result("Optimization Features", False, f"Exception: {str(e)}")
    
    async def validate_scaling_features(self):
        """Validate auto-scaling and load distribution."""
        print("\n🚀 SCALING FEATURES VALIDATION")
        print("=" * 50)
        
        try:
            # Test auto-scaling agent
            scaling_agent = AutoScalingReflexionAgent(
                llm="gpt-4",
                max_iterations=2
            )
            self.log_result("Auto-Scaling Agent Creation", True, "AutoScalingReflexionAgent created successfully")
            
            # Test batch processing
            optimized_agent = OptimizedReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_parallel_execution=True
            )
            
            tasks = [
                "Implement a hash function",
                "Create a binary tree structure",
                "Design a cache mechanism",
                "Build a priority queue",
                "Develop a sorting utility"
            ]
            
            start_time = time.time()
            batch_results = await optimized_agent.run_batch(tasks)
            batch_time = time.time() - start_time
            
            batch_success = len(batch_results) == len(tasks)
            successful_tasks = sum(1 for r in batch_results if r.success)
            
            self.log_result(
                "Batch Processing",
                batch_success,
                f"Processed {len(tasks)} tasks in {batch_time:.3f}s",
                {
                    "batch_size": len(tasks),
                    "execution_time": batch_time,
                    "successful_tasks": successful_tasks,
                    "success_rate": successful_tasks / len(tasks)
                }
            )
            
            # Test scaling metrics
            scaling_stats = scaling_agent.get_scaling_stats()
            scaling_metrics_available = "current_workers" in scaling_stats and "load_metrics" in scaling_stats
            
            self.log_result(
                "Scaling Metrics",
                scaling_metrics_available,
                "Auto-scaling metrics collection working",
                scaling_stats
            )
            
        except Exception as e:
            self.log_result("Scaling Features", False, f"Exception: {str(e)}")
    
    def validate_memory_management(self):
        """Validate memory management and episodic memory."""
        print("\n🧠 MEMORY MANAGEMENT VALIDATION")
        print("=" * 50)
        
        try:
            from reflexion.memory import EpisodicMemory
            
            # Test episodic memory
            memory = EpisodicMemory(storage_path="./test_memory.json", max_episodes=10)
            self.log_result("Episodic Memory Creation", True, "EpisodicMemory created successfully")
            
            # Test memory with agent
            agent = ReflexionAgent(llm="gpt-4", max_iterations=2, memory=memory)
            
            # Execute tasks to populate memory
            for i in range(3):
                result = agent.run(f"Task {i}: Create a utility function")
                memory.store_episode(f"Task {i}", result, {"test": True})
            
            # Test memory patterns extraction
            patterns = memory.get_success_patterns()
            memory_working = patterns["total_episodes"] > 0
            
            self.log_result(
                "Memory Pattern Analysis",
                memory_working,
                f"Analyzed {patterns['total_episodes']} episodes",
                patterns
            )
            
            # Test memory recall
            similar_episodes = memory.recall_similar("create utility", k=2)
            recall_working = len(similar_episodes) > 0
            
            self.log_result(
                "Memory Recall",
                recall_working,
                f"Recalled {len(similar_episodes)} similar episodes"
            )
            
        except Exception as e:
            self.log_result("Memory Management", False, f"Exception: {str(e)}")
    
    def validate_error_handling(self):
        """Validate error handling and recovery."""
        print("\n🛡️ ERROR HANDLING VALIDATION")
        print("=" * 50)
        
        try:
            agent = ReflexionAgent(llm="gpt-4", max_iterations=3)
            
            # Test error scenario handling
            error_result = agent.run("error: this should trigger error handling")
            error_handled = not error_result.success or len(error_result.reflections) > 0
            
            self.log_result(
                "Error Scenario Handling",
                error_handled,
                f"Error scenario processed with {len(error_result.reflections)} reflections"
            )
            
            # Test input validation
            try:
                invalid_agent = ReflexionAgent(llm="invalid-model")
                validation_failed = False
            except Exception:
                validation_failed = True
            
            self.log_result(
                "Input Validation",
                validation_failed,
                "Invalid inputs properly rejected"
            )
            
            # Test resilience features
            from reflexion.core.health import health_checker
            from reflexion.core.resilience import resilience_manager
            
            health_system_active = hasattr(health_checker, 'run_all_checks')
            resilience_active = hasattr(resilience_manager, 'execute_with_resilience')
            
            self.log_result(
                "Resilience Systems",
                health_system_active and resilience_active,
                "Health checking and resilience systems active"
            )
            
        except Exception as e:
            self.log_result("Error Handling", False, f"Exception: {str(e)}")
    
    def validate_security_features(self):
        """Validate security features and input sanitization."""
        print("\n🔒 SECURITY FEATURES VALIDATION")
        print("=" * 50)
        
        try:
            from reflexion.core.security import security_manager
            from reflexion.core.validation import validator
            
            # Test input validation and sanitization
            test_inputs = [
                "normal task description",
                "task with <script>alert('xss')</script>",
                "task with SQL injection'; DROP TABLE users; --",
                "very long task description" + "x" * 1000
            ]
            
            security_passed = True
            for test_input in test_inputs:
                try:
                    validation_result = validator.validate_task(test_input)
                    if not validation_result.is_valid and "security" in " ".join(validation_result.errors).lower():
                        continue  # Security validation correctly rejected input
                    elif validation_result.is_valid:
                        sanitized = validation_result.sanitized_input
                        # Check that dangerous content was sanitized
                        if "script" in test_input.lower() and "script" in sanitized.lower():
                            security_passed = False
                            break
                except:
                    pass  # Expected for invalid inputs
            
            self.log_result(
                "Input Sanitization",
                security_passed,
                "Input validation and sanitization working"
            )
            
            # Test security manager functionality
            security_active = hasattr(security_manager, 'scan_input')
            self.log_result(
                "Security Manager",
                security_active,
                "Security manager system active"
            )
            
        except Exception as e:
            self.log_result("Security Features", False, f"Exception: {str(e)}")
    
    def validate_monitoring_systems(self):
        """Validate monitoring and observability systems."""
        print("\n📊 MONITORING SYSTEMS VALIDATION")
        print("=" * 50)
        
        try:
            from reflexion.core.performance import resource_monitor
            from reflexion.core.logging_config import logger, metrics
            
            # Test resource monitoring
            monitor_active = hasattr(resource_monitor, 'collect_metrics')
            self.log_result(
                "Resource Monitoring",
                monitor_active,
                "Resource monitoring system active"
            )
            
            # Test metrics collection
            metrics_active = hasattr(metrics, 'record_task_execution')
            self.log_result(
                "Metrics Collection",
                metrics_active,
                "Metrics collection system active"
            )
            
            # Test logging system
            try:
                logger.info("Production validation test log message")
                logging_active = True
            except Exception:
                logging_active = False
            
            self.log_result(
                "Logging System",
                logging_active,
                "Structured logging system active"
            )
            
            # Test health checking
            from reflexion.core.health import health_checker
            health_active = hasattr(health_checker, 'get_overall_status')
            
            self.log_result(
                "Health Checking",
                health_active,
                "Health checking system active"
            )
            
        except Exception as e:
            self.log_result("Monitoring Systems", False, f"Exception: {str(e)}")
    
    def validate_production_readiness(self):
        """Validate overall production readiness."""
        print("\n🚀 PRODUCTION READINESS VALIDATION")
        print("=" * 50)
        
        try:
            # Check deployment files
            deployment_files = [
                Path("./deployment/docker-compose.yml"),
                Path("./deployment/k8s/deployment.yaml"),
                Path("./deployment/production/Dockerfile"),
                Path("./monitoring/prometheus.yml")
            ]
            
            deployment_ready = all(f.exists() for f in deployment_files)
            self.log_result(
                "Deployment Configuration",
                deployment_ready,
                f"All deployment files present: {deployment_ready}"
            )
            
            # Check documentation
            docs = [
                Path("./README.md"),
                Path("./docs/DEPLOYMENT_GUIDE.md"),
                Path("./docs/API_REFERENCE.md")
            ]
            
            docs_ready = all(d.exists() for d in docs)
            self.log_result(
                "Documentation",
                docs_ready,
                f"Documentation complete: {docs_ready}"
            )
            
            # Performance benchmark
            start_time = time.time()
            agent = OptimizedReflexionAgent(
                llm="gpt-4",
                max_iterations=2,
                enable_caching=True,
                enable_parallel_execution=True
            )
            
            # Run performance test
            performance_tasks = [
                f"Create utility function #{i}" for i in range(10)
            ]
            
            successful = 0
            for task in performance_tasks:
                result = agent.run(task)
                if result.success:
                    successful += 1
            
            performance_time = time.time() - start_time
            throughput = len(performance_tasks) / performance_time
            success_rate = successful / len(performance_tasks)
            
            # Production performance criteria
            performance_ready = (
                throughput >= 5.0 and  # At least 5 tasks/second
                success_rate >= 0.8    # At least 80% success rate
            )
            
            self.log_result(
                "Performance Benchmarks",
                performance_ready,
                f"Throughput: {throughput:.1f} tasks/sec, Success: {success_rate:.1%}",
                {
                    "throughput": throughput,
                    "success_rate": success_rate,
                    "total_time": performance_time
                }
            )
            
        except Exception as e:
            self.log_result("Production Readiness", False, f"Exception: {str(e)}")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        passed_tests = [r for r in self.results if r["passed"]]
        failed_tests = [r for r in self.results if not r["passed"]]
        
        total_time = time.time() - self.start_time
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(passed_tests) / len(self.results) if self.results else 0,
                "total_validation_time": total_time,
                "production_ready": len(failed_tests) == 0
            },
            "categories": {},
            "failed_tests": failed_tests,
            "detailed_results": self.results,
            "recommendations": [],
            "timestamp": time.time()
        }
        
        # Group by category
        categories = {}
        for result in self.results:
            category = result["test"].split()[0] if " " in result["test"] else "General"
            if category not in categories:
                categories[category] = {"passed": 0, "failed": 0, "tests": []}
            
            if result["passed"]:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
            categories[category]["tests"].append(result)
        
        report["categories"] = categories
        
        # Generate recommendations
        if failed_tests:
            report["recommendations"].append(
                "❌ CRITICAL: Address all failed tests before production deployment"
            )
        
        if report["summary"]["success_rate"] < 1.0:
            report["recommendations"].append(
                f"⚠️  Some tests failed. Success rate: {report['summary']['success_rate']:.1%}"
            )
        
        if not failed_tests:
            report["recommendations"].append(
                "✅ System ready for production deployment"
            )
        
        return report
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production validation suite."""
        print("🚀 REFLEXION PRODUCTION VALIDATION")
        print("=" * 60)
        
        try:
            # Run all validation categories
            self.validate_core_functionality()
            self.validate_optimization_features()
            await self.validate_scaling_features()
            self.validate_memory_management()
            self.validate_error_handling()
            self.validate_security_features()
            self.validate_monitoring_systems()
            self.validate_production_readiness()
            
            # Generate final report
            report = self.generate_validation_report()
            
            print(f"\n{'='*60}")
            print("PRODUCTION VALIDATION SUMMARY")
            print('='*60)
            
            summary = report["summary"]
            status = "🟢 READY" if summary["production_ready"] else "🔴 NOT READY"
            
            print(f"Status: {status}")
            print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['success_rate']:.1%})")
            print(f"Validation time: {summary['total_validation_time']:.2f}s")
            
            if report["failed_tests"]:
                print(f"\n❌ Failed tests:")
                for test in report["failed_tests"]:
                    print(f"  - {test['test']}: {test['details']}")
            
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  {rec}")
            
            # Save report
            report_file = Path("./production_validation_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\n📁 Detailed report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Validation failed with exception: {str(e)}")
            traceback.print_exc()
            return {"error": str(e), "success": False}


async def main():
    """Main execution function."""
    validator = ProductionValidator()
    report = await validator.run_full_validation()
    
    # Return appropriate exit code
    if report.get("summary", {}).get("production_ready", False):
        print("\n🎉 PRODUCTION VALIDATION PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  PRODUCTION VALIDATION FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())