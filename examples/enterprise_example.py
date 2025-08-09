#!/usr/bin/env python3
"""Enterprise features example for the Reflexion Agent."""

import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reflexion.enterprise.governance import GovernanceFramework, ComplianceStandard
from reflexion.enterprise.multi_tenant import MultiTenantManager, TenantTier
from reflexion.scaling.auto_scaler import AutoScaler, MetricsCollector
from reflexion.scaling.distributed import DistributedReflexionManager, TaskPriority
from reflexion import ReflexionAgent, ReflectionType


async def governance_example():
    """Demonstrate governance and compliance features."""
    print("=== Governance & Compliance Example ===")
    
    # Initialize governance framework
    governance = GovernanceFramework({
        "audit_enabled": True,
        "compliance_monitoring": True
    })
    
    print("Governance framework initialized")
    
    # Test compliance enforcement
    operation_context = {
        "processes_personal_data": True,
        "consent_obtained": True,
        "data_pseudonymized": False,  # This will trigger a violation
        "authenticated": True,
        "encrypted": True
    }
    
    enforcement_result = governance.enforce_governance(
        operation="reflexion_execution",
        context=operation_context
    )
    
    print(f"Operation allowed: {enforcement_result['allowed']}")
    print(f"Violations found: {len(enforcement_result['violations'])}")
    
    if enforcement_result["violations"]:
        print("Compliance violations:")
        for violation in enforcement_result["violations"]:
            print(f"  - {violation.standard.value}: {violation.status}")
    
    # Generate governance report
    report = governance.generate_governance_report()
    print(f"\nGovernance Report Summary:")
    print(f"  Compliance Scores: {list(report['compliance_scores'].keys())}")
    print(f"  Recent Violations: {len(report['recent_violations'])}")


async def multi_tenant_example():
    """Demonstrate multi-tenant capabilities."""
    print("\n\n=== Multi-Tenant Example ===")
    
    # Initialize multi-tenant manager
    tenant_manager = MultiTenantManager()
    
    # Create tenants
    tenant1 = tenant_manager.create_tenant(
        tenant_id="acme_corp",
        name="ACME Corporation",
        tier=TenantTier.ENTERPRISE,
        settings={"max_concurrent_tasks": 20}
    )
    
    tenant2 = tenant_manager.create_tenant(
        tenant_id="small_biz",
        name="Small Business",
        tier=TenantTier.BASIC,
        settings={"max_concurrent_tasks": 2}
    )
    
    print(f"Created tenant: {tenant1.name} ({tenant1.tier.value})")
    print(f"Created tenant: {tenant2.name} ({tenant2.tier.value})")
    
    # Create execution contexts
    context1 = tenant_manager.create_execution_context(
        tenant_id="acme_corp",
        user_id="user123",
        session_id="session456"
    )
    
    context2 = tenant_manager.create_execution_context(
        tenant_id="small_biz", 
        user_id="user789",
        session_id="session012"
    )
    
    print(f"Created execution contexts for both tenants")
    
    # Execute tasks with tenant isolation
    agent = ReflexionAgent(
        llm="gpt-4",
        max_iterations=2,
        reflection_type=ReflectionType.BINARY
    )
    
    task = "Create a data validation function"
    
    try:
        # Execute for enterprise tenant
        result1 = await tenant_manager.execute_with_tenant_context(
            context1, agent, task
        )
        print(f"Enterprise tenant task: Success={result1.success}")
        
        # Execute for basic tenant
        result2 = await tenant_manager.execute_with_tenant_context(
            context2, agent, task
        )
        print(f"Basic tenant task: Success={result2.success}")
        
        # Show tenant status
        status1 = tenant_manager.get_tenant_status("acme_corp")
        status2 = tenant_manager.get_tenant_status("small_biz")
        
        print(f"\nTenant Status:")
        print(f"  ACME Corp API calls: {status1['resource_usage']['resources'].get('api_calls', {}).get('usage', 0)}")
        print(f"  Small Biz API calls: {status2['resource_usage']['resources'].get('api_calls', {}).get('usage', 0)}")
        
    except Exception as e:
        print(f"Tenant execution error: {e}")


async def auto_scaling_example():
    """Demonstrate auto-scaling capabilities."""
    print("\n\n=== Auto-Scaling Example ===")
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create auto-scaler with default policy
    auto_scaler = AutoScaler(
        scaling_policies=[],
        metrics_collector=metrics_collector
    )
    
    # Add default policy
    default_policy = auto_scaler.create_default_policy(
        name="reflexion_scaling",
        min_instances=1,
        max_instances=5
    )
    auto_scaler.policies["reflexion_scaling"] = default_policy
    
    print(f"Created auto-scaler with policy: {default_policy.name}")
    print(f"  Min instances: {default_policy.min_instances}")
    print(f"  Max instances: {default_policy.max_instances}")
    print(f"  Thresholds: {len(default_policy.thresholds)}")
    
    # Register scaling callback
    scaling_events = []
    
    def scaling_callback(old_instances, new_instances):
        scaling_events.append({
            "timestamp": asyncio.get_event_loop().time(),
            "old_instances": old_instances,
            "new_instances": new_instances
        })
        print(f"Scaling event: {old_instances} -> {new_instances} instances")
    
    auto_scaler.register_scaling_callback(scaling_callback)
    
    # Collect current metrics
    metrics = await metrics_collector.collect_metrics()
    print(f"\nCurrent Metrics:")
    print(f"  CPU: {metrics.cpu_utilization:.1f}%")
    print(f"  Memory: {metrics.memory_utilization:.1f}%")
    print(f"  Queue Length: {metrics.queue_length}")
    
    # Test scaling decision
    decision = auto_scaler.decision_engine.should_scale(
        metrics, default_policy, auto_scaler.current_instances
    )
    
    print(f"\nScaling Decision:")
    print(f"  Should scale: {decision['should_scale']}")
    print(f"  Direction: {decision['direction'].value if decision.get('direction') else 'none'}")
    print(f"  Reason: {decision['reason']}")
    
    # Get scaling status
    status = auto_scaler.get_scaling_status()
    print(f"\nAuto-scaler Status:")
    print(f"  Running: {status['running']}")
    print(f"  Current instances: {status['current_instances']}")
    print(f"  Active policies: {status['active_policies']}")


async def distributed_processing_example():
    """Demonstrate distributed processing capabilities."""
    print("\n\n=== Distributed Processing Example ===")
    
    # Initialize distributed manager
    dist_manager = DistributedReflexionManager()
    
    # Add worker nodes
    node1 = dist_manager.add_worker_node(
        node_id="worker_1",
        host="localhost", 
        port=8001,
        capacity=5,
        capabilities={"reflexion", "analysis"}
    )
    
    node2 = dist_manager.add_worker_node(
        node_id="worker_2",
        host="localhost",
        port=8002, 
        capacity=3,
        capabilities={"reflexion", "coding"}
    )
    
    print(f"Added worker nodes:")
    print(f"  {node1.node_id}: {node1.capacity} capacity")
    print(f"  {node2.node_id}: {node2.capacity} capacity")
    
    # Start distributed system
    await dist_manager.start()
    print("Started distributed processing")
    
    # Submit tasks
    tasks = [
        "Analyze this dataset for patterns",
        "Write a sorting algorithm", 
        "Create a data validation function",
        "Design a caching strategy",
        "Implement error handling"
    ]
    
    task_ids = []
    for task in tasks:
        task_id = await dist_manager.submit_reflexion_task(
            task=task,
            agent_config={
                "llm": "gpt-4",
                "max_iterations": 2,
                "reflection_type": "binary"
            },
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    
    print(f"\nSubmitted {len(task_ids)} tasks")
    
    # Wait for some tasks to complete (with short timeout for demo)
    completed_tasks = 0
    for task_id in task_ids:
        try:
            result = await dist_manager.wait_for_task_completion(
                task_id, timeout=10
            )
            if result["status"] == "completed":
                completed_tasks += 1
                print(f"Task completed: {result['task_id'][:8]}...")
            else:
                print(f"Task failed: {result['task_id'][:8]}... ({result['status']})")
        except asyncio.TimeoutError:
            print(f"Task timeout: {task_id[:8]}...")
    
    print(f"Completed {completed_tasks}/{len(task_ids)} tasks")
    
    # Get system status
    status = dist_manager.get_system_status()
    print(f"\nDistributed System Status:")
    print(f"  Total nodes: {status['node_statistics']['total_nodes']}")
    print(f"  Healthy nodes: {status['node_statistics']['healthy_nodes']}")
    print(f"  System utilization: {status['system_capacity']['utilization']:.1f}%")
    print(f"  Pending tasks: {status['queue_statistics']['pending_tasks']}")
    print(f"  Completed tasks: {status['queue_statistics']['completed_tasks']}")
    
    # Stop distributed system
    await dist_manager.stop()
    print("Stopped distributed processing")


def synchronous_enterprise_example():
    """Synchronous enterprise features demo."""
    print("\n\n=== Synchronous Enterprise Demo ===")
    
    from reflexion.enterprise.governance import AuditTrail, ComplianceMonitor
    
    # Create audit trail
    audit_trail = AuditTrail("./demo_audit.json")
    
    # Record some events
    audit_trail.record_event(
        event_type="agent_execution",
        actor="user123",
        resource="reflexion_task",
        action="execute",
        outcome="success"
    )
    
    audit_trail.record_event(
        event_type="data_processing",
        actor="system",
        resource="user_data",
        action="analyze",
        outcome="completed"
    )
    
    print(f"Recorded {len(audit_trail.events)} audit events")
    
    # Create compliance monitor
    compliance_monitor = ComplianceMonitor(audit_trail)
    
    # Test compliance assessment
    assessment = compliance_monitor.assess_compliance(
        standard=ComplianceStandard.GDPR,
        context={
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_pseudonymized": True
        }
    )
    
    print(f"\nCompliance Assessment:")
    print(f"  Standard: {assessment.standard.value}")
    print(f"  Status: {assessment.status}")
    print(f"  Score: {assessment.score:.2f}")
    print(f"  Findings: {len(assessment.findings)}")


async def main():
    """Run all enterprise examples."""
    print("Reflexion Enterprise Features Demo")
    print("=" * 50)
    
    try:
        await governance_example()
        await multi_tenant_example()
        await auto_scaling_example()
        await distributed_processing_example()
        synchronous_enterprise_example()
        
        print("\n" + "=" * 50)
        print("Enterprise examples completed successfully!")
        
    except Exception as e:
        print(f"Enterprise example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())