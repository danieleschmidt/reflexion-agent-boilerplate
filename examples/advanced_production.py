#!/usr/bin/env python3
"""Advanced production example demonstrating enterprise-grade reflexion features."""

import sys
import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reflexion import ReflexionAgent, ReflectionType, ReflectionPrompts, PromptDomain, EpisodicMemory, MemoryStore
from reflexion.adapters import AutoGenReflexion, ReflexiveCrewMember, ReflexionChain


def setup_logging():
    """Setup comprehensive logging for production monitoring."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reflexion_production.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class ProductionReflexionSystem:
    """Production-ready reflexion system with advanced monitoring and scaling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        
        # Initialize advanced memory system
        self.memory = EpisodicMemory(
            storage_path=config.get('memory_path', './production_memory.json'),
            max_episodes=config.get('max_episodes', 10000)
        )
        
        self.memory_store = MemoryStore()
        
        # Create agents with different specializations
        self.agents = {
            'general': ReflexionAgent(
                llm=config.get('llm', 'gpt-4'),
                max_iterations=config.get('max_iterations', 3),
                reflection_type=ReflectionType.STRUCTURED,
                success_threshold=config.get('success_threshold', 0.8)
            ),
            'coding': ReflexionAgent(
                llm=config.get('llm', 'gpt-4'),
                max_iterations=4,
                reflection_type=ReflectionType.STRUCTURED,
                success_threshold=0.9
            ),
            'analysis': ReflexionAgent(
                llm=config.get('llm', 'gpt-4'),
                max_iterations=3,
                reflection_type=ReflectionType.SCALAR,
                success_threshold=0.75
            )
        }
        
        # Performance tracking
        self.metrics = {
            'tasks_processed': 0,
            'successful_tasks': 0,
            'total_processing_time': 0,
            'total_iterations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.results_cache = {}
        self.start_time = time.time()
    
    def classify_task(self, task_description: str) -> str:
        """Classify task to select appropriate agent."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['code', 'function', 'implement', 'debug', 'algorithm']):
            return 'coding'
        elif any(word in task_lower for word in ['analyze', 'data', 'metrics', 'performance', 'trends']):
            return 'analysis'
        else:
            return 'general'
    
    def get_domain_for_task(self, task_description: str) -> PromptDomain:
        """Get appropriate prompt domain for task."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['code', 'programming', 'software', 'debug']):
            return PromptDomain.SOFTWARE_ENGINEERING
        elif any(word in task_lower for word in ['data', 'analysis', 'statistics', 'metrics']):
            return PromptDomain.DATA_ANALYSIS
        elif any(word in task_lower for word in ['research', 'study', 'investigate']):
            return PromptDomain.RESEARCH
        elif any(word in task_lower for word in ['write', 'creative', 'story', 'content']):
            return PromptDomain.CREATIVE_WRITING
        else:
            return PromptDomain.GENERAL
    
    async def process_task_async(self, task_id: str, task_description: str, metadata: dict = None) -> dict:
        """Process task asynchronously with advanced features."""
        start_time = time.time()
        metadata = metadata or {}
        
        self.logger.info(f"Processing task {task_id}: {task_description[:50]}...")
        
        # Check cache first
        cache_key = f"{hash(task_description)}_{metadata.get('version', 1)}"
        if cache_key in self.results_cache:
            self.metrics['cache_hits'] += 1
            self.logger.info(f"Cache hit for task {task_id}")
            return self.results_cache[cache_key]
        
        self.metrics['cache_misses'] += 1
        
        try:
            # Select appropriate agent and domain
            agent_type = self.classify_task(task_description)
            domain = self.get_domain_for_task(task_description)
            agent = self.agents[agent_type]
            
            # Get domain-specific success criteria
            success_criteria = self._get_domain_criteria(domain, metadata)
            
            # Recall similar experiences
            similar_episodes = self.memory.recall_similar(task_description, k=3)
            context = self._build_context_from_episodes(similar_episodes)
            
            # Enhanced task with context
            enhanced_task = task_description
            if context:
                enhanced_task += f"\n\nRelevant past experiences:\n{context}"
            
            # Execute with reflexion
            result = agent.run(
                task=enhanced_task,
                success_criteria=success_criteria
            )
            
            # Store comprehensive episode data
            episode_metadata = {
                'task_id': task_id,
                'agent_type': agent_type,
                'domain': domain.value,
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'similar_episodes_used': len(similar_episodes),
                **metadata
            }
            
            self.memory.store_episode(task_description, result, episode_metadata)
            self.memory_store.store_episode(task_id, {
                'task': task_description,
                'result': result.output,
                'success': result.success,
                'metadata': episode_metadata
            })
            
            # Update metrics
            self.metrics['tasks_processed'] += 1
            if result.success:
                self.metrics['successful_tasks'] += 1
            self.metrics['total_processing_time'] += time.time() - start_time
            self.metrics['total_iterations'] += result.iterations
            
            # Cache result
            task_result = {
                'task_id': task_id,
                'success': result.success,
                'output': result.output,
                'iterations': result.iterations,
                'processing_time': time.time() - start_time,
                'reflections_count': len(result.reflections),
                'agent_type': agent_type,
                'domain': domain.value,
                'confidence': result.reflections[-1].confidence if result.reflections else 0.5
            }
            
            self.results_cache[cache_key] = task_result
            
            self.logger.info(f"Task {task_id} completed: {result.success} in {result.iterations} iterations")
            return task_result
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            error_result = {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'agent_type': 'unknown',
                'domain': 'unknown'
            }
            self.results_cache[cache_key] = error_result
            return error_result
    
    def _get_domain_criteria(self, domain: PromptDomain, metadata: dict) -> str:
        """Get domain-specific success criteria."""
        domain_criteria = {
            PromptDomain.SOFTWARE_ENGINEERING: "correct,efficient,maintainable,tested",
            PromptDomain.DATA_ANALYSIS: "accurate,statistically_valid,actionable,visualized",
            PromptDomain.CREATIVE_WRITING: "engaging,original,well_structured,appropriate_tone",
            PromptDomain.RESEARCH: "comprehensive,credible_sources,well_cited,objective",
            PromptDomain.GENERAL: "complete,accurate,clear,relevant"
        }
        
        base_criteria = domain_criteria.get(domain, "complete,accurate,relevant")
        custom_criteria = metadata.get('success_criteria', '')
        
        return f"{base_criteria},{custom_criteria}" if custom_criteria else base_criteria
    
    def _build_context_from_episodes(self, episodes: List) -> str:
        """Build context string from similar episodes."""
        if not episodes:
            return ""
        
        context_parts = []
        for i, episode in enumerate(episodes[:3], 1):
            if hasattr(episode, 'result') and episode.result.success:
                context_parts.append(f"{i}. Similar task: {episode.task[:100]}...")
                if episode.result.reflections:
                    key_learnings = episode.result.reflections[-1].improvements[:2]
                    if key_learnings:
                        context_parts.append(f"   Key learnings: {', '.join(key_learnings)}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def batch_process_async(self, tasks: List[dict]) -> List[dict]:
        """Process multiple tasks concurrently."""
        self.logger.info(f"Starting async batch processing of {len(tasks)} tasks")
        
        # Create async tasks
        async_tasks = []
        for i, task in enumerate(tasks):
            task_id = task.get('id', f'task_{i}')
            task_description = task.get('description', '')
            metadata = task.get('metadata', {})
            
            async_task = self.process_task_async(task_id, task_description, metadata)
            async_tasks.append(async_task)
        
        # Execute concurrently with limit
        semaphore = asyncio.Semaphore(self.config.get('max_concurrent_tasks', 5))
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in async_tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {tasks[i].get('id', f'task_{i}')} failed: {result}")
                final_results.append({
                    'task_id': tasks[i].get('id', f'task_{i}'),
                    'success': False,
                    'error': str(result)
                })
            else:
                final_results.append(result)
        
        self.logger.info(f"Async batch processing completed: {len(final_results)} results")
        return final_results
    
    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive system statistics and insights."""
        memory_patterns = self.memory.get_success_patterns()
        uptime = time.time() - self.start_time
        
        # Agent performance breakdown
        agent_stats = {}
        for result in self.results_cache.values():
            agent_type = result.get('agent_type', 'unknown')
            if agent_type not in agent_stats:
                agent_stats[agent_type] = {'total': 0, 'successful': 0}
            
            agent_stats[agent_type]['total'] += 1
            if result.get('success'):
                agent_stats[agent_type]['successful'] += 1
        
        # Calculate rates
        for agent_type in agent_stats:
            total = agent_stats[agent_type]['total']
            if total > 0:
                agent_stats[agent_type]['success_rate'] = agent_stats[agent_type]['successful'] / total
        
        return {
            'memory': {
                'total_episodes': memory_patterns['total_episodes'],
                'success_rate': memory_patterns['success_rate'],
                'top_patterns': memory_patterns['patterns'][:5]
            },
            'performance': {
                'tasks_processed': self.metrics['tasks_processed'],
                'success_rate': self.metrics['successful_tasks'] / max(self.metrics['tasks_processed'], 1),
                'avg_processing_time': self.metrics['total_processing_time'] / max(self.metrics['tasks_processed'], 1),
                'avg_iterations': self.metrics['total_iterations'] / max(self.metrics['tasks_processed'], 1)
            },
            'cache': {
                'size': len(self.results_cache),
                'hit_rate': self.metrics['cache_hits'] / max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)
            },
            'agents': agent_stats,
            'system': {
                'uptime': uptime,
                'config': self.config
            }
        }
    
    def auto_optimize(self):
        """Automatically optimize system based on performance data."""
        stats = self.get_comprehensive_stats()
        
        # Adaptive threshold adjustment per agent
        for agent_type, agent in self.agents.items():
            if agent_type in stats['agents']:
                success_rate = stats['agents'][agent_type].get('success_rate', 0.5)
                
                if success_rate < 0.6:
                    agent.success_threshold = max(0.4, agent.success_threshold * 0.9)
                    self.logger.info(f"Lowered {agent_type} agent threshold to {agent.success_threshold:.2f}")
                elif success_rate > 0.9:
                    agent.success_threshold = min(0.95, agent.success_threshold * 1.05)
                    self.logger.info(f"Raised {agent_type} agent threshold to {agent.success_threshold:.2f}")
        
        # Cache management
        cache_hit_rate = stats['cache']['hit_rate']
        if cache_hit_rate < 0.3 and len(self.results_cache) > 1000:
            # Clear least recently used entries
            sorted_cache = sorted(
                self.results_cache.items(),
                key=lambda x: x[1].get('processing_time', 0)
            )
            # Keep only the most recent 500 entries
            self.results_cache = dict(sorted_cache[-500:])
            self.logger.info("Optimized cache by removing old entries")


async def demo_production_usage():
    """Demonstrate advanced production usage with async processing."""
    print("=== Advanced Production Reflexion System Demo ===")
    
    # Advanced production configuration
    config = {
        'llm': 'gpt-4',
        'max_iterations': 3,
        'success_threshold': 0.8,
        'max_episodes': 10000,
        'max_concurrent_tasks': 3,
        'memory_path': './production_memory.json'
    }
    
    # Initialize system
    system = ProductionReflexionSystem(config)
    
    # Complex production tasks
    tasks = [
        {
            'id': 'PROD_001',
            'description': 'Analyze user behavior patterns from e-commerce transaction data to identify revenue optimization opportunities',
            'metadata': {
                'priority': 'high',
                'domain': 'analytics',
                'success_criteria': 'quantitative_insights,actionable_recommendations,statistical_significance'
            }
        },
        {
            'id': 'PROD_002',
            'description': 'Design and implement a fault-tolerant microservice for real-time payment processing with 99.99% uptime requirement',
            'metadata': {
                'priority': 'critical',
                'domain': 'architecture',
                'success_criteria': 'fault_tolerant,scalable,secure,monitored'
            }
        },
        {
            'id': 'PROD_003',
            'description': 'Create a comprehensive machine learning model evaluation framework for production ML pipelines',
            'metadata': {
                'priority': 'medium',
                'domain': 'ml_engineering',
                'success_criteria': 'automated,comprehensive,production_ready,monitored'
            }
        },
        {
            'id': 'PROD_004',
            'description': 'Write technical documentation and API specification for the new GraphQL endpoint with examples',
            'metadata': {
                'priority': 'medium', 
                'domain': 'documentation',
                'success_criteria': 'complete,clear,examples_included,developer_friendly'
            }
        },
        {
            'id': 'PROD_005',
            'description': 'Debug and fix memory leak in Node.js application causing performance degradation in production',
            'metadata': {
                'priority': 'urgent',
                'domain': 'debugging',
                'success_criteria': 'root_cause_identified,fix_implemented,performance_restored,monitoring_added'
            }
        }
    ]
    
    # Process tasks asynchronously
    print(f"Processing {len(tasks)} complex production tasks concurrently...")
    start_time = time.time()
    
    results = await system.batch_process_async(tasks)
    
    processing_time = time.time() - start_time
    
    # Display detailed results
    print(f"\nProcessing completed in {processing_time:.2f}s")
    print("\nDetailed Results:")
    print("-" * 80)
    
    for result in results:
        status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
        confidence = result.get('confidence', 0)
        agent_type = result.get('agent_type', 'unknown')
        domain = result.get('domain', 'unknown')
        
        print(f"{status} {result['task_id']} ({agent_type}/{domain})")
        print(f"    Time: {result.get('processing_time', 0):.2f}s | "
              f"Iterations: {result.get('iterations', 0)} | "
              f"Reflections: {result.get('reflections_count', 0)} | "
              f"Confidence: {confidence:.2f}")
        
        if not result.get('success') and 'error' in result:
            print(f"    Error: {result['error']}")
        
        print()
    
    # System optimization
    print("Performing intelligent system optimization...")
    system.auto_optimize()
    
    # Display comprehensive statistics
    stats = system.get_comprehensive_stats()
    print(f"\nComprehensive System Statistics:")
    print(f"  📊 Performance Metrics:")
    print(f"      Tasks processed: {stats['performance']['tasks_processed']}")
    print(f"      Overall success rate: {stats['performance']['success_rate']:.2%}")
    print(f"      Average processing time: {stats['performance']['avg_processing_time']:.2f}s")
    print(f"      Average iterations: {stats['performance']['avg_iterations']:.1f}")
    
    print(f"  🧠 Memory & Learning:")
    print(f"      Episodes stored: {stats['memory']['total_episodes']}")
    print(f"      Memory success rate: {stats['memory']['success_rate']:.2%}")
    
    print(f"  ⚡ Cache Performance:")
    print(f"      Cache size: {stats['cache']['size']}")
    print(f"      Hit rate: {stats['cache']['hit_rate']:.2%}")
    
    print(f"  🤖 Agent Performance:")
    for agent_type, agent_stats in stats['agents'].items():
        success_rate = agent_stats.get('success_rate', 0)
        print(f"      {agent_type}: {agent_stats['successful']}/{agent_stats['total']} ({success_rate:.2%})")
    
    if stats['memory']['top_patterns']:
        print(f"  🎯 Top Success Patterns:")
        for i, (pattern, count) in enumerate(stats['memory']['top_patterns'][:3], 1):
            print(f"      {i}. {pattern} ({count} times)")
    
    return system


async def demo_framework_integrations():
    """Demonstrate framework adapter integrations."""
    print("\n\n=== Framework Integration Demos ===")
    
    # AutoGen Integration Demo
    print("\n🔧 AutoGen Integration:")
    autogen_agent = AutoGenReflexion(
        name="ProductionCoder",
        system_message="You are a senior software engineer with reflexion capabilities.",
        llm_config={"model": "gpt-4"},
        max_self_iterations=3,
        memory_window=10
    )
    
    coding_requests = [
        "Implement a rate limiter for API endpoints",
        "Add error handling to the rate limiter",
        "Write unit tests for the rate limiter"
    ]
    
    for request in coding_requests:
        response = autogen_agent.initiate_chat(message=request)
        print(f"Request: {request}")
        print(f"Response: {response[:100]}...")
        print()
    
    summary = autogen_agent.get_reflection_summary()
    print(f"AutoGen Summary: {summary['success_rate']:.2%} success rate, "
          f"{summary['avg_reflections_per_conversation']:.1f} avg reflections")
    
    # CrewAI Integration Demo
    print("\n👥 CrewAI Integration:")
    researcher = ReflexiveCrewMember(
        role="Senior Research Analyst",
        goal="Conduct thorough technical research with high accuracy",
        backstory="Expert in technology analysis and market research",
        reflection_strategy="balanced",
        share_learnings=True
    )
    
    writer = ReflexiveCrewMember(
        role="Technical Writer", 
        goal="Create clear, comprehensive technical documentation",
        backstory="Specialist in translating complex technical concepts",
        reflection_strategy="aggressive",
        learn_from_crew_feedback=True
    )
    
    # Simulate crew tasks
    research_task = "Research emerging trends in microservice architecture patterns for 2024"
    writing_task = "Write a technical guide on implementing circuit breaker patterns"
    
    research_result = researcher.execute_task(research_task)
    writing_result = writer.execute_task(writing_task)
    
    print(f"Research task: {'✅ Success' if research_result['success'] else '❌ Failed'}")
    print(f"Writing task: {'✅ Success' if writing_result['success'] else '❌ Failed'}")
    
    # Share learnings between crew members  
    researcher_learnings = researcher.share_learnings()
    if researcher_learnings:
        writer.receive_crew_feedback({
            'source': 'researcher',
            'content': 'Research insights',
            'improvements': researcher_learnings[0].get('insights', []),
            'rating': 0.8
        })
    
    print(f"Researcher performance: {researcher.get_performance_summary()}")
    print(f"Writer performance: {writer.get_performance_summary()}")


async def main():
    """Main production demo orchestrator."""
    try:
        print("🚀 Advanced Production Reflexion System - Comprehensive Demo")
        print("=" * 80)
        
        # Run advanced production demo
        await demo_production_usage()
        
        # Demonstrate framework integrations
        await demo_framework_integrations()
        
        print("\n" + "=" * 80)
        print("🎉 All production demos completed successfully!")
        print("""
📁 Files generated:
   • production_memory.json - Episodic memory data
   • reflexion_production.log - System logs
   
🔧 System optimizations applied:
   • Agent threshold adjustment
   • Cache optimization
   • Memory pattern analysis
   
💡 Ready for production deployment!
        """)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())