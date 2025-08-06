"""CrewAI framework adapter for reflexion capabilities."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.agent import ReflexionAgent
from ..core.types import ReflectionType, ReflexionResult
from ..memory.episodic import EpisodicMemory


class ReflexiveCrewMember:
    """CrewAI agent enhanced with reflexion capabilities."""
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str = "",
        llm_config: Optional[Dict] = None,
        tools: Optional[List] = None,
        reflection_after_tools: bool = True,
        learn_from_crew_feedback: bool = True,
        reflection_strategy: str = "balanced",
        share_learnings: bool = False,
        max_reflections: int = 3,
        **kwargs
    ):
        """Initialize reflexive CrewAI member.
        
        Args:
            role: Agent role in the crew
            goal: Agent's primary goal
            backstory: Agent's background story
            llm_config: LLM configuration
            tools: Available tools for the agent
            reflection_after_tools: Whether to reflect after tool usage
            learn_from_crew_feedback: Whether to learn from other crew members
            reflection_strategy: Strategy for reflection ("aggressive", "balanced", "minimal")
            share_learnings: Whether to share learnings with the crew
            max_reflections: Maximum number of reflections per task
            **kwargs: Additional configuration
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.reflection_after_tools = reflection_after_tools
        self.learn_from_crew_feedback = learn_from_crew_feedback
        self.reflection_strategy = reflection_strategy
        self.share_learnings = share_learnings
        
        # Configure reflexion based on strategy
        strategy_config = self._get_strategy_config(reflection_strategy)
        
        # Initialize reflexion agent
        self.reflexion_agent = ReflexionAgent(
            llm=llm_config.get("model", "gpt-4") if llm_config else "gpt-4",
            max_iterations=min(max_reflections, strategy_config["max_iterations"]),
            reflection_type=ReflectionType.STRUCTURED,
            success_threshold=strategy_config["threshold"],
            **kwargs
        )
        
        # Initialize memory for learning
        self.memory = EpisodicMemory(storage_path=f"./crew_memory_{role.lower().replace(' ', '_')}.json")
        
        # Track crew interactions and learnings
        self.crew_learnings: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
        
    def _get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """Get configuration for reflection strategy."""
        strategies = {
            "aggressive": {"max_iterations": 5, "threshold": 0.9},
            "balanced": {"max_iterations": 3, "threshold": 0.7},
            "minimal": {"max_iterations": 2, "threshold": 0.5}
        }
        return strategies.get(strategy, strategies["balanced"])
    
    def execute_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a task with reflexion enhancement."""
        task_start = datetime.now()
        
        # Build enhanced task prompt with role context
        enhanced_task = self._build_role_aware_prompt(task_description, context)
        
        # Apply reflexion to task execution
        result = self.reflexion_agent.run(
            task=enhanced_task,
            success_criteria=f"meets {self.role} standards,achieves goal,actionable"
        )
        
        # Store in memory
        task_record = {
            "task": task_description,
            "result": result.output,
            "success": result.success,
            "reflections": len(result.reflections),
            "execution_time": (datetime.now() - task_start).total_seconds(),
            "timestamp": task_start.isoformat()
        }
        
        self.task_history.append(task_record)
        self.memory.store_episode(task_description, result, {"role": self.role})
        
        # Generate crew-shareable insights if enabled
        if self.share_learnings and result.reflections:
            learning = self._extract_shareable_learning(result)
            self.crew_learnings.append(learning)
        
        return {
            "output": result.output,
            "success": result.success,
            "metadata": {
                "role": self.role,
                "reflections": len(result.reflections),
                "learning_extracted": self.share_learnings and result.reflections
            }
        }
    
    def _build_role_aware_prompt(self, task: str, context: Optional[Dict] = None) -> str:
        """Build task prompt with role awareness."""
        prompt = f"""
Role: {self.role}
Goal: {self.goal}
Context: {self.backstory}

Task: {task}
"""
        
        if context:
            prompt += f"\nAdditional Context: {context}"
        
        # Add recent learnings if available
        recent_patterns = self.memory.get_success_patterns()
        if recent_patterns["patterns"]:
            top_patterns = recent_patterns["patterns"][:3]
            prompt += f"\n\nRecent Successful Approaches:\n"
            for pattern, count in top_patterns:
                prompt += f"- {pattern} (used {count} times successfully)\n"
        
        return prompt
    
    def _extract_shareable_learning(self, result: ReflexionResult) -> Dict[str, Any]:
        """Extract learning that can be shared with crew."""
        if not result.reflections:
            return {}
        
        latest_reflection = result.reflections[-1]
        
        return {
            "role": self.role,
            "task_type": self._classify_task_type(result.task),
            "key_insights": latest_reflection.improvements[:2],  # Top 2 insights
            "common_issues": latest_reflection.issues[:2],  # Top 2 issues
            "success": result.success,
            "confidence": latest_reflection.confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def _classify_task_type(self, task: str) -> str:
        """Classify task type for learning categorization."""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['research', 'analyze', 'investigate', 'find']):
            return 'research'
        elif any(word in task_lower for word in ['write', 'create', 'draft', 'compose']):
            return 'creation'
        elif any(word in task_lower for word in ['review', 'evaluate', 'assess', 'critique']):
            return 'evaluation'
        elif any(word in task_lower for word in ['plan', 'strategy', 'design', 'architect']):
            return 'planning'
        else:
            return 'general'
    
    def receive_crew_feedback(self, feedback: Dict[str, Any]):
        """Receive and learn from crew feedback."""
        if not self.learn_from_crew_feedback:
            return
        
        # Store feedback as a learning episode
        feedback_learning = {
            "type": "crew_feedback",
            "source": feedback.get("source", "unknown"),
            "content": feedback.get("content", ""),
            "rating": feedback.get("rating", 0.5),
            "actionable_items": feedback.get("improvements", []),
            "timestamp": datetime.now().isoformat()
        }
        
        self.crew_learnings.append(feedback_learning)
    
    def share_learnings(self) -> List[Dict[str, Any]]:
        """Share learnings with other crew members."""
        if not self.share_learnings:
            return []
        
        # Filter and return shareable learnings
        shareable = []
        for learning in self.crew_learnings:
            if learning.get("confidence", 0) > 0.6:  # Only share confident learnings
                shareable.append({
                    "from_role": self.role,
                    "task_type": learning.get("task_type", "general"),
                    "insights": learning.get("key_insights", []),
                    "best_practices": learning.get("common_issues", []),
                    "confidence": learning.get("confidence", 0.5)
                })
        
        return shareable
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the crew member."""
        if not self.task_history:
            return {"status": "no_tasks_executed"}
        
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for task in self.task_history if task["success"])
        total_reflections = sum(task["reflections"] for task in self.task_history)
        avg_execution_time = sum(task["execution_time"] for task in self.task_history) / total_tasks
        
        return {
            "role": self.role,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
            "total_reflections": total_reflections,
            "avg_reflections_per_task": total_reflections / total_tasks,
            "avg_execution_time": avg_execution_time,
            "learnings_shared": len(self.crew_learnings) if self.share_learnings else 0
        }


class ReflexiveCrew:
    """Enhanced crew with collective learning capabilities."""
    
    def __init__(self, agents: List[ReflexiveCrewMember], enable_cross_learning: bool = True):
        """Initialize reflexive crew.
        
        Args:
            agents: List of reflexive crew members
            enable_cross_learning: Whether agents should learn from each other
        """
        self.agents = agents
        self.enable_cross_learning = enable_cross_learning
        self.crew_memory = EpisodicMemory(storage_path="./crew_collective_memory.json")
        self.task_results: List[Dict[str, Any]] = []
    
    def kickoff(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks with collective learning."""
        results = []
        
        for task in tasks:
            task_description = task.get("description", "")
            assigned_agent = task.get("agent", self.agents[0])
            
            # Execute task
            result = assigned_agent.execute_task(task_description, task.get("context"))
            results.append({
                "task": task_description,
                "agent": assigned_agent.role,
                "result": result
            })
            
            # Facilitate cross-learning if enabled
            if self.enable_cross_learning:
                self._facilitate_cross_learning(assigned_agent, result)
        
        # Store crew-level results
        crew_result = {
            "total_tasks": len(tasks),
            "results": results,
            "crew_performance": self._calculate_crew_performance(results),
            "timestamp": datetime.now().isoformat()
        }
        
        self.task_results.append(crew_result)
        return crew_result
    
    def _facilitate_cross_learning(self, executing_agent: ReflexiveCrewMember, result: Dict[str, Any]):
        """Facilitate learning between crew members."""
        learnings = executing_agent.share_learnings()
        
        # Share learnings with other agents
        for agent in self.agents:
            if agent != executing_agent and learnings:
                for learning in learnings:
                    agent.receive_crew_feedback({
                        "source": executing_agent.role,
                        "content": f"Learning from {executing_agent.role}",
                        "improvements": learning.get("insights", []),
                        "rating": learning.get("confidence", 0.5)
                    })
    
    def _calculate_crew_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall crew performance metrics."""
        total_tasks = len(results)
        if total_tasks == 0:
            return {"status": "no_tasks"}
        
        successful_tasks = sum(1 for r in results if r["result"]["success"])
        
        # Agent-level performance
        agent_performance = {}
        for agent in self.agents:
            agent_results = [r for r in results if r["agent"] == agent.role]
            if agent_results:
                agent_success_rate = sum(1 for r in agent_results if r["result"]["success"]) / len(agent_results)
                agent_performance[agent.role] = {
                    "tasks": len(agent_results),
                    "success_rate": agent_success_rate
                }
        
        return {
            "overall_success_rate": successful_tasks / total_tasks,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "agent_performance": agent_performance,
            "cross_learning_enabled": self.enable_cross_learning
        }
    
    def get_crew_insights(self) -> Dict[str, Any]:
        """Get insights about crew performance and learning."""
        if not self.task_results:
            return {"status": "no_data"}
        
        # Aggregate crew learnings
        all_learnings = []
        for agent in self.agents:
            all_learnings.extend(agent.share_learnings())
        
        # Performance trends
        recent_performance = self.task_results[-5:] if len(self.task_results) >= 5 else self.task_results
        avg_success_rate = sum(r["crew_performance"]["overall_success_rate"] for r in recent_performance) / len(recent_performance)
        
        return {
            "crew_size": len(self.agents),
            "total_sessions": len(self.task_results),
            "recent_success_rate": avg_success_rate,
            "collective_learnings": len(all_learnings),
            "agent_summaries": [agent.get_performance_summary() for agent in self.agents],
            "cross_learning_active": self.enable_cross_learning
        }