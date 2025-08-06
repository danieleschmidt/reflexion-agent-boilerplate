"""LangChain framework adapter for reflexion capabilities."""

from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod

from ..core.agent import ReflexionAgent
from ..core.types import ReflectionType, ReflexionResult


class ReflexionChain:
    """LangChain chain wrapper with reflexion capabilities."""
    
    def __init__(
        self,
        chain: Any,
        reflection_parser: Optional[Callable] = None,
        improvement_prompt_template: Optional[str] = None,
        max_iterations: int = 3,
        success_threshold: float = 0.7,
        reflection_triggers: Optional[List[str]] = None,
        **reflexion_kwargs
    ):
        """Initialize reflexion-enhanced LangChain chain.
        
        Args:
            chain: Base LangChain chain to enhance
            reflection_parser: Function to parse chain output for reflection
            improvement_prompt_template: Template for generating improvement prompts
            max_iterations: Maximum reflection iterations
            success_threshold: Threshold for considering output successful
            reflection_triggers: Conditions that trigger reflection
            **reflexion_kwargs: Additional reflexion configuration
        """
        self.base_chain = chain
        self.reflection_parser = reflection_parser or self._default_reflection_parser
        self.improvement_prompt_template = improvement_prompt_template or self._default_improvement_template
        self.reflection_triggers = reflection_triggers or ["low_quality", "incomplete", "error"]
        
        # Initialize reflexion agent
        self.reflexion_agent = ReflexionAgent(
            llm="gpt-4",  # Default, should be configurable
            max_iterations=max_iterations,
            reflection_type=ReflectionType.STRUCTURED,
            success_threshold=success_threshold,
            **reflexion_kwargs
        )
        
        self.execution_history: List[Dict[str, Any]] = []
    
    def run(self, inputs: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Run chain with reflexion enhancement."""
        # Convert string input to dict if necessary
        if isinstance(inputs, str):
            chain_inputs = {"input": inputs}
            task_description = inputs
        else:
            chain_inputs = inputs
            task_description = str(inputs)
        
        # Execute with reflexion
        reflexion_result = self.reflexion_agent.run(
            task=f"Execute LangChain task: {task_description}",
            success_criteria="accurate,complete,well-formatted"
        )
        
        # Store execution history
        execution_record = {
            "inputs": chain_inputs,
            "task": task_description,
            "reflexion_result": {
                "output": reflexion_result.output,
                "success": reflexion_result.success,
                "iterations": reflexion_result.iterations,
                "reflections_count": len(reflexion_result.reflections)
            },
            "kwargs": kwargs
        }
        self.execution_history.append(execution_record)
        
        return {
            "output": reflexion_result.output,
            "success": reflexion_result.success,
            "metadata": {
                "chain_type": type(self.base_chain).__name__,
                "reflexion_iterations": reflexion_result.iterations,
                "improvement_applied": len(reflexion_result.reflections) > 0
            }
        }
    
    def _default_reflection_parser(self, output: str) -> Dict[str, Any]:
        """Default parser for chain output reflection."""
        # Simple heuristic-based evaluation
        quality_indicators = [
            len(output) > 50,  # Sufficient length
            not output.lower().startswith("error"),  # No obvious errors
            "." in output,  # Contains sentences
            len(output.split()) > 10  # Adequate word count
        ]
        
        score = sum(quality_indicators) / len(quality_indicators)
        
        issues = []
        if len(output) <= 50:
            issues.append("Output too brief")
        if output.lower().startswith("error"):
            issues.append("Error in output")
        if "." not in output:
            issues.append("Output lacks proper sentence structure")
        if len(output.split()) <= 10:
            issues.append("Insufficient detail provided")
        
        return {
            "score": score,
            "issues": issues,
            "success": score >= 0.7
        }
    
    @property
    def _default_improvement_template(self) -> str:
        """Default template for improvement prompts."""
        return """
The previous LangChain execution had the following issues:
{issues}

Please improve the response by addressing these specific problems:
{improvements}

Original task: {task}
Previous output: {output}

Provide an improved response:
"""
    
    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """LangChain-compatible invoke method."""
        return self.run(inputs, **kwargs)
    
    def batch(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple inputs with reflexion."""
        results = []
        for input_item in inputs:
            result = self.run(input_item, **kwargs)
            results.append(result)
        return results
    
    def stream(self, inputs: Dict[str, Any], **kwargs):
        """Streaming interface (simplified implementation)."""
        result = self.run(inputs, **kwargs)
        yield result
    
    def get_chain_performance(self) -> Dict[str, Any]:
        """Get performance metrics for the reflexion-enhanced chain."""
        if not self.execution_history:
            return {"status": "no_executions"}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for record in self.execution_history 
            if record["reflexion_result"]["success"]
        )
        
        total_iterations = sum(
            record["reflexion_result"]["iterations"] 
            for record in self.execution_history
        )
        
        improved_executions = sum(
            1 for record in self.execution_history
            if record["reflexion_result"]["reflections_count"] > 0
        )
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_iterations": total_iterations / total_executions,
            "improvement_rate": improved_executions / total_executions,
            "chain_type": type(self.base_chain).__name__ if hasattr(self, 'base_chain') else "unknown"
        }


class ReflexionLangChainAgent:
    """LangChain agent wrapper with comprehensive reflexion capabilities."""
    
    def __init__(
        self,
        agent_executor: Any,
        tools: Optional[List] = None,
        reflection_on_tool_use: bool = True,
        reflection_on_final_answer: bool = True,
        memory_integration: bool = True,
        **reflexion_kwargs
    ):
        """Initialize reflexion-enhanced LangChain agent.
        
        Args:
            agent_executor: LangChain AgentExecutor instance
            tools: List of available tools
            reflection_on_tool_use: Whether to reflect after tool usage
            reflection_on_final_answer: Whether to reflect on final answers
            memory_integration: Whether to integrate with episodic memory
            **reflexion_kwargs: Additional reflexion configuration
        """
        self.agent_executor = agent_executor
        self.tools = tools or []
        self.reflection_on_tool_use = reflection_on_tool_use
        self.reflection_on_final_answer = reflection_on_final_answer
        self.memory_integration = memory_integration
        
        # Initialize reflexion capabilities
        self.reflexion_agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=reflexion_kwargs.get("max_iterations", 3),
            reflection_type=ReflectionType.STRUCTURED,
            success_threshold=reflexion_kwargs.get("success_threshold", 0.7)
        )
        
        self.execution_log: List[Dict[str, Any]] = []
        
        # Initialize memory if enabled
        if memory_integration:
            from ..memory.episodic import EpisodicMemory
            self.memory = EpisodicMemory(storage_path="./langchain_agent_memory.json")
        else:
            self.memory = None
    
    def run(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Run agent with reflexion enhancement."""
        # Apply reflexion to the entire agent execution
        reflexion_result = self.reflexion_agent.run(
            task=f"Execute agent task: {input_text}",
            success_criteria="task_completed,accurate_information,proper_tool_usage"
        )
        
        # Simulate tool usage reflection if enabled
        if self.reflection_on_tool_use and self.tools:
            tool_reflection = self._reflect_on_tool_usage(input_text, reflexion_result.output)
            reflexion_result.metadata["tool_reflection"] = tool_reflection
        
        # Store in memory if enabled
        if self.memory:
            self.memory.store_episode(input_text, reflexion_result, {
                "agent_type": "langchain",
                "tools_available": len(self.tools)
            })
        
        # Log execution
        self.execution_log.append({
            "input": input_text,
            "output": reflexion_result.output,
            "success": reflexion_result.success,
            "iterations": reflexion_result.iterations,
            "reflections": len(reflexion_result.reflections),
            "kwargs": kwargs
        })
        
        return {
            "output": reflexion_result.output,
            "success": reflexion_result.success,
            "metadata": {
                "reflexion_applied": True,
                "iterations": reflexion_result.iterations,
                "tool_reflection": self.reflection_on_tool_use,
                "memory_updated": self.memory_integration
            }
        }
    
    def _reflect_on_tool_usage(self, task: str, output: str) -> Dict[str, Any]:
        """Reflect on tool usage effectiveness."""
        # Analyze if tools were used appropriately
        tool_analysis = {
            "tools_available": len(self.tools),
            "tools_mentioned": self._count_tool_mentions(output),
            "appropriate_usage": True,  # Simplified assessment
            "suggestions": []
        }
        
        # Add suggestions based on analysis
        if tool_analysis["tools_mentioned"] == 0 and len(self.tools) > 0:
            tool_analysis["suggestions"].append("Consider using available tools for better results")
            tool_analysis["appropriate_usage"] = False
        
        return tool_analysis
    
    def _count_tool_mentions(self, output: str) -> int:
        """Count mentions of available tools in the output."""
        mentions = 0
        output_lower = output.lower()
        
        for tool in self.tools:
            tool_name = getattr(tool, 'name', str(tool)).lower()
            if tool_name in output_lower:
                mentions += 1
        
        return mentions
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated memory."""
        if not self.memory:
            return {"status": "memory_disabled"}
        
        patterns = self.memory.get_success_patterns()
        return {
            "success_patterns": patterns,
            "total_episodes": len(self.memory.episodes),
            "memory_enabled": True
        }


class LangChainReflexionMiddleware:
    """Middleware for adding reflexion to any LangChain component."""
    
    @staticmethod
    def wrap_chain(chain: Any, **reflexion_config) -> ReflexionChain:
        """Wrap any LangChain chain with reflexion capabilities."""
        return ReflexionChain(chain, **reflexion_config)
    
    @staticmethod
    def wrap_agent(agent_executor: Any, **reflexion_config) -> ReflexionLangChainAgent:
        """Wrap any LangChain agent with reflexion capabilities."""
        return ReflexionLangChainAgent(agent_executor, **reflexion_config)
    
    @staticmethod
    def create_reflexive_retrieval_qa(
        retriever: Any,
        llm: Any,
        reflection_on_retrieval: bool = True,
        **reflexion_config
    ) -> Dict[str, Any]:
        """Create a reflexive retrieval QA chain."""
        # This would integrate with LangChain's RetrievalQA
        # Simplified implementation for demonstration
        
        return {
            "type": "reflexive_retrieval_qa",
            "retriever": retriever,
            "llm": llm,
            "reflection_enabled": reflection_on_retrieval,
            "config": reflexion_config,
            "status": "configured"
        }