"""AutoGen framework adapter for reflexion capabilities."""

from typing import Any, Dict, List, Optional

from ..core.agent import ReflexionAgent
from ..core.types import ReflectionType


class AutoGenReflexion:
    """Wrapper to add reflexion capabilities to AutoGen agents."""
    
    def __init__(
        self,
        base_agent=None,
        name: str = "reflexive_agent",
        system_message: str = "You are a helpful AI assistant with self-reflection capabilities.",
        llm_config: Optional[Dict] = None,
        max_self_iterations: int = 3,
        reflection_triggers: Optional[List[str]] = None,
        memory_window: int = 10,
        **kwargs
    ):
        """Initialize AutoGen reflexion wrapper.
        
        Args:
            base_agent: Existing AutoGen agent to wrap (optional)
            name: Agent name
            system_message: System message for the agent
            llm_config: LLM configuration dictionary
            max_self_iterations: Maximum self-reflection iterations
            reflection_triggers: List of triggers for reflection
            memory_window: Number of recent interactions to remember
            **kwargs: Additional configuration
        """
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config or {"model": "gpt-4"}
        self.max_self_iterations = max_self_iterations
        self.reflection_triggers = reflection_triggers or ["task_failure", "user_feedback_negative"]
        self.memory_window = memory_window
        
        # Initialize internal reflexion agent
        self.reflexion_agent = ReflexionAgent(
            llm=self.llm_config.get("model", "gpt-4"),
            max_iterations=max_self_iterations,
            reflection_type=ReflectionType.BINARY,
            **kwargs
        )
        
        self.base_agent = base_agent
        self.conversation_history = []
    
    def initiate_chat(self, recipient=None, message: str = "", **kwargs):
        """Initiate chat with reflexion enhancement."""
        # Use reflexion for message processing
        result = self.reflexion_agent.run(
            task=f"Respond to: {message}",
            success_criteria="helpful,accurate,relevant"
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "message": message,
            "response": result.output,
            "success": result.success,
            "reflections": len(result.reflections)
        })
        
        # Maintain memory window
        if len(self.conversation_history) > self.memory_window:
            self.conversation_history = self.conversation_history[-self.memory_window:]
        
        return result.output
    
    def generate_reply(self, messages, sender=None, **kwargs):
        """Generate reply with reflexion enhancement."""
        if not messages:
            return "I don't have a message to respond to."
        
        latest_message = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        
        # Apply reflexion to generate better responses
        result = self.reflexion_agent.run(
            task=f"Generate appropriate reply to: {latest_message}",
            success_criteria="relevant,helpful,appropriate"
        )
        
        return result.output
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection activities."""
        total_conversations = len(self.conversation_history)
        successful_conversations = sum(1 for conv in self.conversation_history if conv["success"])
        total_reflections = sum(conv["reflections"] for conv in self.conversation_history)
        
        return {
            "total_conversations": total_conversations,
            "successful_conversations": successful_conversations,
            "success_rate": successful_conversations / max(total_conversations, 1),
            "total_reflections": total_reflections,
            "avg_reflections_per_conversation": total_reflections / max(total_conversations, 1)
        }


def create_reflexive_autogen_agent(
    name: str = "ReflexiveAssistant",
    system_message: str = "You are a helpful AI assistant that learns from mistakes.",
    llm_config: Optional[Dict] = None,
    **reflexion_kwargs
) -> AutoGenReflexion:
    """Factory function to create a reflexive AutoGen agent.
    
    Args:
        name: Agent name
        system_message: System message
        llm_config: LLM configuration
        **reflexion_kwargs: Additional reflexion configuration
        
    Returns:
        AutoGenReflexion instance
    """
    return AutoGenReflexion(
        name=name,
        system_message=system_message,
        llm_config=llm_config,
        **reflexion_kwargs
    )