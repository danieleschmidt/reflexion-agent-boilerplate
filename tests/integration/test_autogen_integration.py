import pytest
from unittest.mock import Mock, patch

pytest.importorskip("autogen")

from reflexion.adapters.autogen import AutoGenReflexion


@pytest.mark.integration
class TestAutoGenIntegration:
    """Integration tests for AutoGen adapter."""

    def test_autogen_reflexion_initialization(self):
        """Test AutoGen reflexion adapter initialization."""
        with patch("autogen.AssistantAgent") as mock_agent:
            mock_agent_instance = Mock()
            mock_agent.return_value = mock_agent_instance
            
            reflexive_agent = AutoGenReflexion(
                name="test_agent",
                llm_config={"model": "gpt-4"},
                max_self_iterations=3
            )
            
            assert reflexive_agent.base_agent == mock_agent_instance
            assert reflexive_agent.max_self_iterations == 3

    def test_reflexion_enabled_chat(self):
        """Test chat with reflexion enabled."""
        with patch("autogen.AssistantAgent") as mock_agent, \
             patch("autogen.UserProxyAgent") as mock_user:
            
            mock_agent_instance = Mock()
            mock_user_instance = Mock()
            mock_agent.return_value = mock_agent_instance
            mock_user.return_value = mock_user_instance
            
            reflexive_agent = AutoGenReflexion(
                name="test_agent",
                llm_config={"model": "gpt-4"}
            )
            
            # Mock chat scenario
            mock_agent_instance.initiate_chat = Mock()
            
            reflexive_agent.initiate_chat(
                recipient=mock_user_instance,
                message="Test message"
            )
            
            mock_agent_instance.initiate_chat.assert_called_once()

    def test_memory_integration(self):
        """Test memory integration with AutoGen."""
        with patch("autogen.AssistantAgent") as mock_agent:
            mock_memory = Mock()
            
            reflexive_agent = AutoGenReflexion(
                name="test_agent",
                llm_config={"model": "gpt-4"},
                memory=mock_memory
            )
            
            assert reflexive_agent.memory == mock_memory

    @pytest.mark.slow
    def test_end_to_end_conversation(self):
        """Test end-to-end conversation with reflexion."""
        # This would be a more comprehensive test with actual AutoGen agents
        # For now, we'll mock the interaction
        pass