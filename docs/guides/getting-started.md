# Getting Started with Reflexion Agent Boilerplate

*Last Updated: 2025-08-02 | Compatible with: v0.1.0+*

This guide will help you set up and run your first reflexion-enabled agent in under 10 minutes.

## 🎯 What You'll Learn

- Install and configure the Reflexion Agent Boilerplate
- Create your first self-improving agent
- Understand the reflexion feedback loop
- Integrate with popular agent frameworks

## 📋 Prerequisites

- **Python 3.9+** with pip installed
- **Basic familiarity** with AI agents and LLMs
- **API keys** for your preferred LLM provider (OpenAI, Anthropic, etc.)
- **Optional**: Docker for containerized development

## 🚀 Quick Installation

### Option 1: From PyPI (Recommended)

```bash
# Install the base package
pip install reflexion-agent-boilerplate

# Or install with framework adapters
pip install reflexion-agent-boilerplate[autogen,crewai,langchain]
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/your-org/reflexion-agent-boilerplate
cd reflexion-agent-boilerplate

# Install in development mode
pip install -e ".[dev]"
```

### Option 3: Docker Development

```bash
# Clone and start development environment
git clone https://github.com/your-org/reflexion-agent-boilerplate
cd reflexion-agent-boilerplate
docker-compose up -d

# Access the development container
docker-compose exec reflexion bash
```

## ⚡ Your First Reflexion Agent

### 1. Basic Reflexion Agent

Create a file called `first_agent.py`:

```python
from reflexion import ReflexionAgent
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create a reflexion agent
agent = ReflexionAgent(
    llm="gpt-4",
    max_iterations=3,
    reflect_on_failure=True,
    verbose=True  # Show the reflexion process
)

# Define a challenging task
task = """
Write a Python function that finds the longest common subsequence 
between two strings. The function should be efficient and handle 
edge cases properly.
"""

# Execute with automatic self-improvement
result = agent.run(
    task=task,
    success_criteria="passes all test cases and handles edge cases"
)

# View the results
print(f"Final solution:\n{result.output}")
print(f"Required {result.iterations} iterations")
print(f"Self-reflections: {len(result.reflections)}")

# See what the agent learned
for i, reflection in enumerate(result.reflections):
    print(f"\nReflection {i+1}:")
    print(f"Issues found: {reflection.issues}")
    print(f"Improvements: {reflection.improvements}")
```

Run your first agent:

```bash
python first_agent.py
```

### 2. Understanding the Output

You'll see output similar to:

```
🤖 Initial attempt...
❌ Evaluation failed: Missing edge case handling

🧠 Self-reflection...
Issues identified:
- No handling for empty strings
- Inefficient recursive approach
- Missing input validation

💡 Improvement strategy:
- Add input validation
- Use dynamic programming
- Handle edge cases explicitly

🤖 Improved attempt (iteration 2)...
✅ Success! All criteria met.

Final solution:
def longest_common_subsequence(str1, str2):
    # Input validation
    if not str1 or not str2:
        return ""
    
    # Dynamic programming approach
    # ... (optimized implementation)
    
Required 2 iterations
Self-reflections: 2
```

## 🔧 Framework Integration Examples

### With AutoGen

```python
from autogen import AssistantAgent
from reflexion.adapters import AutoGenReflexion

# Create base AutoGen agent
base_agent = AssistantAgent(
    name="reflexive_coder",
    system_message="You are an expert programmer who learns from mistakes.",
    llm_config={"model": "gpt-4"}
)

# Add reflexion capabilities
agent = AutoGenReflexion(
    base_agent=base_agent,
    memory_type="episodic",
    max_self_iterations=3
)

# Use normally - reflexion happens automatically
agent.initiate_chat(
    message="Implement a thread-safe singleton pattern in Python"
)
```

### With CrewAI

```python
from crewai import Agent, Task, Crew
from reflexion.adapters import CrewAIReflexion

# Create reflexive agent
agent = Agent(
    role="Senior Python Developer",
    goal="Write high-quality, bug-free code",
    backstory="Expert programmer with focus on code quality",
    llm="gpt-4"
)

# Add reflexion wrapper
reflexive_agent = CrewAIReflexion(
    agent=agent,
    reflection_strategy="after_each_task"
)

# Create task
task = Task(
    description="Implement a caching decorator with TTL support",
    agent=reflexive_agent
)

# Execute with automatic reflection
crew = Crew(agents=[reflexive_agent], tasks=[task])
result = crew.kickoff()
```

### With LangChain

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import DuckDuckGoSearchRun
from reflexion.adapters import LangChainReflexion

# Create base LangChain agent
tools = [DuckDuckGoSearchRun()]
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=create_openai_functions_agent(llm, tools, prompt),
    tools=tools
)

# Add reflexion layer
reflexive_executor = LangChainReflexion(
    chain=agent_executor,
    reflection_triggers=["task_timeout", "tool_error"]
)

# Use with automatic reflection
result = reflexive_executor.run(
    "Research the latest developments in AI agent frameworks"
)
```

## 🧠 Memory and Learning

### Enable Episodic Memory

```python
from reflexion import ReflexionAgent
from reflexion.memory import EpisodicMemory

# Create memory system
memory = EpisodicMemory(
    short_term_capacity=100,
    long_term_threshold=0.8,
    embedding_model="text-embedding-ada-002"
)

# Agent with memory
agent = ReflexionAgent(
    llm="gpt-4",
    memory=memory,
    learn_from_similar_tasks=True
)

# The agent will now remember and learn from past experiences
```

### Query Past Learning

```python
# Find similar past experiences
similar_tasks = memory.recall(
    query="debugging memory leaks in Python",
    k=5
)

# Analyze learning patterns
patterns = memory.extract_patterns(
    category="debugging",
    min_occurrences=3
)

print(f"Found {len(patterns)} common debugging patterns")
```

## 📊 Basic Evaluation

### Custom Success Criteria

```python
from reflexion.evaluators import CustomEvaluator

class CodeQualityEvaluator:
    def evaluate(self, task, output):
        # Check for common issues
        has_tests = "def test_" in output
        has_docstring = '"""' in output
        handles_errors = "try:" in output or "except:" in output
        
        score = sum([has_tests, has_docstring, handles_errors]) / 3
        
        return {
            "success": score >= 0.7,
            "score": score,
            "details": {
                "has_tests": has_tests,
                "has_docstring": has_docstring,
                "handles_errors": handles_errors
            }
        }

# Use custom evaluator
agent = ReflexionAgent(
    llm="gpt-4",
    evaluator=CodeQualityEvaluator()
)
```

## 🎛️ Configuration Options

### Environment Variables

Create a `.env` file:

```env
# LLM Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Reflexion Settings
REFLEXION_MAX_ITERATIONS=5
REFLEXION_MEMORY_TYPE=episodic
REFLEXION_VERBOSE=true

# Performance Settings
REFLEXION_TIMEOUT=300
REFLEXION_BATCH_SIZE=10
```

### Configuration File

Create `reflexion_config.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1

reflexion:
  max_iterations: 3
  reflection_type: structured
  dimensions:
    - correctness
    - efficiency
    - readability

memory:
  type: episodic
  short_term_capacity: 50
  long_term_threshold: 0.8

evaluation:
  timeout: 300
  success_threshold: 0.8
```

Load configuration:

```python
from reflexion import ReflexionAgent
from reflexion.config import load_config

config = load_config("reflexion_config.yaml")
agent = ReflexionAgent.from_config(config)
```

## 🔍 Debugging and Troubleshooting

### Enable Verbose Logging

```python
import logging
from reflexion import ReflexionAgent

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

agent = ReflexionAgent(
    llm="gpt-4",
    verbose=True,
    debug=True  # Extra detailed output
)
```

### Common Issues

1. **API Key Issues**
   ```python
   # Verify API key is set
   import os
   print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY', 'Not set')}")
   ```

2. **Memory Issues**
   ```python
   # Clear memory if needed
   agent.memory.clear()
   
   # Check memory usage
   print(f"Memory entries: {len(agent.memory)}")
   ```

3. **Performance Issues**
   ```python
   # Reduce max iterations
   agent = ReflexionAgent(
       llm="gpt-4",
       max_iterations=2,  # Reduce for faster execution
       timeout=60  # Add timeout
   )
   ```

## 📚 Next Steps

Now that you have your first reflexion agent running:

1. **Explore Framework Integration**: Try the [Framework Integration Guide](framework-integration.md)
2. **Set Up Memory Systems**: Learn about [Memory Systems](memory-systems.md)
3. **Configure Evaluation**: Set up proper [Evaluation Metrics](evaluation.md)
4. **Join the Community**: Connect with other developers on [Discord](https://discord.gg/your-org)

## 💡 Pro Tips

- **Start Simple**: Begin with basic reflection before adding complex memory systems
- **Monitor Performance**: Use the built-in metrics to track improvement over time
- **Experiment with Prompts**: The reflexion prompts can be customized for your domain
- **Use Memory Wisely**: Enable memory for agents that will run multiple related tasks

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [GitHub Issues](https://github.com/your-org/reflexion-agent-boilerplate/issues)
3. Ask on [Discord](https://discord.gg/your-org)
4. Email: reflexion@your-org.com

---

*Ready to build more sophisticated agents? Continue to the [Framework Integration Guide](framework-integration.md) →*