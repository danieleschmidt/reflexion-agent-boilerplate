# reflexion-agent

A clean, production-quality implementation of the **Reflexion** loop for LLM agents.

> *"Reflexion: Language Agents with Verbal Reinforcement Learning"*  
> Shinn et al., 2023 — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

---

## What is Reflexion?

Reflexion gives an LLM agent the ability to learn from its own mistakes **without gradient updates or fine-tuning**.

```
┌──────────┐    task + reflections    ┌──────────────┐
│  Actor   │ ─────────────────────► │     LLM       │
└──────────┘ ◄──────────────────── └──────────────┘
     │            output
     ▼
┌───────────┐
│ Evaluator │  → score + feedback
└───────────┘
     │  (if score < threshold)
     ▼
┌───────────┐
│ Reflector │  → verbal reflection
└───────────┘
     │
     └──► injected into next actor call
```

Each iteration the agent sees its prior attempt **plus a verbal reflection** on why it failed. This compounds — by iteration 3, the agent has a detailed history of what it tried and what went wrong.

---

## Quickstart

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

### Coding task (evaluated by actually running the code)

```python
from reflexion import ReflexionAgent
from reflexion.evaluator import PythonCodeEvaluator
from reflexion.llm import AnthropicBackend

TASK = """
Write a Python function called `flatten(lst)` that recursively flattens a
nested list.  Put it in a ```python``` code block.

```python
# TESTS
assert flatten([1, [2, [3]]]) == [1, 2, 3]
assert flatten([]) == []
print("ok")
```
"""

llm = AnthropicBackend(model="claude-3-5-haiku-20241022")
agent = ReflexionAgent(
    llm=llm,
    evaluator=PythonCodeEvaluator(),
    max_iterations=4,
    success_threshold=0.9,
)

result = agent.run(TASK)
print(f"Success: {result.success} in {result.iterations_used} iteration(s)")
print(result.final_output)
```

### Open-ended QA (evaluated by an LLM judge)

```python
from reflexion import ReflexionAgent
from reflexion.evaluator import LLMEvaluator
from reflexion.llm import AnthropicBackend

llm = AnthropicBackend()
agent = ReflexionAgent(llm=llm, evaluator=LLMEvaluator(llm), max_iterations=3)

result = agent.run("Explain the CAP theorem with concrete examples, in under 200 words.")
print(result.final_output)
```

### Bring your own evaluator

```python
from reflexion import ReflexionAgent
from reflexion.evaluator import FunctionEvaluator

def my_eval(task: str, output: str) -> tuple[bool, float, str]:
    has_json = output.strip().startswith("{")
    score = 1.0 if has_json else 0.0
    return has_json, score, "Output must be JSON" if not has_json else "Valid JSON"

agent = ReflexionAgent(evaluator=FunctionEvaluator(my_eval))
result = agent.run('Return {"answer": 42} as JSON')
```

---

## Architecture

```
src/reflexion/
├── __init__.py        # Public API
├── agent.py           # ReflexionAgent — orchestrates the loop
├── actor.py           # Actor — generates responses with reflection context
├── evaluator.py       # Evaluators: LLMEvaluator, PythonCodeEvaluator, FunctionEvaluator
├── reflector.py       # Reflector — generates verbal feedback on failure
├── llm.py             # AnthropicBackend (swap in any LLM)
└── types.py           # EvaluationResult, Reflection, ReflexionResult
```

**Key design choices:**
- **No framework dependencies** — pure Python + Anthropic SDK
- **Pluggable evaluators** — unit tests, LLM judge, or your own function
- **Pluggable LLM backend** — swap `AnthropicBackend` for any provider
- **Verbose logging** — `logging.DEBUG` shows every actor/evaluator/reflector call

---

## Running the demos

```bash
# Coding task (runs code in a subprocess — ground truth evaluation)
python examples/demo_coding_task.py

# QA task (LLM-as-judge evaluation)
python examples/demo_qa_task.py
```

---

## Running tests

```bash
pytest                    # all tests (no API key needed — all mocked)
pytest -v tests/test_agent.py   # just the integration tests
```

---

## License

MIT
