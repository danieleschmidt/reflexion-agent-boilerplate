#!/usr/bin/env python3
"""
Reflexion demo: coding task with unit-test evaluation.

The agent must write a Python function that passes the embedded test suite.
We use PythonCodeEvaluator so the evaluation is ground-truth (actually running
the code) rather than LLM opinion.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/demo_coding_task.py
"""

import logging
import sys
from reflexion import ReflexionAgent
from reflexion.evaluator import LLMEvaluator, PythonCodeEvaluator
from reflexion.llm import AnthropicBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


TASK = """
Write a Python function called `flatten(lst)` that recursively flattens a
nested list of arbitrary depth into a single flat list.

Your answer must be valid Python inside a ```python ... ``` code block.

```python
# TESTS
result = flatten([1, [2, [3, 4], 5], [6, 7]])
assert result == [1, 2, 3, 4, 5, 6, 7], f"Expected [1,2,3,4,5,6,7] but got {result}"

result = flatten([])
assert result == [], f"Expected [] but got {result}"

result = flatten([1, 2, 3])
assert result == [1, 2, 3], f"Expected [1,2,3] but got {result}"

result = flatten([[[[1]]]])
assert result == [1], f"Expected [1] but got {result}"

print("All tests passed!")
```
"""


def print_banner(text: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    llm = AnthropicBackend(model="claude-3-5-haiku-20241022")

    evaluator = PythonCodeEvaluator(
        llm_fallback=LLMEvaluator(llm),
        timeout=10.0,
    )

    agent = ReflexionAgent(
        llm=llm,
        evaluator=evaluator,
        max_iterations=4,
        success_threshold=0.9,
        on_iteration=lambda r: print(
            f"\n[Iteration {r.iteration + 1}] Score: {r.evaluation.score:.1%} "
            f"— {r.evaluation.feedback}\n"
            f"Reflection: {r.reflection_text[:200]}…\n"
        ),
    )

    print_banner("Reflexion Agent Demo — Coding Task")
    print(f"Task:\n{TASK}")

    result = agent.run(TASK)

    print_banner("Final Result")
    print(f"Success:        {result.success}")
    print(f"Iterations:     {result.iterations_used}")
    print(f"Final score:    {result.final_score:.1%}")
    print(f"Improved:       {result.improved}")
    print(f"\nFinal answer:\n{result.final_output}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
