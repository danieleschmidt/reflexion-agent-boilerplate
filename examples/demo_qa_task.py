#!/usr/bin/env python3
"""
Reflexion demo: open-ended QA with LLM-as-judge evaluation.

This shows the loop working on a knowledge / reasoning task where
there is no ground-truth unit test — the evaluator is another LLM call.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/demo_qa_task.py
"""

import logging
import sys
from reflexion import ReflexionAgent
from reflexion.evaluator import LLMEvaluator
from reflexion.llm import AnthropicBackend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


TASK = """
Explain the CAP theorem to a software engineering intern in plain language.
Your explanation must:
1. Define all three letters (Consistency, Availability, Partition Tolerance)
2. Give a concrete real-world example of a system that chooses CA, CP, and AP
3. Explain WHY you can only guarantee two of the three
Be concise (under 300 words) but complete.
"""


def main():
    llm = AnthropicBackend(model="claude-3-5-haiku-20241022")
    evaluator = LLMEvaluator(llm)

    agent = ReflexionAgent(
        llm=llm,
        evaluator=evaluator,
        max_iterations=3,
        success_threshold=0.8,
    )

    print("\n" + "=" * 70)
    print("  Reflexion Agent Demo — QA Task (LLM-as-judge)")
    print("=" * 70)
    print(f"Task:\n{TASK}")

    result = agent.run(TASK)

    print("\n" + "=" * 70)
    print("  Final Result")
    print("=" * 70)
    print(f"Success:     {result.success}")
    print(f"Iterations:  {result.iterations_used}")
    print(f"Score:       {result.final_score:.1%}")

    if result.reflections:
        print("\nReflections made:")
        for r in result.reflections:
            print(f"  Iter {r.iteration + 1}: score={r.evaluation.score:.1%} | {r.evaluation.feedback}")

    print(f"\nFinal answer:\n{result.final_output}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
