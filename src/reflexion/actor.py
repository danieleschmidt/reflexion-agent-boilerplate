"""Actor: generates responses, conditioned on prior reflections."""

from __future__ import annotations

import textwrap
from .types import Reflection


class Actor:
    """LLM-based actor that incorporates prior reflections on retry."""

    SYSTEM = textwrap.dedent("""
        You are a helpful, precise assistant. Complete the task as well as you can.
        When you have prior reflection notes, use them to avoid the same mistakes.
    """).strip()

    def __init__(self, llm):
        self.llm = llm

    def act(self, task: str, reflections: list[Reflection]) -> str:
        messages = self._build_messages(task, reflections)
        return self.llm.complete(messages=messages, system=self.SYSTEM)

    @staticmethod
    def _build_messages(task: str, reflections: list[Reflection]) -> list[dict]:
        """Build the conversation history that includes prior attempts and reflections."""
        messages: list[dict] = []

        for r in reflections:
            # Show the previous attempt
            messages.append({"role": "user", "content": task})
            messages.append({"role": "assistant", "content": r.actor_output})
            # Inject the reflection as a user follow-up
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous answer was not good enough "
                    f"(score: {r.evaluation.score:.1%}).\n\n"
                    f"Reflection:\n{r.reflection_text}\n\n"
                    "Please try again, incorporating this feedback."
                ),
            })

        # Final (current) task request
        if not reflections:
            messages.append({"role": "user", "content": task})
        # If we have reflections the last message already asks for a retry.

        return messages
