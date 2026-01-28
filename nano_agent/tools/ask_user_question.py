"""AskUserQuestion tool for interactive user decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from ..data_structures import TextContent
from .base import Desc, Question, Tool


@dataclass
class AskUserQuestionInput:
    """Input for AskUserQuestionTool."""

    questions: Annotated[
        list[Question],
        Desc("List of questions to ask the user (one question supported by CLI)"),
    ]


@dataclass
class AskUserQuestionTool(Tool):
    """Ask the user a question and return their selection.

    This tool is interactive in the CLI. It supports single-choice and multi-choice
    questions, and can optionally allow a custom freeform response.
    """

    name: str = "AskUserQuestion"
    description: str = """Ask the user a question and return their selection(s).

Supports single-choice or multi-choice options, and optional freeform custom answer
when allowCustom is true.

Input:
- questions: List of Question objects (CLI currently supports 1 question at a time)

Returns:
- Selected option label(s) as text (comma-separated for multi-choice)
- If custom response is chosen, returns the user's custom text
"""

    async def __call__(self, input: AskUserQuestionInput) -> TextContent:
        # CLI handles interactive prompting; this is a fallback for non-interactive use.
        question_count = len(input.questions)
        return TextContent(
            text=(
                "AskUserQuestion requires CLI interaction. "
                f"Received {question_count} question(s)."
            )
        )
