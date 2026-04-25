"""DownloadSkill tool for accessing skill knowledge on demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from ..data_structures import TextContent
from ..execution_context import ExecutionContext
from .base import Desc, Tool


@dataclass(frozen=True)
class Skill:
    """A skill the agent can download to gain its knowledge.

    Attributes:
        name: The skill's identifier, used as the lookup key.
        description: When to use this skill (compiled into the tool description).
        knowledge: The full skill content returned when downloaded.
    """

    name: str
    description: str
    knowledge: str


@dataclass
class DownloadSkillInput:
    """Input for DownloadSkillTool."""

    name: Annotated[str, Desc("The name of the skill to download")]


@dataclass
class DownloadSkillTool(Tool):
    """Download a skill's knowledge by name.

    Initialize with a list of ``Skill`` objects. Each skill's description
    ("when to use") is compiled into the tool's description so the model knows
    which skills are available, and the knowledge is returned when the tool is
    called with that skill's name.

    Example:
        tool = DownloadSkillTool(skills=[
            Skill(
                name="matrix_kung_fu",
                description="Use when you need to know kung fu",
                knowledge="Block, strike, sweep, ...",
            ),
        ])
    """

    name: str = "DownloadSkill"
    description: str = field(init=False, default="")
    skills: list[Skill] = field(default_factory=list)

    _DESCRIPTION_HEADER: ClassVar[str] = (
        "Download a skill to gain its knowledge. Call this tool with the name "
        "of a skill to retrieve the full knowledge for that skill."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.description = self._build_description()

    def _build_description(self) -> str:
        if not self.skills:
            return f"{self._DESCRIPTION_HEADER}\n\nNo skills are currently registered."
        lines = [self._DESCRIPTION_HEADER, "", "Available skills:"]
        for skill in self.skills:
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    async def __call__(
        self,
        input: DownloadSkillInput,
        execution_context: ExecutionContext | None = None,
    ) -> TextContent:
        for skill in self.skills:
            if skill.name == input.name:
                return TextContent(text=skill.knowledge)
        available = ", ".join(sorted(s.name for s in self.skills)) or "(none)"
        return TextContent(
            text=(
                f"Error: Skill '{input.name}' not found. "
                f"Available skills: {available}"
            )
        )
