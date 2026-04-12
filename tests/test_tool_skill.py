from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from nano_agent.core import SkillInfo
from nano_agent.tools.base import ToolCallContext
from nano_agent.tools.skill import skill_tool

from .helpers import assert_equal, assert_true


class _FakeAgent:
    def __init__(self, directory: Path, content: str = "Skill body") -> None:
        self.active_skills: set[str] = set()
        self._skills = {
            "demo": SkillInfo(
                name="demo",
                description="Demo skill",
                directory=directory,
                content=content,
            )
        }

    def current_skills(self) -> dict[str, SkillInfo]:
        return self._skills


def test_skill_tool_executes_and_tracks_active_skill() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _FakeAgent(Path(tmp))
            context = ToolCallContext(
                agent=agent,
                cwd=Path(tmp),
                model="gpt-5",
                provider="openai",
                arguments={"skill_name": "demo"},
            )
            result = await skill_tool.execute(context)

            assert_true(result["success"], "skill tool should succeed")
            assert_equal(result["skill_name"], "demo")
            assert_true(
                "# Skill: demo" in result["content"],
                "skill output should render content",
            )
            assert_true("demo" in agent.active_skills, "skill should be marked active")

    asyncio.run(_run())


def test_skill_tool_rejects_unknown_skill() -> None:
    async def _run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = _FakeAgent(Path(tmp))
            context = ToolCallContext(
                agent=agent,
                cwd=Path(tmp),
                model="gpt-5",
                provider="openai",
                arguments={"skill_name": "missing"},
            )
            try:
                await skill_tool.execute(context)
            except ValueError as exc:
                assert_true(
                    "unknown skill 'missing'" in str(exc),
                    "missing skill should be rejected",
                )
                return
            raise AssertionError("skill tool should reject unknown skills")

    asyncio.run(_run())
