from __future__ import annotations

from typing import Any

from ..core import skill_description
from .base import ToolCallContext, ToolDefinition
from .schemas import skill_schema


class SkillTool(ToolDefinition):
    name = "skill"

    def description(self, agent: Any) -> str:
        return skill_description(agent.current_skills())

    def schema(self, agent: Any) -> dict[str, Any]:
        return skill_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        skill_name = context.arguments.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name is required")
        skills = context.agent.current_skills()
        if skill_name not in skills:
            available = ", ".join(sorted(skills)) or "none"
            raise ValueError(
                f"unknown skill '{skill_name}'. Available skills: {available}"
            )
        if skill_name in context.agent.active_skills:
            raise ValueError(f"skill '{skill_name}' is already active")
        context.agent.active_skills.add(skill_name)
        skill = skills[skill_name]
        content = f"""# Skill: {skill.name}

The skill directory is located at: {skill.directory}

{skill.content}"""
        return {
            "success": True,
            "skill_name": skill.name,
            "directory": str(skill.directory),
            "content": content,
        }


skill_tool = SkillTool()

__all__ = ["SkillTool", "skill_description", "skill_schema", "skill_tool"]
