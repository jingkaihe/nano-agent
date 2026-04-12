from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nano_agent.agent import NanoAgent


@dataclass(frozen=True)
class ToolCallContext:
    agent: NanoAgent
    cwd: Path
    model: str
    provider: str
    arguments: dict[str, Any]


class ToolDefinition(ABC):
    name: str

    @abstractmethod
    def description(self, agent: NanoAgent) -> str:
        raise NotImplementedError

    @abstractmethod
    def schema(self, agent: NanoAgent) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        raise NotImplementedError


def function_tool_spec(tool: ToolDefinition, agent: NanoAgent) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description(agent),
            "parameters": tool.schema(agent),
        },
    }
