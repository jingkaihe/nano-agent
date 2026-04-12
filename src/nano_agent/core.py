from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import httpx
import yaml
from jinja2 import Environment, StrictUndefined
from rich.console import Console
from rich.table import Table

import nano_agent.core as core_module

COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_DEVICE_URL = "https://github.com/login/device/code"
COPILOT_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_SCOPES = ["read:user", "user:email", "copilot"]
COPILOT_EDITOR_VERSION = "vscode/1.102.0"
COPILOT_INTEGRATION_ID = "vscode-chat"
COPILOT_CHAT_PLUGIN_VERSION = "copilot-chat/0.26.7"
COPILOT_CHAT_USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_OPENAI_USER_AGENT = "GithubCopilot/1.342.0"
COPILOT_GITHUB_API_VERSION = "2025-04-01"
COPILOT_VSCODE_USER_AGENT_LIBRARY_VERSION = "electron-fetch"

DEFAULT_MODEL = os.getenv("NANO_AGENT_MODEL", "gpt-4.1")
DEFAULT_PROVIDER = os.getenv("NANO_AGENT_PROVIDER", "copilot")
DEFAULT_REASONING_EFFORT = os.getenv("NANO_AGENT_REASONING_EFFORT", "none")
DEFAULT_PLATFORM = os.getenv("NANO_AGENT_PLATFORM", DEFAULT_PROVIDER)
CONTEXT_FILENAME = "AGENTS.md"
BANNED_COMMANDS = {"vim", "view", "less", "more", "cd"}
MAX_TOOL_OUTPUT_BYTES = 32 * 1024
BASH_MAX_OUTPUT_TOKENS = 10_000
BASH_APPROX_BYTES_PER_TOKEN = 4
VIEW_IMAGE_MAX_WIDTH = 2048
VIEW_IMAGE_MAX_HEIGHT = 768
ARCHIVE_DIR = Path.home() / ".nano-agent" / "web-archives"
HISTORY_DIR = Path.home() / ".nano-agent" / "history"
MODELS_CACHE_TTL_SECONDS = 300
JINJA = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
MAX_READ_OUTPUT_BYTES = 100_000
MAX_READ_LINE_CHARACTERS = 2000
MAX_READ_LINE_LIMIT = 2000
GREP_MAX_RESULTS = 100
GREP_MAX_LINE_LENGTH = 300
GREP_MAX_OUTPUT_BYTES = 50 * 1024
GLOB_MAX_RESULTS = 100
SEARCH_TOOL_TIMEOUT_SECONDS = 60
COPILOT_MODELS_CACHE: list[dict[str, Any]] | None = None
COPILOT_MODELS_CACHE_EXPIRES_AT: float | None = None

ALL_TOOL_NAMES = (
    "bash",
    "apply_patch",
    "read_file",
    "write_file",
    "edit_file",
    "grep",
    "glob",
    "skill",
    "web_fetch",
    "view_image",
)

PLATFORM_NAMES = ("copilot", "openai", "anthropic")
DEFAULT_RESPONSE_TOOL_NAMES = [
    "bash",
    "apply_patch",
    "skill",
    "web_fetch",
    "view_image",
]
DEFAULT_CHAT_TOOL_NAMES = [
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "grep",
    "glob",
    "skill",
    "web_fetch",
    "view_image",
]


class CopilotAuthError(Exception):
    pass

def credentials_path() -> Path:
    raw = os.getenv("NANO_AGENT_COPILOT_CREDS")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nano-agent" / "copilot.json"


def models_cache_path() -> Path:
    raw = os.getenv("NANO_AGENT_MODELS_CACHE")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nano-agent" / "models.json"


def normalize_session_id(value: str) -> str:
    session_id = value.strip()
    if session_id.endswith(".json"):
        session_id = session_id[:-5]
    if not session_id:
        raise ValueError("resume id is required")
    if "/" in session_id or "\\" in session_id:
        raise ValueError("resume id must be a session uuid, not a path")
    return session_id


def conversation_history_path(session_id: str) -> Path:
    return core_module.HISTORY_DIR / f"{normalize_session_id(session_id)}.json"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def first_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def summarize_prompt(text: str, max_words: int = 10) -> str:
    words = text.split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " …"


def load_conversation_history(session_id: str) -> dict[str, Any]:
    normalized = normalize_session_id(session_id)
    path = conversation_history_path(normalized)
    if not path.exists():
        raise ValueError(f"history session not found: {normalized}")

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"history session is invalid: {normalized}")

    model = data.get("model")
    provider = data.get("provider")
    reasoning_effort = data.get("reasoning_effort")
    weak_model = data.get("weak_model")
    allowed_tools = normalize_allowed_tools(data.get("allowed_tools"))
    raw_history = data.get("history")
    history: dict[str, Any] = raw_history if isinstance(raw_history, dict) else {}
    messages = history.get("messages", data.get("messages", []))
    responses_items = history.get("responses_items", data.get("responses_items", []))
    usage = history.get("usage", data.get("usage", {}))
    stored_id = data.get("id")
    created_at = data.get("created_at")
    updated_at = data.get("updated_at")

    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"history session is missing model: {normalized}")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"history session is missing provider: {normalized}")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        reasoning_effort = None
    if weak_model is not None and not isinstance(weak_model, str):
        weak_model = None
    if not isinstance(messages, list):
        raise ValueError(f"history session has invalid messages: {normalized}")
    if not isinstance(responses_items, list):
        raise ValueError(f"history session has invalid responses_items: {normalized}")
    if not isinstance(usage, dict):
        usage = {}
    if not isinstance(created_at, str) or not created_at.strip():
        created_at = (
            datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if not isinstance(updated_at, str) or not updated_at.strip():
        updated_at = (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    return {
        "id": normalize_session_id(stored_id)
        if isinstance(stored_id, str) and stored_id.strip()
        else normalized,
        "model": model,
        "provider": provider,
        "reasoning_effort": reasoning_effort,
        "weak_model": weak_model,
        "allowed_tools": allowed_tools,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
        "responses_items": responses_items,
        "usage": usage,
    }


def list_conversation_histories() -> list[dict[str, Any]]:
    core_module.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []
    for path in sorted(core_module.HISTORY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        raw_history = data.get("history")
        history: dict[str, Any] = raw_history if isinstance(raw_history, dict) else {}
        messages = history.get("messages", data.get("messages", []))
        if not isinstance(messages, list):
            messages = []

        session_id = data.get("id")
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = path.stem

        created_at = data.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            created_at = (
                datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        updated_at = data.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at.strip():
            updated_at = (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        sessions.append(
            {
                "id": normalize_session_id(session_id),
                "provider": data.get("provider") or "",
                "model": data.get("model") or "",
                "created_at": created_at,
                "updated_at": updated_at,
                "first_prompt": summarize_prompt(first_user_prompt(messages)),
            }
        )

    sessions.sort(key=lambda item: cast(str, item["updated_at"]), reverse=True)
    return sessions


def latest_conversation_session_id() -> str:
    sessions = list_conversation_histories()
    if not sessions:
        raise ValueError("no saved conversation history to resume")
    return cast(str, sessions[0]["id"])


def _coerce_expires_at(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.isdigit():
            return float(value)
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    raise CopilotAuthError(f"Unsupported expires_at value: {value!r}")


def _load_models_cache_data(path: Path | None = None) -> dict[str, Any]:
    target = path or models_cache_path()
    try:
        data = json.loads(target.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_models_cache_data(data: dict[str, Any], path: Path | None = None) -> None:
    target = path or models_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")


def load_cached_provider_models(
    provider: str,
    *,
    ttl_seconds: int = MODELS_CACHE_TTL_SECONDS,
    path: Path | None = None,
    now: float | None = None,
) -> tuple[list[dict[str, Any]] | None, float | None]:
    data = _load_models_cache_data(path)
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return None, None

    entry = providers.get(provider.strip().lower())
    if not isinstance(entry, dict):
        return None, None

    fetched_at = entry.get("fetched_at")
    try:
        fetched_at_ts = _coerce_expires_at(fetched_at)
    except Exception:
        return None, None

    current_time = time.time() if now is None else now
    expires_at = fetched_at_ts + ttl_seconds
    if current_time >= expires_at:
        return None, expires_at

    models = entry.get("models")
    if not isinstance(models, list):
        return None, expires_at
    return [item for item in models if isinstance(item, dict)], expires_at


def save_cached_provider_models(
    provider: str,
    models: list[dict[str, Any]],
    *,
    fetched_at: str | None = None,
    path: Path | None = None,
) -> float:
    timestamp = fetched_at or iso_now()
    data = _load_models_cache_data(path)
    providers = data.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    providers[provider.strip().lower()] = {
        "fetched_at": timestamp,
        "models": [dict(item) for item in models if isinstance(item, dict)],
    }
    data["providers"] = providers
    _save_models_cache_data(data, path)
    return _coerce_expires_at(timestamp) + MODELS_CACHE_TTL_SECONDS


def _set_copilot_models_cache(
    models: list[dict[str, Any]], *, expires_at: float | None
) -> None:
    global COPILOT_MODELS_CACHE
    global COPILOT_MODELS_CACHE_EXPIRES_AT
    COPILOT_MODELS_CACHE = [dict(item) for item in models]
    COPILOT_MODELS_CACHE_EXPIRES_AT = expires_at


def load_copilot_credentials() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        raise CopilotAuthError(
            f"Copilot credentials not found at {path}. Run 'nano-agent login' first."
        )
    return json.loads(path.read_text())


def save_copilot_credentials(
    credentials: dict[str, Any], output_path: Path | None = None
) -> Path:
    path = output_path or credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credentials, indent=2) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def refresh_copilot_token(
    creds: dict[str, Any], *, force: bool = False
) -> tuple[str, dict[str, Any]]:
    from .platforms import copilot_token_exchange_headers

    refresh_threshold = time.time() + 10 * 60
    expires_at = _coerce_expires_at(creds["copilot_expires_at"])
    if not force and expires_at > refresh_threshold:
        return creds["copilot_token"], creds

    response = httpx.get(
        COPILOT_EXCHANGE_URL,
        headers=copilot_token_exchange_headers(creds["access_token"]),
        timeout=30,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise CopilotAuthError(
            "Failed to refresh Copilot token. Run `nano-agent login` to re-authenticate."
        ) from exc
    token_data = response.json()
    creds["copilot_token"] = token_data["token"]
    creds["copilot_expires_at"] = token_data["expires_at"]
    save_copilot_credentials(creds)
    return creds["copilot_token"], creds

def is_git_repo(cwd: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def collect_contexts(cwd: Path) -> list[tuple[Path, str]]:
    contexts: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    home_context = Path.home() / ".nano-agent" / CONTEXT_FILENAME
    if home_context.exists():
        seen.add(home_context.resolve())
        contexts.append((home_context, home_context.read_text(errors="replace")))
    current = cwd.resolve()
    dirs = [current, *current.parents]
    for directory in reversed(dirs):
        candidate = directory / CONTEXT_FILENAME
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                contexts.append((candidate, candidate.read_text(errors="replace")))
    return contexts


def _is_gpt_41_model(model: str | None) -> bool:
    if model is None:
        return False
    normalized = model.strip().lower()
    return normalized == "gpt-4.1" or normalized.startswith("gpt-4.1-")


def default_weak_model(provider: str, model: str | None = None) -> str:
    normalized_provider = provider.strip().lower()
    normalized_model = (model or "").strip()
    if normalized_provider == "copilot" and _is_anthropic_like_model(normalized_model):
        return "claude-haiku-4.5"
    if normalized_provider in {"copilot", "openai"}:
        return "gpt-4.1"
    return (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _default_system_instructions(
    model: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    allowed_tools: list[str] | None = None,
) -> str:
    from .tools import tool_guidance_text

    return (
        """You are an interactive CLI tool that helps with software engineering and production operations tasks. Please follows the instructions and tools below to help the user.

# Tone and Style
* Be concise, direct and to the point. When you are performing a non-trivial task, you should explain what it does and why you are doing it. This is especially important when you are making changes to the user's system.
* Your output will be rendered as markdown, please use Github Flavored Markdown for formatting.
* Output text to communicate with the users. DO NOT use `bash` or code comment as a way of communicating with the users.
* You should limit the output (not including the tool call) to 2-3 sentences while maintaining the correctness, quality and helpfulness.
* You should not provide answer with unecessary preamble or postamble unless you are asked to do so.
* Avoid using bullet points unless there is a list of items you need to present.
* When using files as references, always use absolute paths with line number ranges where applicable.

# Proactiveness
You only need to be proactive when the user explicitly requests you to do so. Generally you need to strike a balance between:
* Doing exactly what the user asks for, and make sure that you follow through the actions to fullfill the request.
* Not surprising the user with additional activities without consent from the user.

"""
        + "\n\n# Tooling Guidance\n"
        + tool_guidance_text(model or DEFAULT_MODEL, provider, allowed_tools)
    )


def _gpt_41_system_instructions(
    model: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    allowed_tools: list[str] | None = None,
) -> str:
    from .tools import tool_guidance_text

    return (
        """You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user.

Your thinking should be thorough and so it's fine if it's very long. However, avoid unnecessary repetition and verbosity. You should be concise, but thorough.

You MUST iterate and keep going until the problem is solved.

You have everything you need to resolve this problem. I want you to fully solve this autonomously before coming back to me.

Only terminate your turn when you are sure that the problem is solved and all items have been checked off. Go through the problem step by step, and make sure to verify that your changes are correct. NEVER end your turn without having truly and completely solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

When the task depends on external URLs, third party packages, online documentation, APIs, or linked reference material, you should use `web_fetch` to gather the relevant information from the provided URL's and any important links you discover.

Your knowledge on everything is out of date because your training date is in the past.

When you need current information about libraries, packages, frameworks, APIs, dependencies, or external services, use `web_fetch` to verify your understanding against up-to-date sources. It is not enough to just search; read the relevant pages and recursively gather the information you need by following important links.

Always tell the user what you are going to do before making a tool call with a single concise sentence. This will help them understand what you are doing and why.

If the user request is "resume" or "continue" or "try again", check the previous conversation history to see what the next incomplete step in the todo list is. Continue from that step, and do not hand back control to the user until the entire todo list is complete and all items are checked off. Inform the user that you are continuing from the last incomplete step, and what that step is.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

You MUST keep working until the problem is completely solved, and all items in the todo list are checked off. Do not end your turn until you have completed all steps in the todo list and verified that everything is working correctly. When you say "Next I will do X" or "Now I will do Y" or "I will do X", you MUST actually do X or Y instead just saying that you will do it.

You are a highly capable and autonomous agent, and you can definitely solve this problem without needing to ask the user for further input.

# Workflow
1. Fetch any URL's provided by the user using the `web_fetch` tool.
2. Understand the problem deeply. Carefully read the issue and think critically about what is required. Break the problem into manageable parts. Consider the following:
   - What is the expected behavior?
   - What are the edge cases?
   - What are the potential pitfalls?
   - How does this fit into the larger context of the codebase?
   - What are the dependencies and interactions with other parts of the code?
3. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
4. Research the problem on the internet when current external information is relevant.
5. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps. Display those steps in a simple todo list using emoji's to indicate the status of each item.
6. Implement the fix incrementally. Make small, testable code changes.
7. Debug as needed. Use debugging techniques to isolate and resolve issues.
8. Test frequently. Run tests after each change to verify correctness.
9. Iterate until the root cause is fixed and all tests pass.
10. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there may be hidden tests that must also pass before the solution is truly complete.

Refer to the detailed sections below for more information on each step.

## 1. Fetch Provided URLs
- If the user provides a URL, use the `web_fetch` tool to retrieve the content of the provided URL.
- After fetching, review the content returned by the fetch tool.
- If you find any additional URLs or links that are relevant, use the `web_fetch` tool again to retrieve those links.
- Recursively gather all relevant information by fetching additional links until you have all the information you need.

## 2. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 3. Codebase Investigation
- Explore relevant files and directories.
- Search for key functions, classes, or variables related to the issue.
- Read and understand relevant code snippets.
- Identify the root cause of the problem.
- Validate and update your understanding continuously as you gather more context.

## 4. Internet Research
- Use `web_fetch` whenever you need up-to-date external information.
- If search engine results are necessary, fetch the search results page and then fetch the most relevant linked sources.
- Do not rely only on summaries in search results when the underlying documentation or source material is available.
- As you fetch each link, read the content thoroughly and fetch any additional links within the content that are relevant to the problem.
- Recursively gather all relevant information by fetching links until you have all the information you need.

## 5. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem.
- Create a todo list in markdown format to track your progress.
- Each time you complete a step, check it off using `[x]` syntax.
- Each time you check off a step, display the updated todo list to the user.
- Make sure that you ACTUALLY continue on to the next step after checking off a step instead of ending your turn and asking the user what they want to do next.

## 6. Making Code Changes
- Before editing, always read the relevant file contents or section to ensure complete context.
- Use `bash` for shell commands, inspection, repo navigation, and test execution. For filesystem search, use `fd` for file discovery and `rg` for content search via `bash`.
- Use `read_file`, `write_file`, and `edit_file` for file changes when those tools are available; otherwise use `apply_patch`.
- Use `skill` when a matching skill is relevant; if a listed skill clearly applies, invoke it before other work.
- If a patch is not applied correctly, attempt to reapply it.
- Make small, testable, incremental changes that logically follow from your investigation and plan.
- Do not mention or rely on tools that are not available in this environment.

## 7. Debugging
- Make code changes only if you have high confidence they can solve the problem.
- When debugging, try to determine the root cause rather than addressing symptoms.
- Debug for as long as needed to identify the root cause and identify a fix.
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening.
- To test hypotheses, you can also add test statements or functions.
- Revisit your assumptions if unexpected behavior occurs.

# How to create a Todo List
Use the following format to create a todo list:
```markdown
- [ ] Step 1: Description of the first step
- [ ] Step 2: Description of the second step
- [ ] Step 3: Description of the third step
```

Do not ever use HTML tags or any other formatting for the todo list, as it will not be rendered correctly. Always use the markdown format shown above. Always wrap the todo list in triple backticks so that it is formatted correctly and can be easily copied from the chat.

Always show the completed todo list to the user as the last item in your message, so that they can see that you have addressed all of the steps.

# Communication Guidelines
Always communicate clearly and concisely in a casual, friendly yet professional tone.
<examples>
"Let me fetch the URL you provided to gather more information."
"Ok, I've got all of the information I need on the API and I know how to use it."
"Now, I will search the codebase for the function that handles this behavior."
"I need to update several files here - stand by"
"OK! Now let's run the tests to make sure everything is working correctly."
"Whelp - I see we have some problems. Let's fix those up."
</examples>

- Respond with clear, direct answers. Use bullet points and code blocks for structure.
- Avoid unnecessary explanations, repetition, and filler.
- Always write code directly to the correct files.
- Do not display code to the user unless they specifically ask for it.
- Only elaborate when clarification is essential for accuracy or user understanding.

# Writing Prompts
If you are asked to write a prompt, you should always generate the prompt in markdown format.

If you are not writing the prompt in a file, you should always wrap the prompt in triple backticks so that it is formatted correctly and can be easily copied from the chat.

Remember that todo lists must always be written in markdown format and must always be wrapped in triple backticks.

# Git
If the user tells you to stage and commit, you may do so.

You are NEVER allowed to stage and commit files automatically.
"""
        + "\n\n# Tooling Guidance\n"
        + tool_guidance_text(model or DEFAULT_MODEL, provider, allowed_tools)
    )


def build_system_prompt(
    cwd: Path,
    model: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    allowed_tools: list[str] | None = None,
) -> str:
    contexts = collect_contexts(cwd)
    instructions = (
        _gpt_41_system_instructions(model, provider, allowed_tools)
        if _is_gpt_41_model(model)
        else _default_system_instructions(model, provider, allowed_tools)
    )
    template = JINJA.from_string(
        """{{ instructions }}

# Context
If the current working directory contains a `{{ context_filename }}` file, it will be automatically loaded as a context. Use it for:
* Understanding the structure, organisation and tech stack of the project.
* Keeping record of commands (for linting, testing, building etc) that you have to use repeatedly.
* Recording coding style, conventions and preferences of the project.

If you find a new command that you have to use repeatedly, you can add it to the `{{ context_filename }}` file.
If you have made any significant changes to the project structure, or modified the tech stack, you should update the `{{ context_filename }}` file.

# System Information
Here is the system information:
<system-information>
Current working directory: {{ cwd }}
Is this a git repository? {{ is_git_repo }}
Operating system: {{ os_name }} {{ os_version }}
Date: {{ date }}
</system-information>
{% if contexts %}

Here are some useful context to help you solve the user's problem.
When you are working in these directories, make sure that you are following the guidelines provided in the context.
Note that the contexts in $HOME/.nano-agent/ are universally applicable.
{% for path, content in contexts %}
<context filename="{{ path }}", dir="{{ path.parent }}">
{{ content.rstrip() }}
</context>
{% endfor %}
{% endif %}
"""
    )
    return (
        template.render(
            instructions=instructions,
            context_filename=CONTEXT_FILENAME,
            cwd=cwd,
            is_git_repo=str(is_git_repo(cwd)).lower(),
            os_name=platform.system(),
            os_version=platform.release(),
            date=time.strftime("%Y-%m-%d"),
            contexts=contexts,
        ).strip()
        + "\n"
    )


@dataclass
class SkillInfo:
    name: str
    description: str
    directory: Path
    content: str


@dataclass
class SessionUsage:
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_total_tokens: int = 0
    last_cached_input_tokens: int = 0
    last_reasoning_tokens: int = 0
    last_api: str | None = None
    last_model: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "SessionUsage":
        if not isinstance(value, dict):
            return cls()
        return cls(
            request_count=_coerce_usage_int(value.get("request_count")),
            input_tokens=_coerce_usage_int(value.get("input_tokens")),
            output_tokens=_coerce_usage_int(value.get("output_tokens")),
            total_tokens=_coerce_usage_int(value.get("total_tokens")),
            cached_input_tokens=_coerce_usage_int(value.get("cached_input_tokens")),
            reasoning_tokens=_coerce_usage_int(value.get("reasoning_tokens")),
            last_input_tokens=_coerce_usage_int(value.get("last_input_tokens")),
            last_output_tokens=_coerce_usage_int(value.get("last_output_tokens")),
            last_total_tokens=_coerce_usage_int(value.get("last_total_tokens")),
            last_cached_input_tokens=_coerce_usage_int(
                value.get("last_cached_input_tokens")
            ),
            last_reasoning_tokens=_coerce_usage_int(value.get("last_reasoning_tokens")),
            last_api=value.get("last_api")
            if isinstance(value.get("last_api"), str)
            else None,
            last_model=value.get("last_model")
            if isinstance(value.get("last_model"), str)
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_cached_input_tokens": self.last_cached_input_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "last_api": self.last_api,
            "last_model": self.last_model,
        }

    def record(self, usage: dict[str, int], *, api: str, model: str) -> None:
        self.request_count += 1
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.total_tokens += usage["total_tokens"]
        self.cached_input_tokens += usage["cached_input_tokens"]
        self.reasoning_tokens += usage["reasoning_tokens"]
        self.last_input_tokens = usage["input_tokens"]
        self.last_output_tokens = usage["output_tokens"]
        self.last_total_tokens = usage["total_tokens"]
        self.last_cached_input_tokens = usage["cached_input_tokens"]
        self.last_reasoning_tokens = usage["reasoning_tokens"]
        self.last_api = api
        self.last_model = model


def parse_skill_file(path: Path) -> SkillInfo | None:
    text = path.read_text(errors="replace")
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not match:
        return None
    frontmatter = yaml.safe_load(match.group(1)) or {}
    content = match.group(2).strip()
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not name or not description:
        return None
    return SkillInfo(
        name=name, description=description, directory=path.parent, content=content
    )


def _discover_skills_from_dir(
    base: Path, prefix: str, found: dict[str, SkillInfo]
) -> None:
    if not base.exists() or not base.is_dir():
        return
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        skill = parse_skill_file(skill_path)
        if not skill:
            continue
        full_name = f"{prefix}{skill.name}"
        if full_name not in found:
            skill.name = full_name
            found[full_name] = skill


def discover_skills(cwd: Path) -> dict[str, SkillInfo]:
    found: dict[str, SkillInfo] = {}
    home = Path.home()

    standalone_dirs = [
        cwd / ".agents" / "skills",
        home / ".agents" / "skills",
    ]

    for directory in standalone_dirs:
        _discover_skills_from_dir(directory, "", found)

    return found


def skill_description(skills: dict[str, SkillInfo]) -> str:
    template = JINJA.from_string(
        """When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

# Usage
- Use this tool with the skill name only
- Examples:
  - "kernel-dev" - invoke the kernel-dev skill
  - "xlsx" - invoke the xlsx skill

## Important
- When a skill is relevant, you must invoke this tool IMMEDIATELY as your first action
- NEVER just announce or mention a skill in your text response without actually calling this tool
- This is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- Only use skills listed in "Available Skills" below
- Do not invoke a skill that is already running
- Each skill has a directory containing supporting files (references, examples, scripts, templates) that you can inspect using fd/rg/sed/cat via bash
- Do NOT modify any files in the skill directory - treat skill contents as read-only
- If you need to modify a script or template from the skill directory, copy it to the working directory first and update it using write_file / edit_file / apply_patch as appropriate
- For Python scripts, use uv for managing dependencies, preferably uv with inline metadata dependencies if the script to run is a single file - do NOT install packages using system pip

## Available Skills

{% if skills %}
{% for skill in skills %}
### {{ skill.name }}
- **Description**: {{ skill.description }}
- **Directory**: `{{ skill.directory }}`

{% endfor %}
{% else %}
Skills are currently not available.
{% endif %}
"""
    )
    return template.render(skills=[skills[name] for name in sorted(skills)]).rstrip()



def _to_plain_data(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return value


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_usage_metrics(value: Any) -> dict[str, int] | None:
    plain = _to_plain_data(value)
    if not isinstance(plain, dict):
        return None

    if "input_tokens" in plain or "output_tokens" in plain:
        input_tokens = _coerce_usage_int(plain.get("input_tokens"))
        output_tokens = _coerce_usage_int(plain.get("output_tokens"))
        total_tokens = _coerce_usage_int(plain.get("total_tokens")) or (
            input_tokens + output_tokens
        )
        raw_input_details = plain.get("input_tokens_details")
        input_details: dict[str, Any] = (
            raw_input_details if isinstance(raw_input_details, dict) else {}
        )
        raw_output_details = plain.get("output_tokens_details")
        output_details: dict[str, Any] = (
            raw_output_details if isinstance(raw_output_details, dict) else {}
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _coerce_usage_int(
                input_details.get("cached_tokens")
            ),
            "reasoning_tokens": _coerce_usage_int(
                output_details.get("reasoning_tokens")
            ),
        }

    if "prompt_tokens" in plain or "completion_tokens" in plain:
        input_tokens = _coerce_usage_int(plain.get("prompt_tokens"))
        output_tokens = _coerce_usage_int(plain.get("completion_tokens"))
        total_tokens = _coerce_usage_int(plain.get("total_tokens")) or (
            input_tokens + output_tokens
        )
        raw_prompt_details = plain.get("prompt_tokens_details")
        prompt_details: dict[str, Any] = (
            raw_prompt_details if isinstance(raw_prompt_details, dict) else {}
        )
        raw_completion_details = plain.get("completion_tokens_details")
        completion_details: dict[str, Any] = (
            raw_completion_details if isinstance(raw_completion_details, dict) else {}
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": _coerce_usage_int(
                prompt_details.get("cached_tokens")
            ),
            "reasoning_tokens": _coerce_usage_int(
                completion_details.get("reasoning_tokens")
            ),
        }

    return None


def _format_int(value: int) -> str:
    return f"{value:,}"


def _legacy_context_window_limit(model: str) -> int | None:
    normalized = model.strip().lower()
    if "claude" in normalized or "anthropic" in normalized:
        return 200_000
    return None


def _context_window_limit(model: str, provider: str = "copilot") -> int | None:
    return model_context_window_limit(model, provider=provider)


def _is_copilot_claude_chat(provider: str, model: str) -> bool:
    return (
        _is_copilot_provider(provider)
        and _is_anthropic_like_model(model)
        and not should_use_responses_api(model)
    )


def _format_ratio(value: int, total: int | None) -> str:
    if not total or total <= 0:
        return _format_int(value)
    percentage = value / total * 100
    return f"{_format_int(value)} / {_format_int(total)} ({percentage:.1f}%)"


def _extract_text_fragments(value: Any) -> list[str]:
    plain = _to_plain_data(value)
    if plain is None:
        return []
    if isinstance(plain, str):
        return [plain]
    if isinstance(plain, list):
        fragments: list[str] = []
        for item in plain:
            fragments.extend(_extract_text_fragments(item))
        return fragments
    if isinstance(plain, dict):
        fragments: list[str] = []
        for key in ("text", "content", "value"):
            if isinstance(plain.get(key), str):
                fragments.append(plain[key])
        return fragments
    return []


def _get_delta_fragments(delta: Any, keys: list[str]) -> list[str]:
    plain = _to_plain_data(delta)
    if not isinstance(plain, dict):
        return []
    fragments: list[str] = []
    for key in keys:
        fragments.extend(_extract_text_fragments(plain.get(key)))
    return fragments


def _merge_stream_tool_call(
    accumulator: dict[int, dict[str, Any]], tool_call: Any
) -> None:
    plain = _to_plain_data(tool_call)
    if not isinstance(plain, dict):
        return
    index = plain.get("index", 0)
    current = accumulator.setdefault(
        int(index),
        {
            "id": plain.get("id"),
            "type": plain.get("type", "function"),
            "function": {"name": "", "arguments": ""},
        },
    )
    if plain.get("id"):
        current["id"] = plain["id"]
    if plain.get("type"):
        current["type"] = plain["type"]
    function = plain.get("function") or {}
    if isinstance(function, dict):
        if function.get("name"):
            current["function"]["name"] += function["name"]
        if function.get("arguments"):
            current["function"]["arguments"] += function["arguments"]


def _ordered_tool_calls(accumulator: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [accumulator[index] for index in sorted(accumulator)]


def should_use_responses_api(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("gpt-5") or "codex" in normalized


def _is_copilot_provider(provider: str) -> bool:
    return provider.strip().lower() == "copilot"


def _is_anthropic_like_model(model: str) -> bool:
    normalized = model.strip().lower()
    return "anthropic" in normalized or "claude" in normalized


def should_use_anthropic_messages_api(provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower()
    return normalized_provider == "anthropic" or (
        _is_copilot_provider(provider) and _is_anthropic_like_model(model)
    )


def apply_copilot_ephemeral_cache(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    endpoint: str,
) -> list[dict[str, Any]]:
    from .platforms import model_supports_endpoint

    if not _is_copilot_provider(provider) or not model_supports_endpoint(
        endpoint, model, provider=provider
    ):
        return messages
    system_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "system"
    ][:2]
    final_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") != "system"
    ][-2:]

    target_indexes: list[int] = []
    seen_indexes: set[int] = set()
    for index in [*system_indexes, *final_indexes]:
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        target_indexes.append(index)

    if not target_indexes:
        return messages

    patched = list(messages)
    for index in target_indexes:
        original = patched[index]
        updated = dict(original)
        content = updated.get("content")
        if isinstance(content, list) and content:
            last_part = content[-1]
            if isinstance(last_part, dict):
                updated_content = list(content)
                updated_part = dict(last_part)
                existing = updated_part.get("copilot_cache_control")
                updated_part["copilot_cache_control"] = {
                    **(existing if isinstance(existing, dict) else {}),
                    "type": "ephemeral",
                }
                updated_content[-1] = updated_part
                updated["content"] = updated_content
                patched[index] = updated
                continue

        existing = updated.get("copilot_cache_control")
        updated["copilot_cache_control"] = {
            **(existing if isinstance(existing, dict) else {}),
            "type": "ephemeral",
        }
        patched[index] = updated

    return patched


def _copilot_cache_control(value: dict[str, Any]) -> dict[str, Any] | None:
    existing = value.get("copilot_cache_control")
    if not isinstance(existing, dict):
        return None
    return dict(existing)


def _apply_copilot_cache_control_to_block(
    block: dict[str, Any], cache_control: dict[str, Any] | None
) -> dict[str, Any]:
    if cache_control is None:
        return block
    updated = dict(block)
    existing = updated.get("copilot_cache_control")
    updated["copilot_cache_control"] = {
        **(existing if isinstance(existing, dict) else {}),
        **cache_control,
    }
    return updated


def _anthropic_block_from_internal_block(block: dict[str, Any]) -> dict[str, Any]:
    updated = dict(block)
    existing = updated.pop("copilot_cache_control", None)
    if isinstance(existing, dict):
        existing_cache_control = updated.get("cache_control")
        current_cache_control: dict[str, Any] = (
            dict(existing_cache_control)
            if isinstance(existing_cache_control, dict)
            else {}
        )
        updated["cache_control"] = {
            **current_cache_control,
            **existing,
        }
    return updated


def _apply_copilot_cache_control_to_anthropic_block(
    block: dict[str, Any], cache_control: dict[str, Any] | None
) -> dict[str, Any]:
    updated = _anthropic_block_from_internal_block(block)
    if cache_control is None:
        return updated
    existing = updated.get("cache_control")
    updated["cache_control"] = {
        **(existing if isinstance(existing, dict) else {}),
        **cache_control,
    }
    return updated


def _apply_message_copilot_cache_control_to_last_block(
    blocks: list[dict[str, Any]], message: dict[str, Any]
) -> list[dict[str, Any]]:
    cache_control = _copilot_cache_control(message)
    if cache_control is None or not blocks:
        return blocks
    updated = list(blocks)
    updated[-1] = _apply_copilot_cache_control_to_block(updated[-1], cache_control)
    return updated


def _apply_message_copilot_cache_control_to_last_anthropic_block(
    blocks: list[dict[str, Any]], message: dict[str, Any]
) -> list[dict[str, Any]]:
    updated = [_anthropic_block_from_internal_block(block) for block in blocks]
    cache_control = _copilot_cache_control(message)
    if cache_control is None or not updated:
        return updated
    updated[-1] = _apply_copilot_cache_control_to_anthropic_block(
        updated[-1], cache_control
    )
    return updated


def extract_function_calls_from_response(response: Any) -> list[dict[str, Any]]:
    plain = _to_plain_data(response) or {}
    output = plain.get("output") or []
    calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call"}:
            calls.append(
                {
                    "id": item.get("id"),
                    "call_id": item.get("call_id") or item.get("id"),
                    "type": item_type,
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                    },
                }
            )
    return calls


def chat_tools_to_responses_tools(
    chat_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in chat_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "type": "function",
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {},
            }
        )
    return tools


def chat_tools_to_anthropic_tools(
    chat_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in chat_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {},
            }
        )
    return tools


def anthropic_thinking_config(reasoning_effort: str | None) -> dict[str, Any] | None:
    if not reasoning_effort or reasoning_effort == "none":
        return None
    normalized = reasoning_effort.strip().lower()
    if normalized in {"minimal", "low"}:
        return {"type": "enabled", "budget_tokens": 2048}
    if normalized == "medium":
        return {"type": "enabled", "budget_tokens": 4096}
    if normalized == "high":
        return {"type": "enabled", "budget_tokens": 8192}
    return {"type": "adaptive"}
