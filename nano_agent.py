#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1.8",
#   "openai>=1.68.2",
#   "httpx>=0.28.1",
#   "requests>=2.32.3",
#   "PyYAML>=6.0.2",
#   "markdownify>=0.13.1",
#   "Jinja2>=3.1.4",
#   "rich>=13.9.4",
# ]
# ///

from __future__ import annotations

import asyncio
import difflib
import json
import os
import platform
import re
import select
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import httpx
import requests
import yaml
import click
from jinja2 import Environment, StrictUndefined
from markdownify import markdownify as html_to_markdown
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule


COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_DEVICE_URL = "https://github.com/login/device/code"
COPILOT_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_SCOPES = ["read:user", "user:email", "copilot"]

DEFAULT_MODEL = os.getenv("NANO_AGENT_MODEL", "claude-sonnet-4.5")
DEFAULT_PROVIDER = os.getenv("NANO_AGENT_PROVIDER", "copilot")
DEFAULT_REASONING_EFFORT = os.getenv("NANO_AGENT_REASONING_EFFORT", "high")
CONTEXT_FILENAME = "AGENTS.md"
BANNED_COMMANDS = {"vim", "view", "less", "more", "cd"}
ARCHIVE_DIR = Path.home() / ".nano-agent" / "web-archives"
JINJA = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


class CopilotAuthError(Exception):
    pass


class ChatUI:
    def __init__(self) -> None:
        self.console = Console()
        self._current_stream: str | None = None
        self._stream_open = False

    def _separator(self, label: str, style: str) -> None:
        self.console.print()
        self.console.print(Rule(f"[{style}]{label}[/{style}]", style=style))

    def startup(self, provider: str, model: str, reasoning_effort: str | None) -> None:
        subtitle = f"{provider} / {model}"
        if reasoning_effort:
            subtitle += f" / reasoning={reasoning_effort}"
        self.console.print(Rule(f"[bold cyan]nano-agent[/bold cyan] [dim]{subtitle}[/dim]"))
        self.console.print("[dim]Type /exit to quit.[/dim]")

    def prompt(self) -> str:
        self._separator("user", "green")
        return Prompt.ask("[bold green]You[/bold green]").strip()

    def begin_assistant(self) -> None:
        self._current_stream = None
        self._stream_open = False

    def update_assistant(self, text_fragment: str = "", reasoning_fragment: str = "") -> None:
        if reasoning_fragment:
            if self._current_stream != "thinking":
                if self._stream_open:
                    self.console.print()
                self._separator("thinking", "magenta")
                self.console.print("thinking: ", style="magenta", end="")
                self._current_stream = "thinking"
                self._stream_open = True
            self.console.print(reasoning_fragment, style="dim italic", end="", soft_wrap=True)

        if text_fragment:
            if self._current_stream != "assistant":
                if self._stream_open:
                    self.console.print()
                self._separator("assistant", "blue")
                self.console.print("assistant: ", style="blue", end="")
                self._current_stream = "assistant"
                self._stream_open = True
            self.console.print(text_fragment, end="", soft_wrap=True)

    def end_assistant(self) -> None:
        if self._stream_open:
            self.console.print()
        self._current_stream = None
        self._stream_open = False

    def show_tool_result(self, name: str, result: dict[str, Any]) -> None:
        rendered = json.dumps(result, ensure_ascii=False)
        style = "green" if result.get("success") else "red"
        self._separator(f"tool:{name}", style)
        self.console.print(f"tool:{name}: ", style=style, end="")
        self.console.print(rendered, soft_wrap=True)

    def newline(self) -> None:
        self.console.print()


def credentials_path() -> Path:
    raw = os.getenv("NANO_AGENT_COPILOT_CREDS")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".nano-agent" / "copilot.json"


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


def load_copilot_credentials() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        raise CopilotAuthError(
            f"Copilot credentials not found at {path}. Run 'uv run python nano_agent.py login' first."
        )
    return json.loads(path.read_text())


def save_copilot_credentials(credentials: dict[str, Any], output_path: Path | None = None) -> Path:
    path = output_path or credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credentials, indent=2) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def refresh_copilot_token(creds: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    refresh_threshold = time.time() + 10 * 60
    expires_at = _coerce_expires_at(creds["copilot_expires_at"])
    if expires_at > refresh_threshold:
        return creds["copilot_token"], creds

    response = requests.get(
        COPILOT_EXCHANGE_URL,
        headers={
            "Authorization": f"Bearer {creds['access_token']}",
            "Editor-Version": "vscode/1.102.0",
            "Copilot-Integration-Id": "vscode-chat",
            "Accept": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    token_data = response.json()
    creds["copilot_token"] = token_data["token"]
    creds["copilot_expires_at"] = token_data["expires_at"]
    save_copilot_credentials(creds)
    return creds["copilot_token"], creds


def create_openai_copilot_client() -> AsyncOpenAI:
    base_url = (
        "https://api.business.githubcopilot.com"
        if os.getenv("BUSINESS_COPILOT") == "true"
        else "https://api.githubcopilot.com"
    )
    creds = load_copilot_credentials()
    copilot_token, _ = refresh_copilot_token(creds)
    return AsyncOpenAI(
        api_key=copilot_token,
        base_url=base_url,
        default_headers={
            "User-Agent": "GithubCopilot/1.342.0",
            "Editor-Version": "vscode/1.102.0",
        },
    )


def create_direct_openai_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def create_client(provider: str, api_key: str | None = None, base_url: str | None = None) -> AsyncOpenAI:
    normalized = provider.strip().lower()
    if normalized == "copilot":
        return create_openai_copilot_client()
    if normalized == "openai":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("NANO_AGENT_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass --api-key."
            )
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("NANO_AGENT_BASE_URL")
        return create_direct_openai_client(resolved_api_key, resolved_base_url)
    raise ValueError(f"unsupported provider: {provider}")


def generate_device_flow() -> dict[str, Any]:
    response = requests.post(
        COPILOT_DEVICE_URL,
        data={
            "client_id": COPILOT_CLIENT_ID,
            "scope": " ".join(COPILOT_SCOPES),
        },
        headers={
            "Accept": "application/json",
            "User-Agent": "nano-agent",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def poll_for_token(device_code: str, interval: int, expires_in: int) -> dict[str, Any]:
    timeout_at = time.time() + expires_in
    current_interval = interval
    while time.time() < timeout_at:
        time.sleep(current_interval)
        response = requests.post(
            COPILOT_TOKEN_URL,
            data={
                "client_id": COPILOT_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        error = data.get("error")
        if not error:
            return data
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            current_interval += 5
            continue
        raise CopilotAuthError(
            f"Authentication failed: {error} - {data.get('error_description', '')}"
        )
    raise CopilotAuthError("Authentication timed out")


def exchange_for_copilot_token(access_token: str) -> dict[str, Any]:
    response = requests.get(
        COPILOT_EXCHANGE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Editor-Version": "vscode/1.102.0",
            "Copilot-Integration-Id": "vscode-chat",
            "Accept": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def copilot_login(output_path: Path | None = None) -> dict[str, Any]:
    print("=" * 60)
    print("GitHub Copilot OAuth Login")
    print("=" * 60)
    device_response = generate_device_flow()
    print()
    print("To authenticate with GitHub Copilot:")
    print(f"  1. Open this URL in your browser: {device_response['verification_uri']}")
    print(f"  2. Enter this code when prompted: {device_response['user_code']}")
    print()
    print("Waiting for authentication to complete...")

    token_response = poll_for_token(
        device_response["device_code"],
        int(device_response["interval"]),
        int(device_response["expires_in"]),
    )
    access_token = token_response["access_token"]
    copilot_response = exchange_for_copilot_token(access_token)
    credentials = {
        "access_token": access_token,
        "copilot_token": copilot_response["token"],
        "scope": token_response.get("scope", ""),
        "copilot_expires_at": copilot_response["expires_at"],
    }
    saved = save_copilot_credentials(credentials, output_path)
    print()
    print(f"✓ Authentication successful. Credentials saved to {saved}")
    return credentials


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


def build_system_prompt(cwd: Path) -> str:
    contexts = collect_contexts(cwd)
    template = JINJA.from_string(
        '''You are an interactive CLI tool that helps with software engineering and production operations tasks. Please follows the instructions and tools below to help the user.

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

# Context
If the current working directory contains a `{{ context_filename }}` file, it will be automatically loaded as a context. Use it for:
* Understanding the structure, organisation and tech stack of the project.
* Keeping record of commands (for linting, testing, building etc) that you have to use repeatedly.
* Recording coding style, conventions and preferences of the project.

If you find a new command that you have to use repeatedly, you can add it to the `{{ context_filename }}` file.
If you have made any significant changes to the project structure, or modified the tech stack, you should update the `{{ context_filename }}` file.

# Filesystem Search Tools
For filesystem search activities, use `fd` for file discovery and `rg` for content search via the `bash` tool only.

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
'''
    )
    return template.render(
        context_filename=CONTEXT_FILENAME,
        cwd=cwd,
        is_git_repo=str(is_git_repo(cwd)).lower(),
        os_name=platform.system(),
        os_version=platform.release(),
        date=time.strftime('%Y-%m-%d'),
        contexts=contexts,
    ).strip() + "\n"


@dataclass
class SkillInfo:
    name: str
    description: str
    directory: Path
    content: str


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
    return SkillInfo(name=name, description=description, directory=path.parent, content=content)


def _discover_skills_from_dir(base: Path, prefix: str, found: dict[str, SkillInfo]) -> None:
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
        '''When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

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
- If you need to modify a script or template from the skill directory, copy it to the working directory first and update it using apply_patch
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
'''
    )
    return template.render(skills=[skills[name] for name in sorted(skills)]).rstrip()


def bash_description() -> str:
    return JINJA.from_string(
        '''Run a bash command in a persistent shell session.

# Restrictions
Banned commands:
- vim
- view
- less
- more
- cd

# Input
- command: required single-line bash command
- description: required, 5-10 words
- timeout: required, 10-120

# Rules
- Use parallel tool calling for independent commands.
- Do not run interactive commands.
- For multiple commands, use ';' or '&&' on one line.
- Avoid direct cd; use absolute paths or subshell: (cd /path && cmd).
- For filesystem search activities, use fd and rg via this tool only.
- Do not use heredoc; use file_write or apply_patch instead.

Examples:
- (cd /repo && mise run test)
'''
    ).render()


def apply_patch_description() -> str:
    return JINJA.from_string(
        '''Use the `apply_patch` tool to edit files.
Your patch language is a stripped-down, file-oriented diff format designed to be easy to parse and safe to apply. You can think of it as a high-level envelope:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Within that envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more "hunks", each introduced by @@ (optionally followed by a hunk header).
Within a hunk each line starts with:

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single `@@` statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@   def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

The full grammar definition is below:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

A full patch can combine several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

It is important to remember:

- You must include a header with your intended action (Add/Delete/Update)
- You must prefix new lines with `+` even when creating a new file
- File references can only be relative, NEVER ABSOLUTE.
'''
    ).render()


def web_fetch_description() -> str:
    return JINJA.from_string(
        '''Fetch content from a public URL.

# Input
- url: required URL to fetch
- prompt: optional instruction for extracting info from HTML/Markdown pages

# Rules
- Use HTTPS for external domains.
- HTTP is allowed only for localhost/internal addresses.
- Redirects are followed only within the same domain (max 10).
- Binary content types (zip/pdf/image/audio/video/octet-stream) are rejected.

# Behavior
- Code/text/JSON/XML/etc:
  - Save to ~/.nano-agent/web-archives/{domain}/{filename}.{ext}
  - Return content with line numbers (truncated if output is too large)
- HTML/Markdown without prompt:
  - Return full page content as Markdown (HTML is converted)
- HTML/Markdown with prompt:
  - Run AI extraction against page content and return only the extracted result

# Prompt guidance
Use prompt when:
- You need specific facts/sections from an HTML/Markdown page
- The page is large and you do not want full-page output

Examples:
- url: https://docs.example.com/api-reference
  prompt: List all endpoints with HTTP methods
- url: https://company.example.com/changelog
  prompt: Summarize breaking changes in the latest release

Do not use prompt when:
- You want raw file contents with line numbers
- You want full-page output

Examples:
- url: https://raw.githubusercontent.com/user/repo/main/config.yaml
- url: https://example.com/data.json

# Notes
- Only public URLs are supported (no auth/session handling).
- Prompt is ignored for non-HTML/Markdown responses (code/text/JSON/XML).
'''
    ).render()


def bash_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run"},
            "description": {"type": "string", "description": "A description of the command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (10-120)"},
        },
        "required": ["command", "description", "timeout"],
        "additionalProperties": False,
    }


def apply_patch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The entire contents of the apply_patch command",
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }


def skill_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The name of the skill to invoke",
            }
        },
        "required": ["skill_name"],
        "additionalProperties": False,
    }


def web_fetch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from"},
            "prompt": {
                "type": "string",
                "description": "Information to extract from HTML/Markdown content (optional)",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }


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


def _merge_stream_tool_call(accumulator: dict[int, dict[str, Any]], tool_call: Any) -> None:
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


def chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message.get("content") or ""}],
                }
            )
        elif role == "assistant":
            content = message.get("content")
            if content:
                items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": message.get("content") or "",
                }
            )
    return items


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


def chat_tools_to_responses_tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


class PersistentBashSession:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def run(self, command: str, timeout: int) -> tuple[int, str]:
        self._ensure_started()
        assert self.proc and self.proc.stdin and self.proc.stdout
        token = uuid.uuid4().hex
        exit_marker = f"__NANO_AGENT_EXIT_{token}__"
        done_marker = f"__NANO_AGENT_DONE_{token}__"
        payload = (
            f"{command}\n"
            f"printf '\n{exit_marker}:%s\n' \"$?\"\n"
            f"printf '{done_marker}\n'\n"
        ).encode()
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        stdout_fd = self.proc.stdout.fileno()
        deadline = time.time() + timeout
        chunks = bytearray()

        while time.time() < deadline:
            ready, _, _ = select.select([stdout_fd], [], [], 0.1)
            if not ready:
                continue
            chunk = os.read(stdout_fd, 65536)
            if not chunk:
                break
            chunks.extend(chunk)
            text = chunks.decode(errors="replace")
            if done_marker in text:
                before_done = text.split(done_marker, 1)[0]
                match = re.search(rf"(?s)(.*)\n{re.escape(exit_marker)}:(-?\d+)\n$", before_done)
                if not match:
                    return 1, before_done.rstrip()
                output = match.group(1).rstrip()
                exit_code = int(match.group(2))
                return exit_code, output

        self.close()
        self.proc = None
        raise TimeoutError(f"command timed out after {timeout} seconds")


def validate_bash_args(command: Any, description: Any, timeout: Any) -> tuple[str, str, int]:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command is required")
    if "\n" in command or "\r" in command:
        raise ValueError("command must be single-line")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    if not isinstance(timeout, int) or timeout < 10 or timeout > 120:
        raise ValueError("timeout must be between 10 and 120 seconds")

    for part in re.split(r"&&|\|\||;", command):
        part = part.strip()
        if not part:
            continue
        try:
            tokens = shlex.split(part)
        except ValueError as exc:
            raise ValueError(f"invalid shell syntax: {exc}") from exc
        if not tokens:
            continue
        if tokens[0] in BANNED_COMMANDS:
            raise ValueError(f"command is banned: {tokens[0]}")

    return command, description, timeout


@dataclass
class Hunk:
    header: str
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FilePatch:
    op: str
    path: str
    move_to: str | None = None
    add_lines: list[str] = field(default_factory=list)
    hunks: list[Hunk] = field(default_factory=list)


def _require_relative_path(path: str) -> str:
    if not path or Path(path).is_absolute():
        raise ValueError("File references must be relative, NEVER ABSOLUTE")
    return path


def parse_apply_patch(patch_text: str) -> list[FilePatch]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("Patch must start with '*** Begin Patch' and end with '*** End Patch'")

    ops: list[FilePatch] = []
    i = 1
    while i < len(lines) - 1:
        line = lines[i]
        if line.startswith("*** Add File: "):
            path = _require_relative_path(line.removeprefix("*** Add File: ").strip())
            i += 1
            add_lines: list[str] = []
            while i < len(lines) - 1 and not lines[i].startswith("*** "):
                current = lines[i]
                if not current.startswith("+"):
                    raise ValueError(f"Add File lines must start with '+': {current}")
                add_lines.append(current[1:])
                i += 1
            ops.append(FilePatch(op="add", path=path, add_lines=add_lines))
            continue
        if line.startswith("*** Delete File: "):
            path = _require_relative_path(line.removeprefix("*** Delete File: ").strip())
            ops.append(FilePatch(op="delete", path=path))
            i += 1
            continue
        if line.startswith("*** Update File: "):
            path = _require_relative_path(line.removeprefix("*** Update File: ").strip())
            patch = FilePatch(op="update", path=path)
            i += 1
            if i < len(lines) - 1 and lines[i].startswith("*** Move to: "):
                patch.move_to = _require_relative_path(lines[i].removeprefix("*** Move to: ").strip())
                i += 1
            while i < len(lines) - 1 and not lines[i].startswith("*** "):
                if not lines[i].startswith("@@"):
                    raise ValueError(f"Expected hunk header, got: {lines[i]}")
                hunk = Hunk(header=lines[i][2:].strip())
                i += 1
                while i < len(lines) - 1 and not lines[i].startswith("@@") and not lines[i].startswith("*** "):
                    if lines[i] == "*** End of File":
                        i += 1
                        continue
                    prefix = lines[i][:1]
                    if prefix not in {" ", "+", "-"}:
                        raise ValueError(f"Invalid hunk line: {lines[i]}")
                    hunk.lines.append((prefix, lines[i][1:]))
                    i += 1
                patch.hunks.append(hunk)
            ops.append(patch)
            continue
        if not line.strip():
            i += 1
            continue
        raise ValueError(f"Unknown patch operation: {line}")

    return ops


def _find_hunk_position(source_lines: list[str], needle: list[str], start: int) -> int | None:
    if not needle:
        return start
    max_start = len(source_lines) - len(needle)
    for candidate in range(start, max_start + 1):
        if source_lines[candidate : candidate + len(needle)] == needle:
            return candidate
    for candidate in range(0, max_start + 1):
        if source_lines[candidate : candidate + len(needle)] == needle:
            return candidate
    stripped_needle = [line.rstrip() for line in needle]
    for candidate in range(start, max_start + 1):
        if [line.rstrip() for line in source_lines[candidate : candidate + len(needle)]] == stripped_needle:
            return candidate
    return None


def apply_hunks_to_text(original: str, hunks: list[Hunk], path: str) -> str:
    newline = "\r\n" if "\r\n" in original else "\n"
    source_lines = original.splitlines()
    output: list[str] = []
    cursor = 0
    for hunk in hunks:
        source_fragment = [text for prefix, text in hunk.lines if prefix in {" ", "-"}]
        target_fragment = [text for prefix, text in hunk.lines if prefix in {" ", "+"}]
        pos = _find_hunk_position(source_lines, source_fragment, cursor)
        if pos is None:
            raise ValueError(f"Failed to apply hunk to {path}: context not found")
        output.extend(source_lines[cursor:pos])
        output.extend(target_fragment)
        cursor = pos + len(source_fragment)
    output.extend(source_lines[cursor:])
    result = newline.join(output)
    if original.endswith(("\n", "\r\n")) and output:
        result += newline
    return result


def _unified_diff(old: str, new: str, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff)


def execute_apply_patch(patch_text: str, cwd: Path) -> dict[str, Any]:
    ops = parse_apply_patch(patch_text)
    changes: list[dict[str, Any]] = []
    diffs: list[str] = []

    for op in ops:
        target = cwd / op.path
        if op.op == "add":
            target.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(op.add_lines)
            if op.add_lines:
                content += "\n"
            target.write_text(content)
            changes.append({"action": "add", "path": op.path})
            diffs.append(_unified_diff("", content, f"a/{op.path}", f"b/{op.path}"))
            continue

        if op.op == "delete":
            if not target.exists():
                raise ValueError(f"Cannot delete missing file: {op.path}")
            old = target.read_text(errors="replace")
            target.unlink()
            changes.append({"action": "delete", "path": op.path})
            diffs.append(_unified_diff(old, "", f"a/{op.path}", f"b/{op.path}"))
            continue

        if op.op == "update":
            if not target.exists():
                raise ValueError(f"Cannot update missing file: {op.path}")
            old = target.read_text(errors="replace")
            new = apply_hunks_to_text(old, op.hunks, op.path)
            write_target = cwd / (op.move_to or op.path)
            write_target.parent.mkdir(parents=True, exist_ok=True)
            write_target.write_text(new)
            if op.move_to and write_target.resolve() != target.resolve():
                target.unlink()
            changes.append({
                "action": "update",
                "path": op.path,
                "move_to": op.move_to,
            })
            diffs.append(
                _unified_diff(
                    old,
                    new,
                    f"a/{op.path}",
                    f"b/{op.move_to or op.path}",
                )
            )
            continue

    return {
        "success": True,
        "summary": f"Applied patch to {len(changes)} file(s)",
        "changes": changes,
        "diff": "\n".join(diff for diff in diffs if diff).strip(),
    }


def is_local_host(hostname: str) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def validate_fetch_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("invalid URL")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and is_local_host(parsed.hostname or ""):
        return
    raise ValueError(
        "only HTTPS scheme is supported for external domains, HTTP is allowed for localhost/internal addresses"
    )


async def fetch_with_same_domain_redirects(raw_url: str) -> tuple[str, str]:
    validate_fetch_url(raw_url)
    async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
        current = raw_url
        original_host = (urlparse(raw_url).hostname or "").lower()
        for _ in range(10):
            response = await client.get(current, headers={"User-Agent": "nano-agent"})
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("redirect response missing location header")
                current = urljoin(current, location)
                new_host = (urlparse(current).hostname or "").lower()
                if new_host != original_host:
                    raise ValueError("redirects are only followed within the same domain")
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "text/plain").lower()
            if (
                "application/zip" in content_type
                or "application/pdf" in content_type
                or content_type.startswith("image/")
                or content_type.startswith("audio/")
                or content_type.startswith("video/")
                or "application/octet-stream" in content_type
            ):
                raise ValueError(f"binary content type is not supported: {content_type}")
            return response.text, content_type
    raise ValueError("too many redirects")


def is_markdown_like(raw_url: str, content_type: str) -> bool:
    path = urlparse(raw_url).path.lower()
    return (
        "text/html" in content_type
        or "text/markdown" in content_type
        or path.endswith(".md")
        or path.endswith(".markdown")
    )


def archive_filename(raw_url: str, content_type: str) -> Path:
    parsed = urlparse(raw_url)
    domain = parsed.hostname or "unknown"
    name = Path(parsed.path).name or "index"
    if "." not in name:
        if "json" in content_type:
            name += ".json"
        elif "xml" in content_type:
            name += ".xml"
        elif "javascript" in content_type:
            name += ".js"
        elif "html" in content_type:
            name += ".html"
        else:
            name += ".txt"
    return ARCHIVE_DIR / domain / name


def add_line_numbers(text: str, max_lines: int = 400, max_chars: int = 30000) -> str:
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    numbered = "\n".join(f"{idx + 1:>4} | {line}" for idx, line in enumerate(lines))
    if len(numbered) > max_chars:
        numbered = numbered[:max_chars]
        truncated = True
    if truncated:
        numbered += "\n... [truncated]"
    return numbered


class NanoAgent:
    def __init__(
        self,
        model: str,
        cwd: Path,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.cwd = cwd.resolve()
        self.provider = provider
        self.reasoning_effort = None if reasoning_effort in {None, "none"} else reasoning_effort
        self.client = create_client(provider=provider, api_key=api_key, base_url=base_url)
        self.messages: list[dict[str, Any]] = []
        self.bash = PersistentBashSession()
        self.active_skills: set[str] = set()
        self.ui = ChatUI()

    def completion_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def responses_reasoning(self) -> dict[str, Any] | None:
        if not self.reasoning_effort:
            return None
        return {"effort": self.reasoning_effort, "summary": "auto"}

    def close(self) -> None:
        self.bash.close()

    def current_skills(self) -> dict[str, SkillInfo]:
        return discover_skills(self.cwd)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": bash_description(),
                    "parameters": bash_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": apply_patch_description(),
                    "parameters": apply_patch_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "skill",
                    "description": skill_description(self.current_skills()),
                    "parameters": skill_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": web_fetch_description(),
                    "parameters": web_fetch_schema(),
                },
            },
        ]

    async def extract_from_markdown(self, prompt: str, markdown: str) -> str:
        limited = markdown[:50000]
        if should_use_responses_api(self.model):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": "You extract only the information requested from supplied web content. If the answer is not present, say so plainly.",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Request: {prompt}\n\nWeb content:\n\n{limited}",
                            }
                        ],
                    }
                ],
            }
            reasoning = self.responses_reasoning()
            if reasoning:
                kwargs["reasoning"] = reasoning
            response = await self.client.responses.create(**kwargs)
            plain = _to_plain_data(response) or {}
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                return output_text
            fragments: list[str] = []
            for item in plain.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for content_item in item.get("content") or []:
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        fragments.append(content_item["text"])
            return "".join(fragments)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract only the information requested from supplied web content. If the answer is not present, say so plainly.",
                },
                {
                    "role": "user",
                    "content": f"Request: {prompt}\n\nWeb content:\n\n{limited}",
                },
            ],
            **self.completion_kwargs(),
        )
        return response.choices[0].message.content or ""

    async def send_via_responses(self) -> None:
        previous_response_id: str | None = None
        pending_input = chat_messages_to_responses_input(self.messages)

        while True:
            self.ui.begin_assistant()
            chat_tools = self.tool_specs()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": pending_input,
                "instructions": build_system_prompt(self.cwd),
                "tools": chat_tools_to_responses_tools(chat_tools),
                "tool_choice": "auto",
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            reasoning = self.responses_reasoning()
            if reasoning:
                kwargs["reasoning"] = reasoning

            async with self.client.responses.stream(**kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = cast(str, getattr(event, "delta", ""))
                        if delta:
                            self.ui.update_assistant(text_fragment=delta)
                    elif event_type in {
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = cast(str, getattr(event, "delta", ""))
                        if delta:
                            self.ui.update_assistant(reasoning_fragment=delta)

                final_response = await stream.get_final_response()

            self.ui.end_assistant()

            previous_response_id = getattr(final_response, "id", None)
            output_text = getattr(final_response, "output_text", None)
            assistant_text = output_text if isinstance(output_text, str) and output_text else None
            self.messages.append({"role": "assistant", "content": assistant_text})

            function_calls = extract_function_calls_from_response(final_response)
            if not function_calls:
                return

            tool_outputs: list[dict[str, Any]] = []
            for call in function_calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    result = {"success": False, "error": f"invalid JSON arguments: {exc}"}
                else:
                    result = await self.run_tool(call["function"]["name"], args)
                self.ui.show_tool_result(call["function"]["name"], result)
                tool_content = json.dumps(result, ensure_ascii=False)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["call_id"],
                        "content": tool_content,
                    }
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": tool_content,
                    }
                )

            pending_input = tool_outputs

    async def send_via_chat_completions(self) -> None:
        while True:
            assistant_content_parts: list[str] = []
            tool_call_parts: dict[int, dict[str, Any]] = {}
            self.ui.begin_assistant()
            chat_messages = cast(
                list[ChatCompletionMessageParam],
                [{"role": "system", "content": build_system_prompt(self.cwd)}, *self.messages],
            )
            chat_tools = cast(list[ChatCompletionToolParam], self.tool_specs())

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                tools=chat_tools,
                tool_choice="auto",
                stream=True,
                **self.completion_kwargs(),
            )

            async for chunk in stream:
                choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
                if choice is None:
                    continue

                delta = _to_plain_data(choice.delta) or {}

                for fragment in _get_delta_fragments(delta, ["reasoning", "thinking"]):
                    if fragment:
                        self.ui.update_assistant(reasoning_fragment=fragment)

                content = delta.get("content")
                if isinstance(content, str):
                    self.ui.update_assistant(text_fragment=content)
                    assistant_content_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        item_plain = _to_plain_data(item)
                        if isinstance(item_plain, dict):
                            for fragment in _extract_text_fragments(item_plain):
                                self.ui.update_assistant(text_fragment=fragment)
                                assistant_content_parts.append(fragment)

                for streamed_tool_call in delta.get("tool_calls") or []:
                    _merge_stream_tool_call(tool_call_parts, streamed_tool_call)

            self.ui.end_assistant()

            message_tool_calls = _ordered_tool_calls(tool_call_parts)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(assistant_content_parts) or None,
            }
            if message_tool_calls:
                assistant_message["tool_calls"] = message_tool_calls

            self.messages.append(assistant_message)

            if not message_tool_calls:
                return

            for call in message_tool_calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    result = {"success": False, "error": f"invalid JSON arguments: {exc}"}
                else:
                    result = await self.run_tool(call["function"]["name"], args)
                self.ui.show_tool_result(call["function"]["name"], result)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

    async def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "bash":
                command, _description, timeout = validate_bash_args(
                    arguments.get("command"),
                    arguments.get("description"),
                    arguments.get("timeout"),
                )
                exit_code, output = self.bash.run(command, timeout)
                return {
                    "success": exit_code == 0,
                    "command": command,
                    "exit_code": exit_code,
                    "output": output,
                }

            if name == "apply_patch":
                patch_input = arguments.get("input")
                if not isinstance(patch_input, str) or not patch_input.strip():
                    raise ValueError("input is required")
                return execute_apply_patch(patch_input, self.cwd)

            if name == "skill":
                skill_name = arguments.get("skill_name")
                if not isinstance(skill_name, str) or not skill_name.strip():
                    raise ValueError("skill_name is required")
                skills = self.current_skills()
                if skill_name not in skills:
                    available = ", ".join(sorted(skills)) or "none"
                    raise ValueError(f"unknown skill '{skill_name}'. Available skills: {available}")
                if skill_name in self.active_skills:
                    raise ValueError(f"skill '{skill_name}' is already active")
                self.active_skills.add(skill_name)
                skill = skills[skill_name]
                content = f'''# Skill: {skill.name}

The skill directory is located at: {skill.directory}

{skill.content}'''
                return {
                    "success": True,
                    "skill_name": skill.name,
                    "directory": str(skill.directory),
                    "content": content,
                }

            if name == "web_fetch":
                raw_url = arguments.get("url")
                prompt = arguments.get("prompt", "")
                if not isinstance(raw_url, str) or not raw_url.strip():
                    raise ValueError("url is required")
                if prompt is None:
                    prompt = ""
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")

                content, content_type = await fetch_with_same_domain_redirects(raw_url)
                if not is_markdown_like(raw_url, content_type):
                    archive_path = archive_filename(raw_url, content_type)
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    archive_path.write_text(content)
                    return {
                        "success": True,
                        "url": raw_url,
                        "file_path": str(archive_path),
                        "content": add_line_numbers(content),
                    }

                markdown = (
                    html_to_markdown(content, heading_style="ATX")
                    if "text/html" in content_type
                    else content
                )
                if prompt.strip():
                    extracted = await self.extract_from_markdown(prompt, markdown)
                    return {
                        "success": True,
                        "url": raw_url,
                        "prompt": prompt,
                        "content": extracted,
                    }
                return {"success": True, "url": raw_url, "content": markdown}

            raise ValueError(f"unknown tool: {name}")
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def send(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        if should_use_responses_api(self.model):
            await self.send_via_responses()
        else:
            await self.send_via_chat_completions()


async def run_chat(
    model: str,
    prompt: str | None,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    agent = NanoAgent(
        model=model,
        cwd=Path.cwd(),
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )
    try:
        if prompt is not None:
            await agent.send(prompt)
            return

        agent.ui.startup(provider, model, reasoning_effort)
        while True:
            try:
                user_input = agent.ui.prompt()
            except EOFError:
                agent.ui.newline()
                break
            if not user_input:
                continue
            if user_input in {"/exit", "exit", "quit"}:
                break
            await agent.send(user_input)
    finally:
        agent.close()


@click.group(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.option(
    "--provider",
    type=click.Choice(["copilot", "openai"], case_sensitive=False),
    default=DEFAULT_PROVIDER,
    show_default=True,
    help="LLM provider to use",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to use")
@click.option("--api-key", help="API key for direct OpenAI-compatible endpoint mode")
@click.option("--base-url", help="Custom base URL for direct OpenAI-compatible endpoint mode")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high"], case_sensitive=False),
    default=DEFAULT_REASONING_EFFORT,
    show_default=True,
    help="Reasoning effort for supported models",
)
@click.pass_context
def main(
    ctx: click.Context,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    reasoning_effort: str,
) -> None:
    """Nano agent using chat completions via Copilot or OpenAI."""
    if ctx.invoked_subcommand is not None:
        return

    prompt = " ".join(ctx.args).strip() or None

    try:
        asyncio.run(
            run_chat(
                model=model,
                prompt=prompt,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                reasoning_effort=reasoning_effort,
            )
        )
    except CopilotAuthError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo()


@cast(Any, main).command()
@click.option("--output", type=click.Path(path_type=Path), help="Credentials output path")
def login(output: Path | None) -> None:
    """Authenticate with GitHub Copilot."""
    try:
        copilot_login(output)
    except KeyboardInterrupt:
        raise click.ClickException("Authentication cancelled")
    except Exception as exc:
        raise click.ClickException(f"Authentication failed: {exc}") from exc


if __name__ == "__main__":
    cast(Any, main)()
