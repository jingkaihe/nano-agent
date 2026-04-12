from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from markdownify import markdownify as html_to_markdown
from pydantic import BaseModel, ConfigDict, Field

from ._common import ARCHIVE_DIR, JINJA
from .base import ToolCallContext, ToolDefinition


class WebFetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="The URL to fetch content from")
    prompt: str | None = Field(
        default=None,
        description="Information to extract from HTML/Markdown content (optional)",
    )


def web_fetch_schema() -> dict[str, Any]:
    return WebFetchArgs.model_json_schema()


def web_fetch_description() -> str:
    return JINJA.from_string(
        """Fetch content from a public URL.

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
"""
    ).render()


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


def with_numbered_lines(lines: list[str], start_line: int) -> str:
    if not lines:
        return ""
    width = max(4, len(str(start_line + len(lines) - 1)))
    return "\n".join(
        f"{start_line + index:>{width}} | {line}" for index, line in enumerate(lines)
    )


class WebFetchTool(ToolDefinition):
    name = "web_fetch"

    def description(self, agent: Any) -> str:
        return web_fetch_description()

    def schema(self, agent: Any) -> dict[str, Any]:
        return web_fetch_schema()

    async def execute(self, context: ToolCallContext) -> dict[str, Any]:
        raw_url = context.arguments.get("url")
        prompt = context.arguments.get("prompt", "")
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
            extracted = await context.agent.extract_from_markdown(prompt, markdown)
            return {
                "success": True,
                "url": raw_url,
                "prompt": prompt,
                "content": extracted,
            }
        return {"success": True, "url": raw_url, "content": markdown}


web_fetch_tool = WebFetchTool()

__all__ = [
    "WebFetchTool",
    "add_line_numbers",
    "archive_filename",
    "fetch_with_same_domain_redirects",
    "is_markdown_like",
    "validate_fetch_url",
    "web_fetch_tool",
    "web_fetch_description",
    "web_fetch_schema",
    "with_numbered_lines",
]
