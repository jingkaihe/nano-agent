from __future__ import annotations

from ._common import *


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
                    raise ValueError(
                        "redirects are only followed within the same domain"
                    )
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
                raise ValueError(
                    f"binary content type is not supported: {content_type}"
                )
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
