from __future__ import annotations

from ._web_fetch_impl import (
    add_line_numbers,
    archive_filename,
    fetch_with_same_domain_redirects,
    is_markdown_like,
    validate_fetch_url,
)
from .descriptions import web_fetch_description
from .schemas import web_fetch_schema

__all__ = [
    "add_line_numbers",
    "archive_filename",
    "fetch_with_same_domain_redirects",
    "is_markdown_like",
    "validate_fetch_url",
    "web_fetch_description",
    "web_fetch_schema",
]
