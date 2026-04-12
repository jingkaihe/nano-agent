from __future__ import annotations

import base64
import difflib
import json
import math
import mimetypes
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

from ..core import (
    ALL_TOOL_NAMES,
    ARCHIVE_DIR,
    BANNED_COMMANDS,
    BASH_APPROX_BYTES_PER_TOKEN,
    BASH_MAX_OUTPUT_TOKENS,
    DEFAULT_CHAT_TOOL_NAMES,
    DEFAULT_RESPONSE_TOOL_NAMES,
    GLOB_MAX_RESULTS,
    GREP_MAX_LINE_LENGTH,
    GREP_MAX_OUTPUT_BYTES,
    GREP_MAX_RESULTS,
    JINJA,
    MAX_READ_LINE_CHARACTERS,
    MAX_READ_LINE_LIMIT,
    MAX_READ_OUTPUT_BYTES,
    MAX_TOOL_OUTPUT_BYTES,
    SEARCH_TOOL_TIMEOUT_SECONDS,
    VIEW_IMAGE_MAX_HEIGHT,
    VIEW_IMAGE_MAX_WIDTH,
    should_use_anthropic_messages_api,
    should_use_responses_api,
)
from ..platforms import _model_catalog_entry
