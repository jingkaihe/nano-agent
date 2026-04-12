from __future__ import annotations

import inspect
import os
import time
import uuid

import anthropic
import httpx
import nano_agent.core as core
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

from ..core import (
    COPILOT_CHAT_PLUGIN_VERSION,
    COPILOT_CHAT_USER_AGENT,
    COPILOT_CLIENT_ID,
    COPILOT_DEVICE_URL,
    COPILOT_EDITOR_VERSION,
    COPILOT_EXCHANGE_URL,
    COPILOT_GITHUB_API_VERSION,
    COPILOT_INTEGRATION_ID,
    COPILOT_OPENAI_USER_AGENT,
    COPILOT_SCOPES,
    COPILOT_TOKEN_URL,
    COPILOT_VSCODE_USER_AGENT_LIBRARY_VERSION,
    CopilotAuthError,
    _legacy_context_window_limit,
    load_cached_provider_models,
    load_copilot_credentials,
    refresh_copilot_token,
    save_cached_provider_models,
    save_copilot_credentials,
    should_use_anthropic_messages_api,
)


DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
