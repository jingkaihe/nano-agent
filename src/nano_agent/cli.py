from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import click
import httpx

from .agent import run_chat
from .core import *
from .platforms import *

@click.group(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.option(
    "--platform",
    "platform_name",
    type=click.Choice(list(PLATFORM_NAMES), case_sensitive=False),
    default=DEFAULT_PLATFORM,
    show_default=True,
    help="LLM platform to use",
)
@click.option(
    "--provider",
    "provider_alias",
    type=click.Choice(list(PLATFORM_NAMES), case_sensitive=False),
    default=None,
    hidden=True,
    help="Compatibility alias for --platform",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to use")
@click.option("--api-key", help="API key for direct OpenAI-compatible endpoint mode")
@click.option(
    "--base-url", help="Custom base URL for direct OpenAI-compatible endpoint mode"
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(
        ["none", "minimal", "low", "medium", "high"], case_sensitive=False
    ),
    default=DEFAULT_REASONING_EFFORT,
    show_default=True,
    help="Reasoning effort for supported models",
)
@click.option(
    "--weak-model",
    default=None,
    help="Model used for helper tasks like web_fetch extraction",
)
@click.pass_context
def main(
    ctx: click.Context,
    platform_name: str,
    provider_alias: str | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
    reasoning_effort: str,
    weak_model: str | None,
) -> None:
    """Nano agent CLI."""
    resolved_platform = provider_alias or platform_name
    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "platform": resolved_platform,
            "provider": resolved_platform,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "weak_model": weak_model,
        }
    )

    if ctx.invoked_subcommand is not None:
        return

    if ctx.args:
        ctx.invoke(
            run,
            prompt_args=tuple(ctx.args),
        )
        return

    raise click.UsageError("Specify a subcommand such as 'run' or 'login'.")


@cast(Any, main).command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@click.option(
    "follow_latest",
    "-f",
    is_flag=True,
    help="Resume the latest saved conversation",
)
@click.option(
    "--resume", default=None, help="Resume a saved conversation by session UUID"
)
@click.option(
    "--provider",
    type=click.Choice(list(PLATFORM_NAMES), case_sensitive=False),
    default=None,
    help="LLM platform to use",
)
@click.option("--model", default=None, help="Model to use")
@click.option(
    "--api-key", default=None, help="API key for direct OpenAI-compatible endpoint mode"
)
@click.option(
    "--base-url",
    default=None,
    help="Custom base URL for direct OpenAI-compatible endpoint mode",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(
        ["none", "minimal", "low", "medium", "high"], case_sensitive=False
    ),
    default=None,
    help="Reasoning effort for supported models",
)
@click.option(
    "--weak-model",
    default=None,
    help="Model used for helper tasks like web_fetch extraction",
)
@click.option(
    "--allowed-tools",
    default=None,
    help="Comma-separated list of allowed tools for this run",
)
@click.option(
    "--debug-tools",
    is_flag=True,
    help="Print resolved tool allowlist and tool-call decisions",
)
@click.argument("prompt_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def run(
    ctx: click.Context,
    prompt_args: tuple[str, ...],
    follow_latest: bool,
    resume: str | None,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
    weak_model: str | None,
    allowed_tools: str | None,
    debug_tools: bool,
) -> None:
    """Run the nano agent chat loop or send a one-shot prompt."""
    prompt = " ".join(prompt_args).strip() or None
    config = ctx.find_root().obj or {}
    if follow_latest:
        if resume:
            raise click.UsageError("Cannot use -f together with --resume")
        resume = latest_conversation_session_id()
    resumed = False
    session_id: str | None = None
    restored_messages: list[dict[str, Any]] | None = None
    restored_responses_items: list[dict[str, Any]] | None = None
    restored_provider: str | None = None
    restored_model: str | None = None
    restored_reasoning_effort: str | None = None
    restored_weak_model: str | None = None
    restored_allowed_tools: list[str] | None = None
    restored_created_at: str | None = None
    restored_usage: dict[str, Any] | None = None

    if resume:
        loaded = load_conversation_history(resume)
        resumed = True
        session_id = cast(str, loaded["id"])
        restored_provider = cast(str, loaded["provider"])
        restored_model = cast(str, loaded["model"])
        restored_reasoning_effort = cast(str | None, loaded["reasoning_effort"])
        restored_weak_model = cast(str | None, loaded.get("weak_model"))
        restored_allowed_tools = cast(list[str] | None, loaded["allowed_tools"])
        restored_created_at = cast(str, loaded["created_at"])
        restored_messages = cast(list[dict[str, Any]], loaded["messages"])
        restored_responses_items = cast(list[dict[str, Any]], loaded["responses_items"])
        restored_usage = cast(dict[str, Any], loaded["usage"])

    resolved_provider = (
        provider
        if provider is not None
        else restored_provider or cast(str, config["provider"])
    )
    resolved_model = (
        model if model is not None else restored_model or cast(str, config["model"])
    )
    resolved_api_key = (
        api_key if api_key is not None else cast(str | None, config["api_key"])
    )
    resolved_base_url = (
        base_url if base_url is not None else cast(str | None, config["base_url"])
    )
    resolved_reasoning_effort = (
        reasoning_effort
        if reasoning_effort is not None
        else restored_reasoning_effort or cast(str, config["reasoning_effort"])
    )
    resolved_weak_model = (
        weak_model
        if weak_model is not None
        else restored_weak_model or cast(str | None, config.get("weak_model"))
    )
    resolved_allowed_tools = (
        normalize_allowed_tools(allowed_tools)
        if allowed_tools is not None
        else restored_allowed_tools
    )
    if resolved_model is None or resolved_provider is None:
        raise click.ClickException("model and provider must be resolved before running")

    try:
        asyncio.run(
            run_chat(
                model=resolved_model,
                prompt=prompt,
                provider=resolved_provider,
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                reasoning_effort=resolved_reasoning_effort,
                weak_model=resolved_weak_model,
                session_id=session_id,
                resumed=resumed,
                messages=restored_messages,
                responses_items=restored_responses_items,
                usage=restored_usage,
                created_at=restored_created_at,
                allowed_tools=resolved_allowed_tools,
                debug_tools=debug_tools,
            )
        )
    except CopilotAuthError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo()


@cast(Any, main).command(
    name="chat",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.option(
    "follow_latest",
    "-f",
    is_flag=True,
    help="Resume the latest saved conversation",
)
@click.option(
    "--resume", default=None, help="Resume a saved conversation by session UUID"
)
@click.option(
    "--provider",
    type=click.Choice(list(PLATFORM_NAMES), case_sensitive=False),
    default=None,
    help="LLM platform to use",
)
@click.option("--model", default=None, help="Model to use")
@click.option(
    "--api-key", default=None, help="API key for direct OpenAI-compatible endpoint mode"
)
@click.option(
    "--base-url",
    default=None,
    help="Custom base URL for direct OpenAI-compatible endpoint mode",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(
        ["none", "minimal", "low", "medium", "high"], case_sensitive=False
    ),
    default=None,
    help="Reasoning effort for supported models",
)
@click.option(
    "--weak-model",
    default=None,
    help="Model used for helper tasks like web_fetch extraction",
)
@click.option(
    "--allowed-tools",
    default=None,
    help="Comma-separated list of allowed tools for this run",
)
@click.option(
    "--debug-tools",
    is_flag=True,
    help="Print resolved tool allowlist and tool-call decisions",
)
@click.argument("prompt_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def chat(
    ctx: click.Context,
    prompt_args: tuple[str, ...],
    follow_latest: bool,
    resume: str | None,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
    weak_model: str | None,
    allowed_tools: str | None,
    debug_tools: bool,
) -> None:
    """Alias for run."""
    ctx.invoke(
        run,
        prompt_args=prompt_args,
        follow_latest=follow_latest,
        resume=resume,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        weak_model=weak_model,
        allowed_tools=allowed_tools,
        debug_tools=debug_tools,
    )


@cast(Any, main).command()
@click.option(
    "--output", type=click.Path(path_type=Path), help="Credentials output path"
)
def login(output: Path | None) -> None:
    """Authenticate with GitHub Copilot."""
    try:
        copilot_login(output)
    except KeyboardInterrupt:
        raise click.ClickException("Authentication cancelled")
    except Exception as exc:
        raise click.ClickException(f"Authentication failed: {exc}") from exc


@cast(Any, main).command(name="history")
def history_command() -> None:
    """List saved conversation history."""
    console = Console()
    sessions = list_conversation_histories()
    if not sessions:
        console.print("No saved history.")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("UUID", style="green", no_wrap=True)
    table.add_column("First prompt")
    table.add_column("Created at", style="dim", no_wrap=True)
    table.add_column("Updated at", style="dim", no_wrap=True)

    for session in sessions:
        table.add_row(
            cast(str, session["id"]),
            cast(str, session["first_prompt"]),
            cast(str, session["created_at"]),
            cast(str, session["updated_at"]),
        )

    console.print(table)


@cast(Any, main).command(name="models")
@click.option(
    "--provider",
    type=click.Choice(list(PLATFORM_NAMES), case_sensitive=False),
    default=None,
    help="LLM platform to query",
)
@click.option(
    "--api-key", default=None, help="API key for direct OpenAI-compatible endpoint mode"
)
@click.option(
    "--base-url",
    default=None,
    help="Custom base URL for direct OpenAI-compatible endpoint mode",
)
@click.pass_context
def models_command(
    ctx: click.Context,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """List available models."""
    config = ctx.find_root().obj or {}
    resolved_provider = provider or cast(str, config["provider"])
    resolved_api_key = (
        api_key if api_key is not None else cast(str | None, config["api_key"])
    )
    resolved_base_url = (
        base_url if base_url is not None else cast(str | None, config["base_url"])
    )

    try:
        models = list_provider_models(
            provider=resolved_provider,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            force_reload=True,
        )
        render_models_table(models, provider=resolved_provider)
    except CopilotAuthError as exc:
        raise click.ClickException(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise click.ClickException(f"Failed to list models: {exc}") from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

if __name__ == "__main__":
    cast(Any, main)()
