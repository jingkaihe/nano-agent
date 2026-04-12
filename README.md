# nano-agent

`nano-agent` is a small packaged coding agent CLI.

## Install

Install directly from Git with `uv tool`:

```bash
uv tool install "git+https://github.com/jingkaihe/nano-agent"
```

Install a different branch:

```bash
uv tool install --force "git+https://github.com/jingkaihe/nano-agent@my-branch"
```

For local development from a checkout:

```bash
uv run --project . nano-agent --help
uv run pytest tests
```

`nano-agent` is packaged with [`uv`](https://docs.astral.sh/uv/), so `uv tool install` works against the Git repository directly.

## Platforms And APIs

`nano-agent` now separates platform choice from API shape:

- Platforms: `copilot`, `openai`, `anthropic`
- API modes: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages

The runtime selects the appropriate API mode from the chosen platform and model.

## Enterprise usage

If you are running `nano-agent` in an enterprise environment, export the required `uv` and company-specific environment variables before installing or running the CLI.

```bash
export UV_NATIVE_TLS=true
export BUSINESS_COPILOT=true
export UV_DEFAULT_INDEX=http://example.com/pypi/simple
```

This is especially useful when your company requires native TLS trust settings and an internal Python package index.
