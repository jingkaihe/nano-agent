# Repository Guidelines

## Project Structure & Module Organization
`nano-agent` is a packaged Python CLI under `src/nano_agent/`. Keep related runtime logic in focused modules (`core`, `platforms`, `tools`, `agent`, `cli`, `ui`) and prefer package directories when an area grows large. `src/nano_agent/platforms/` holds platform-specific code by provider, and `src/nano_agent/tools/` holds tool-specific code by tool. Put regression coverage under `tests/`. Local skill experiments belong under ignored `.agents/`, and generated artifacts such as `__pycache__/`, `.ruff_cache/`, and `.venv/` should stay uncommitted.

## Build, Test, and Development Commands
Use the packaged project through `uv` from the repo root:

- `uv run --project . nano-agent --help`: invoke the installed console entry point from the local checkout.
- `uv run --project . nano-agent run "summarize this repo"`: run a one-shot prompt; omit the prompt for chat mode.
- `uv run --project . nano-agent history`: inspect saved conversation sessions.
- `uv run pytest tests`: run the regression suite.
- `uvx ruff check src tests`: run linting without requiring a global `ruff` install.
- `uvx ruff format src tests`: apply consistent formatting with Ruff.

## Coding Style & Naming Conventions
Target Python 3.11+ and manage dependencies through `pyproject.toml`. Follow the existing style: 4-space indentation, `snake_case` for functions, variables, and helpers, `PascalCase` for classes/dataclasses, and explicit type hints on new code. Keep Click entrypoints thin; push parsing, validation, and API logic into focused helper functions.

## Testing Guidelines
Regression coverage lives under `tests/`. Keep tests deterministic and offline when possible, especially around provider/network behavior. Before submitting changes, run `uv run pytest tests`, `uvx ruff check src tests`, `uvx ruff format src tests`, and any targeted CLI smoke checks for the command path you touched.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commits, for example `feat(agent): ...`, `fix(cli): ...`, and `refactor: ...`. Keep commit subjects imperative and scoped when helpful. PRs should include a short summary, note user-visible CLI changes, list verification steps, and attach terminal output snippets or screenshots when changing the interactive UI.

## Security & Configuration Tips
Do not commit API keys, Copilot credentials, or conversation history; runtime data belongs under `~/.nano-agent/`. Prefer environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NANO_AGENT_API_KEY`, `OPENAI_BASE_URL`, and `ANTHROPIC_BASE_URL` over hardcoded values. Keep this repository-level `AGENTS.md` current because the CLI loads it as working context automatically.
