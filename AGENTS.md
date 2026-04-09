# Repository Guidelines

## Project Structure & Module Organization
`nano-agent` is the entire application: a single executable Python CLI script with inline `uv` dependency metadata, Click commands, provider integrations, tool execution, history handling, and built-in self-tests. Keep new logic close to related helpers instead of splitting files prematurely. Local skill experiments belong under ignored `.agents/`, and generated artifacts such as `__pycache__/` and `.ruff_cache/` should stay uncommitted.

## Build, Test, and Development Commands
Use the executable directly from the repo root:

- `./nano-agent --help`: list CLI commands and global options.
- `./nano-agent run "summarize this repo"`: run a one-shot prompt; omit the prompt for chat mode.
- `./nano-agent history`: inspect saved conversation sessions.
- `./nano-agent self-test`: run the built-in regression checks.
- `uvx ruff check nano-agent`: run linting without requiring a global `ruff` install.
- `uvx ruff format nano-agent`: apply consistent formatting with Ruff.
- `uvx --with click --with openai --with anthropic --with httpx --with Pillow --with prompt_toolkit --with PyYAML --with markdownify --with Jinja2 --with rich ty check nano-agent`: run static type checks with `ty` in a `uvx` environment that matches the script header dependencies.

## Coding Style & Naming Conventions
Target Python 3.11+ and add dependencies only in the script header’s `# dependencies = [...]` block. Follow the existing style: 4-space indentation, `snake_case` for functions, variables, and helpers, `PascalCase` for classes/dataclasses, and explicit type hints on new code. Keep Click entrypoints thin; push parsing, validation, and API logic into focused helper functions.

## Testing Guidelines
This repo currently uses built-in self-tests instead of a separate `tests/` package. Add new regression coverage as `self_test_<behavior>()` functions and register each case in `SELF_TEST_CASES`. Keep tests deterministic and offline when possible, especially around provider/network behavior. Before submitting changes, run `./nano-agent self-test`, `uvx ruff check nano-agent`, `uvx ruff format nano-agent`, the `uvx ... ty check nano-agent` command above, and any targeted CLI smoke checks for the command path you touched.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commits, for example `feat(agent): ...`, `fix(cli): ...`, and `refactor: ...`. Keep commit subjects imperative and scoped when helpful. PRs should include a short summary, note user-visible CLI changes, list verification steps, and attach terminal output snippets or screenshots when changing the interactive UI.

## Security & Configuration Tips
Do not commit API keys, Copilot credentials, or conversation history; runtime data belongs under `~/.nano-agent/`. Prefer environment variables such as `OPENAI_API_KEY`, `NANO_AGENT_API_KEY`, and `OPENAI_BASE_URL` over hardcoded values. Keep this repository-level `AGENTS.md` current because the CLI loads it as working context automatically.
