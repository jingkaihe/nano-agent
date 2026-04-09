# nano-agent

`nano-agent` is a tiny single-file coding agent CLI.

## Install

By default the installer drops `nano-agent` into `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/jingkaihe/nano-agent/main/install.sh | bash
```

Install a different branch:

```bash
curl -fsSL https://raw.githubusercontent.com/jingkaihe/nano-agent/main/install.sh | bash -s -- --branch my-branch
```

`nano-agent` runs via [`uv`](https://docs.astral.sh/uv/), and the installer uses `git`, so make sure both are installed and that `~/.local/bin` is on your `PATH`.

## Enterprise usage

If you are running `nano-agent` in an enterprise environment, export the required `uv` and company-specific environment variables before installing or running the CLI.

```bash
export UV_NATIVE_TLS=true
export BUSINESS_COPILOT=true
export UV_DEFAULT_INDEX=http://example.com/pypi/simple
```

This is especially useful when your company requires native TLS trust settings and an internal Python package index.
