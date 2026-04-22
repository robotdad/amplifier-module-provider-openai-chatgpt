# Amplifier ChatGPT Subscription Provider Module

ChatGPT subscription auth provider for [Amplifier](https://github.com/microsoft/amplifier) -- uses raw HTTP + manual SSE against the ChatGPT backend API (`chatgpt.com/backend-api/codex/responses`).

## Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- A ChatGPT Plus/Pro/Team subscription with device code auth enabled in ChatGPT security settings

## Purpose

Connects Amplifier to the ChatGPT backend API using OAuth device code authentication. This is a separate module from `provider-openai` because the ChatGPT backend is a distinct, undocumented API surface that rejects many standard OpenAI API parameters and requires raw HTTP + manual SSE parsing (the OpenAI Python SDK's streaming accumulator does not work against it).

## Contract

| Field | Value |
|-------|-------|
| Module Type | Provider |
| Mount Point | `providers` |
| Entry Point | `amplifier_module_provider_openai_chatgpt:mount` |

## Configuration

```toml
[providers.provider-openai-chatgpt]
default_model = "o4-mini"
```

### All Config Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_model` | str | `"gpt-4o"` | Model to use for inference |
| `raw` | bool | `false` | Include full request/response payloads in `llm:request`/`llm:response` hook events (for debugging) |
| `login_on_mount` | bool | `true` | Trigger interactive device code login if tokens are absent or expired. Set `false` for non-interactive environments. |
| `token_file_path` | str | `~/.amplifier/openai-chatgpt-oauth.json` | Path to the OAuth token JSON file |
| `timeout` | float | `300.0` | HTTP timeout in seconds for streaming requests |

### Authentication

On first use, the provider initiates an OAuth device code flow:

1. Displays a verification URL (`https://auth.openai.com/codex/device`) and a code in the terminal
2. You open the URL in a browser and enter the code
3. Tokens are cached to `~/.amplifier/openai-chatgpt-oauth.json` for subsequent use

Tokens auto-refresh silently when they expire. If the refresh token itself expires, the device code flow runs again.

Requires "Sign in with device code" to be enabled in your ChatGPT account security settings (Settings > Security).

Works in SSH/headless sessions -- the device code flow only requires a browser on any device, not the machine running Amplifier.

## Features

- OAuth device code authentication with PKCE (no API key needed)
- Raw httpx + manual SSE streaming (not the OpenAI SDK)
- Automatic token refresh with 4-step fallback chain
- Tool calling support
- Reasoning model support (o4-mini, o3, etc.)
- `-fast` model suffix support (e.g. `o4-mini-fast` -> `o4-mini` with `service_tier: "priority"`)
- `llm:request`/`llm:response` hook events with optional raw payload inclusion

## Local Development

```bash
# Clone
git clone https://github.com/robotdad/amplifier-module-provider-openai-chatgpt.git
cd amplifier-module-provider-openai-chatgpt

# Install deps (including dev group: amplifier-core, pytest, ruff)
uv sync

# Run tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_sse.py -v

# Lint and format check
uv run ruff check .
uv run ruff format --check .
```

### Testing with a Local Amplifier Install

**Option A: CLI source override (quickest for dev testing)**

```bash
# Register the local checkout
amplifier source add provider-openai-chatgpt \
  /path/to/amplifier-module-provider-openai-chatgpt \
  --local

# Verify
amplifier source list

# Test
amplifier run "Hello, can you hear me?"

# Cleanup when done
amplifier source remove provider-openai-chatgpt --local
```

The `--local` flag writes to `.amplifier/settings.local.yaml` (gitignored). No `file:///` prefix needed -- just a bare path.

Your active bundle must include the provider in its `providers:` section for Amplifier to load it.

**Option B: Inline source in a test bundle (self-contained)**

Create a test bundle file (e.g. `test-chatgpt.md`):

```markdown
---
bundle:
  name: test-openai-chatgpt
  version: 0.1.0

includes:
  - bundle: git+https://github.com/microsoft/amplifier-foundation@main

providers:
  - module: provider-openai-chatgpt
    source: /path/to/amplifier-module-provider-openai-chatgpt
    config:
      default_model: o4-mini
---

# Test: provider-openai-chatgpt
```

Then run directly against it:

```bash
amplifier run --bundle ./test-chatgpt.md "Hello, can you hear me?"
```

## Dependencies

- `httpx` - HTTP client for raw API requests

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
