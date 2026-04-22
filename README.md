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
model = "o4-mini"
```

### Authentication

On first use, the provider initiates an OAuth device code flow:

1. Displays a URL and code in the terminal
2. You open the URL in a browser and enter the code
3. Tokens are cached locally for subsequent use

Requires "Sign in with device code" to be enabled in your ChatGPT account security settings.

## Environment Variables

No API key required. Authentication is handled via OAuth device code flow.

Token cache location can be configured via:

```bash
export CHATGPT_TOKEN_PATH="~/.config/amplifier/chatgpt_tokens.json"
```

## Features

- OAuth device code authentication (no API key needed)
- Raw httpx + manual SSE streaming (not the OpenAI SDK)
- Automatic token refresh
- Tool calling support
- Reasoning model support (o4-mini, o3, etc.)

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
