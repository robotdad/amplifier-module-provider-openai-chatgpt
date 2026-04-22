"""ChatGPT subscription provider for Amplifier.

Implements the Amplifier Provider Protocol using raw httpx + manual SSE
against the ChatGPT backend API with OAuth authentication.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from amplifier_core import ModelInfo, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    TextBlock,  # noqa: F401 — used in complete() implementation
    ToolCall,
    ToolCallBlock,  # noqa: F401 — used in complete() implementation
    Usage,  # noqa: F401 — used in complete() implementation
)
from amplifier_core.utils import redact_secrets  # noqa: F401 — used in complete() implementation

from ._sse import ParsedResponse, SSEError, parse_sse_events  # noqa: F401 — used in complete()
from .oauth import (
    CHATGPT_CODEX_BASE_URL,  # noqa: F401 — used in complete() implementation
    is_token_valid,  # noqa: F401 — used in complete() implementation
    load_tokens,  # noqa: F401 — used in complete() implementation
    refresh_tokens,  # noqa: F401 — used in complete() implementation
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

CHATGPT_MODELS: list[dict[str, Any]] = [
    # GPT-5.4 (supports none/low/medium/high/xhigh reasoning)
    {"name": "gpt-5.4", "context_window": 272000, "max_output_tokens": 128000},
    {"name": "gpt-5.4-pro", "context_window": 272000, "max_output_tokens": 128000},
    {"name": "gpt-5.4-fast", "context_window": 272000, "max_output_tokens": 128000},
    {"name": "gpt-5.4-mini", "context_window": 400000, "max_output_tokens": 128000},
    # GPT-5.3 codex
    {"name": "gpt-5.3-codex", "context_window": 272000, "max_output_tokens": 128000},
    {
        "name": "gpt-5.3-codex-spark",
        "context_window": 128000,
        "max_output_tokens": 128000,
    },
    # GPT-5.2 models (supports none/low/medium/high/xhigh reasoning)
    {"name": "gpt-5.2", "context_window": 272000, "max_output_tokens": 128000},
    {"name": "gpt-5.2-codex", "context_window": 272000, "max_output_tokens": 128000},
    # GPT-5.1 models
    {"name": "gpt-5.1", "context_window": 272000, "max_output_tokens": 128000},
    {"name": "gpt-5.1-codex", "context_window": 272000, "max_output_tokens": 128000},
    {
        "name": "gpt-5.1-codex-mini",
        "context_window": 272000,
        "max_output_tokens": 128000,
    },
    {
        "name": "gpt-5.1-codex-max",
        "context_window": 272000,
        "max_output_tokens": 128000,
    },
    # GPT-5 Codex models (original)
    {"name": "gpt-5-codex-mini", "context_window": 272000, "max_output_tokens": 128000},
    # GPT-4 models (for ChatGPT Plus users)
    {"name": "gpt-4o", "context_window": 128000, "max_output_tokens": 16384},
    {"name": "gpt-4o-mini", "context_window": 128000, "max_output_tokens": 16384},
    # o1 series
    {"name": "o1", "context_window": 200000, "max_output_tokens": 100000},
    {"name": "o1-pro", "context_window": 200000, "max_output_tokens": 100000},
    {"name": "o3", "context_window": 200000, "max_output_tokens": 100000},
    {"name": "o3-mini", "context_window": 200000, "max_output_tokens": 100000},
    {"name": "o4-mini", "context_window": 200000, "max_output_tokens": 100000},
]


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class ChatGPTProvider:
    """Amplifier provider for ChatGPT subscription API (OAuth-authenticated)."""

    name = "openai-chatgpt"

    def __init__(
        self,
        config: dict[str, Any],
        coordinator: Any,
        tokens: dict[str, Any] | None,
    ) -> None:
        self._config = config
        self._coordinator = coordinator
        self._tokens = tokens

        self.raw: bool = bool(config.get("raw", False))
        self.default_model: str = config.get("default_model", "gpt-4o")
        self.timeout: float = float(config.get("timeout", 300.0))

        # Lazy httpx client — created on first use
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Provider Protocol
    # ------------------------------------------------------------------

    def get_info(self) -> ProviderInfo:
        """Return provider metadata."""
        return ProviderInfo(
            id="openai-chatgpt",
            display_name="OpenAI ChatGPT",
            capabilities=["streaming", "tools"],
        )

    def list_models(self) -> list[ModelInfo]:
        """Return ModelInfo objects for all known ChatGPT models."""
        return [
            ModelInfo(
                id=entry["name"],
                display_name=entry["name"],
                context_window=entry["context_window"],
                max_output_tokens=entry["max_output_tokens"],
            )
            for entry in CHATGPT_MODELS
        ]

    def parse_tool_calls(self, raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Filter tool calls with non-None arguments and convert to ToolCall objects."""
        result: list[ToolCall] = []
        for call in raw_calls:
            if call.get("arguments") is None:
                continue
            result.append(
                ToolCall(
                    id=call.get("id", ""),
                    name=call.get("name", ""),
                    arguments=call["arguments"],
                )
            )
        return result

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a completion request. Not yet implemented."""
        raise NotImplementedError(
            "complete() is not yet implemented for ChatGPTProvider"
        )
