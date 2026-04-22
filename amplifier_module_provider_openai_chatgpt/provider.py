"""ChatGPT subscription provider for Amplifier.

Implements the Amplifier Provider Protocol using raw httpx + manual SSE
against the ChatGPT backend API with OAuth authentication.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from amplifier_core import ModelInfo, ProviderInfo
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultBlock,
    Usage,
)
from amplifier_core.utils import redact_secrets

from ._sse import ParsedResponse, parse_sse_events
from .oauth import (
    CHATGPT_CODEX_BASE_URL,
    is_token_valid,
    load_tokens,
    refresh_tokens,
)

logger = logging.getLogger(__name__)

# Full endpoint for the ChatGPT Responses API
CHATGPT_CODEX_ENDPOINT = CHATGPT_CODEX_BASE_URL + "/responses"

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
        config: dict[str, Any] | None = None,
        coordinator: Any = None,
        tokens: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        self._coordinator = coordinator
        self._tokens = tokens

        self.priority: int = int(self._config.get("priority", 100))
        self.raw: bool = bool(self._config.get("raw", False))
        self.default_model: str = self._config.get("default_model", "gpt-4o")
        self.timeout: float = float(self._config.get("timeout", 300.0))
        self._token_file_path: str | None = self._config.get("token_file_path")

        # No persistent client — httpx.AsyncClient is created per-request in complete().
        # This is intentional: token refresh may change headers between calls.

    # ------------------------------------------------------------------
    # Provider Protocol
    # ------------------------------------------------------------------

    def get_info(self) -> ProviderInfo:
        """Return provider metadata."""
        return ProviderInfo(
            id="openai-chatgpt",
            display_name="OpenAI ChatGPT",
            capabilities=["streaming", "tools", "reasoning"],
        )

    async def list_models(self) -> list[ModelInfo]:
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

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from a ChatResponse.

        Args:
            response: Typed chat response containing tool_calls.

        Returns:
            List of ToolCall objects from the response, or [] if none present.
        """
        if not response.tool_calls:
            return []
        return list(response.tool_calls)

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _convert_content(
        self, content: str | list[Any], role: str = "user"
    ) -> list[dict[str, Any]]:
        """Convert Amplifier message content to Responses API content format.

        - str → [{type: input_text|output_text, text}]
        - TextBlock → {type: input_text|output_text, text}
        - ThinkingBlock → {type: input_text|output_text, text: block.thinking}
        - Other block types (ToolCallBlock, ToolResultBlock) are skipped here
          and handled directly in _build_payload.

        Uses ``output_text`` when role is ``"assistant"``; ``input_text`` otherwise.
        """
        text_type = "output_text" if role == "assistant" else "input_text"

        if isinstance(content, str):
            return [{"type": text_type, "text": content}]

        result: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, TextBlock):
                result.append({"type": text_type, "text": block.text})
            elif isinstance(block, ThinkingBlock):
                result.append({"type": text_type, "text": block.thinking})
            # ToolCallBlock and ToolResultBlock are handled in _build_payload
        return result

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Build Responses API payload from an Amplifier ChatRequest.

        Key rules enforced:
        - Uses ``input`` array (not ``messages``)
        - First system/developer message → top-level ``instructions``
        - ``stream: True`` and ``store: False`` are mandatory
        - Rejected params (max_output_tokens, temperature, truncation,
          parallel_tool_calls, include) are never included
        - ``-fast`` model suffix → strip suffix + ``service_tier: 'priority'``
        - request.model overrides default_model
        - Tools → {type, name, description, parameters} + tool_choice: 'auto'
        - ToolResultBlock → {type: function_call_output, call_id, output}
        - ToolCallBlock → {type: function_call, call_id, name, arguments}
        - Reasoning effort → {reasoning: {effort, summary: 'detailed'}}
        """
        # Resolve model (request overrides provider default)
        model: str = request.model or self.default_model

        # Handle -fast suffix → priority service tier
        service_tier: str | None = None
        if model.endswith("-fast"):
            model = model.removesuffix("-fast")
            service_tier = "priority"

        # Build input array and extract instructions from system/developer message
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []

        for message in request.messages:
            # First system/developer message becomes top-level instructions
            if message.role in ("system", "developer") and instructions is None:
                if isinstance(message.content, str):
                    instructions = message.content
                else:
                    texts = [
                        block.text
                        for block in message.content
                        if isinstance(block, TextBlock)
                    ]
                    instructions = " ".join(texts) if texts else ""
                continue  # Do not add to input array

            if message.role == "assistant":
                if isinstance(message.content, list):
                    # Split mixed content: text blocks go in role message,
                    # tool call blocks become standalone function_call items
                    text_parts: list[dict[str, Any]] = []
                    for block in message.content:
                        if isinstance(block, ToolCallBlock):
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": block.id,
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                }
                            )
                        elif isinstance(block, TextBlock):
                            text_parts.append(
                                {"type": "output_text", "text": block.text}
                            )
                        elif isinstance(block, ThinkingBlock):
                            text_parts.append(
                                {"type": "output_text", "text": block.thinking}
                            )
                    if text_parts:
                        input_items.append({"role": "assistant", "content": text_parts})
                else:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": self._convert_content(
                                message.content, role="assistant"
                            ),
                        }
                    )

            elif message.role == "tool":
                # Tool result messages → standalone function_call_output items
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            output = block.output
                            if not isinstance(output, str):
                                output = json.dumps(output)
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": block.tool_call_id,
                                    "output": output,
                                }
                            )
                else:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": getattr(message, "tool_call_id", "unknown"),
                            "output": str(message.content),
                        }
                    )

            else:
                # user, developer (additional after first), function
                input_items.append(
                    {
                        "role": message.role,
                        "content": self._convert_content(message.content),
                    }
                )

        # Assemble base payload (no rejected params)
        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
            "store": False,
        }

        # ChatGPT backend requires instructions even when no system message is present.
        payload["instructions"] = instructions or ""

        if service_tier is not None:
            payload["service_tier"] = service_tier

        # Tools → Responses API format
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"

        # Reasoning effort
        if request.reasoning_effort:
            payload["reasoning"] = {
                "effort": request.reasoning_effort,
                "summary": "detailed",
            }

        return payload

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Return the HTTP headers required for the ChatGPT Codex API.

        Reads ``access_token`` and ``account_id`` from ``self._tokens``.

        Returns:
            Dict with six required headers.

        Raises:
            ValueError: If ``access_token`` is absent or ``_tokens`` is None.
            ValueError: If ``account_id`` is absent.
        """
        if not self._tokens or not self._tokens.get("access_token"):
            raise ValueError("No valid OAuth tokens available")

        account_id = self._tokens.get("account_id")
        if not account_id:
            raise ValueError("No account_id in tokens — cannot build request headers")

        return {
            "Authorization": f"Bearer {self._tokens['access_token']}",
            "ChatGPT-Account-Id": account_id,
            "OpenAI-Beta": "responses=v1",
            "OpenAI-Originator": "codex",
            "Content-Type": "application/json",
            "accept": "text/event-stream",
        }

    async def close(self) -> None:
        """No-op — httpx clients are created per-request in complete()."""

    async def _ensure_valid_tokens(self) -> None:
        """Guarantee ``self._tokens`` holds a valid, unexpired access token.

        Resolution order:
        1. In-memory tokens pass ``is_token_valid()`` → done.
        2. Tokens loaded from disk pass ``is_token_valid()`` → update in-memory, done.
        3. Refresh using in-memory ``refresh_token`` → update in-memory, done.
        4. Refresh using disk ``refresh_token`` → update in-memory, done.
        5. None of the above succeeded → raise ``ValueError``.

        Raises:
            ValueError: If no valid tokens can be obtained by any means.
        """
        # 1. In-memory tokens still valid.
        if is_token_valid(self._tokens):
            return

        # 2. Fresh load from disk.
        disk_tokens = load_tokens(path=self._token_file_path)
        if is_token_valid(disk_tokens):
            self._tokens = disk_tokens
            return

        # 3. Refresh using in-memory refresh_token.
        if self._tokens and self._tokens.get("refresh_token"):
            refreshed = await refresh_tokens(
                self._tokens["refresh_token"], path=self._token_file_path
            )
            if refreshed:
                self._tokens = refreshed
                return

        # 4. Refresh using disk refresh_token (different from in-memory).
        if disk_tokens and disk_tokens.get("refresh_token"):
            refreshed = await refresh_tokens(
                disk_tokens["refresh_token"], path=self._token_file_path
            )
            if refreshed:
                self._tokens = refreshed
                return

        raise ValueError(
            "No valid OAuth tokens available — please run the login flow again"
        )

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Send a completion request to the ChatGPT Responses API.

        Flow:
        1. Ensure valid OAuth tokens.
        2. Build request payload.
        3. Emit ``llm:request`` event.
        4. POST to CHATGPT_CODEX_ENDPOINT with httpx streaming, collect SSE lines.
        5. Check HTTP status (non-200 raises ValueError).
        6. Parse SSE events.
        7. Emit ``llm:response`` event with usage and timing.
        8. Return ChatResponse via ``_to_chat_response()``.

        On any exception an ``llm:response`` event with ``status='error'`` is
        emitted before re-raising.
        """
        # 1. Ensure valid OAuth tokens.
        await self._ensure_valid_tokens()

        # 2. Build request payload.
        payload = self._build_payload(request)

        # Resolve effective model name (mirrors _build_payload logic) for events.
        model: str = request.model or self.default_model
        if model.endswith("-fast"):
            model = model.removesuffix("-fast")

        headers = self._build_headers()

        # 3. Emit llm:request event.
        _has_hooks = self._coordinator and hasattr(self._coordinator, "hooks")
        if _has_hooks:
            req_event: dict[str, Any] = {
                "provider": self.name,
                "model": model,
                "message_count": len(request.messages),
            }
            if self.raw:
                req_event["raw"] = redact_secrets(payload)
            await self._coordinator.hooks.emit("llm:request", req_event)

        start_time = time.monotonic()

        try:
            # 4. POST with httpx streaming, collect SSE lines.
            lines: list[str] = []
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    CHATGPT_CODEX_ENDPOINT,
                    json=payload,
                    headers=headers,
                ) as resp:
                    # 5. Check HTTP status.
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise ValueError(
                            f"ChatGPT API error {resp.status_code}: "
                            f"{error_body.decode(errors='replace')}"
                        )
                    async for line in resp.aiter_lines():
                        lines.append(line)

            # 6. Parse SSE events.
            parsed = parse_sse_events(lines)

            duration_ms = (time.monotonic() - start_time) * 1000

            # 7. Emit llm:response event (success).
            if _has_hooks:
                resp_event: dict[str, Any] = {
                    "provider": self.name,
                    "model": model,
                    "usage": {
                        "input_tokens": parsed.input_tokens,
                        "output_tokens": parsed.output_tokens,
                    },
                    "status": "ok",
                    "duration_ms": duration_ms,
                }
                if self.raw:
                    resp_event["raw"] = redact_secrets({"events": parsed.raw_events})
                await self._coordinator.hooks.emit("llm:response", resp_event)

            # 8. Return ChatResponse.
            return self._to_chat_response(parsed, model)

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            if _has_hooks:
                await self._coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": model,
                        "status": "error",
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise

    def _to_chat_response(self, parsed: ParsedResponse, model: str) -> ChatResponse:
        """Convert a ``ParsedResponse`` into an Amplifier ``ChatResponse``.

        Text content → ``TextBlock``.
        Tool calls → ``ToolCallBlock`` (in ``content``) + ``ToolCall`` (in
        ``tool_calls``).  JSON arguments are parsed; malformed JSON falls back
        to ``{"_raw": <original_string>}``.
        ``finish_reason`` is ``"tool_calls"`` when tool calls are present,
        otherwise ``"stop"``.
        """
        content_blocks: list[Any] = []
        tool_call_list: list[ToolCall] = []

        # Text content → TextBlock
        if parsed.content:
            content_blocks.append(TextBlock(text=parsed.content))

        # Tool calls → ToolCallBlock + ToolCall
        for tc in parsed.tool_calls:
            func = tc.get("function", {})
            name: str = func.get("name", "")
            call_id: str = tc.get("id", "")
            raw_args: str = func.get("arguments", "")

            # Parse JSON arguments; fall back to {"_raw": ...} on failure.
            try:
                arguments: dict[str, Any] = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, ValueError):
                arguments = {"_raw": raw_args}

            content_blocks.append(ToolCallBlock(id=call_id, name=name, input=arguments))
            tool_call_list.append(ToolCall(id=call_id, name=name, arguments=arguments))

        finish_reason = "tool_calls" if tool_call_list else "stop"

        usage = Usage(
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            total_tokens=parsed.input_tokens + parsed.output_tokens,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_call_list if tool_call_list else None,
            usage=usage,
            finish_reason=finish_reason,
        )
