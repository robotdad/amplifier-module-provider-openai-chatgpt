# ChatGPT Provider Module Implementation Plan

> **Execution:** Use the subagent-driven-development workflow to implement this plan.

**Goal:** Build a standalone Amplifier provider module that authenticates with ChatGPT via OAuth device code flow and sends inference requests to the ChatGPT backend API using raw httpx + manual SSE parsing.

**Architecture:** Separate module (`amplifier-module-provider-openai-chatgpt`) that does NOT use the OpenAI Python SDK. The ChatGPT backend API (`chatgpt.com/backend-api/codex/responses`) is undocumented, rejects many standard OpenAI parameters, and requires mandatory streaming — every working Python implementation (Letta, codex-backend-sdk, LiteLLM) uses raw HTTP + manual SSE. OAuth device code flow handles authentication, with token refresh and local file persistence. SSE events are parsed from raw `data:` lines, with `response.output_item.done` as the canonical accumulation event.

**Tech Stack:** Python 3.11+, httpx (HTTP client + SSE streaming), pytest + pytest-asyncio (testing), hatchling (build), amplifier-core (provider protocol, models, utilities)

---

## Phase 1: Core Infrastructure (Tasks 1–12)

This plan covers the complete module implementation. Tasks are ordered bottom-up: OAuth → SSE Parser → Provider → Mount.

---

### Task 1: Copy and adapt oauth.py constants and token storage

**Files:**
- Create: `amplifier_module_provider_openai_chatgpt/oauth.py`
- Test: `tests/test_oauth.py`

**Context:** The OAuth code exists on the `feat/oauth-subscription-auth` branch of `amplifier-module-provider-openai` at `/home/robotdad/Work/openaisub/amplifier-module-provider-openai/amplifier_module_provider_openai/oauth.py`. That branch is already checked out. Copy the entire file and adapt imports. The module is self-contained (uses only stdlib + no amplifier imports).

**Step 1: Write the failing test**

Create `tests/test_oauth.py`:

```python
"""Tests for OAuth constants, PKCE helpers, and token storage."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta, timezone

from amplifier_module_provider_openai_chatgpt.oauth import (
    CHATGPT_CODEX_BASE_URL,
    DEVICE_CODE_CALLBACK_URL,
    DEVICE_CODE_POLL_INTERVAL,
    DEVICE_CODE_TOKEN_URL,
    DEVICE_CODE_USERCODE_URL,
    DEVICE_CODE_VERIFICATION_URL,
    OAUTH_CLIENT_ID,
    OAUTH_ISSUER,
    OAUTH_SCOPES,
    OAUTH_TOKEN_URL,
    TOKEN_FILE_PATH,
    generate_pkce_pair,
    is_token_valid,
    load_tokens,
    save_tokens,
)


class TestConstants:
    """Verify all OAuth constant values."""

    def test_oauth_issuer(self):
        assert OAUTH_ISSUER == "https://auth.openai.com"

    def test_oauth_token_url(self):
        assert OAUTH_TOKEN_URL == "https://auth.openai.com/oauth/token"

    def test_oauth_client_id(self):
        assert OAUTH_CLIENT_ID == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_oauth_scopes(self):
        assert OAUTH_SCOPES == "openid profile email offline_access"

    def test_device_code_usercode_url(self):
        assert (
            DEVICE_CODE_USERCODE_URL
            == "https://auth.openai.com/api/accounts/deviceauth/usercode"
        )

    def test_device_code_token_url(self):
        assert (
            DEVICE_CODE_TOKEN_URL
            == "https://auth.openai.com/api/accounts/deviceauth/token"
        )

    def test_device_code_verification_url(self):
        assert DEVICE_CODE_VERIFICATION_URL == "https://auth.openai.com/codex/device"

    def test_device_code_poll_interval(self):
        assert DEVICE_CODE_POLL_INTERVAL == 5

    def test_chatgpt_codex_base_url(self):
        assert CHATGPT_CODEX_BASE_URL == "https://chatgpt.com/backend-api/codex"

    def test_token_file_path(self):
        assert TOKEN_FILE_PATH == "~/.amplifier/openai-chatgpt-oauth.json"

    def test_device_code_callback_url(self):
        assert DEVICE_CODE_CALLBACK_URL == "https://auth.openai.com/deviceauth/callback"
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_oauth.py::TestConstants -v
```
Expected: FAIL with ImportError (oauth.py doesn't exist yet)

**Step 3: Write the implementation**

Create `amplifier_module_provider_openai_chatgpt/oauth.py` by copying from `/home/robotdad/Work/openaisub/amplifier-module-provider-openai/amplifier_module_provider_openai/oauth.py` and making these changes:

1. Change `TOKEN_FILE_PATH` from `"~/.amplifier/openai-oauth.json"` to `"~/.amplifier/openai-chatgpt-oauth.json"` (avoid collision with the standard OpenAI provider)
2. Remove `OAUTH_AUTHORIZE_URL` (not needed — we only use device code flow)
3. Remove `OAUTH_CALLBACK_PORT` and `OAUTH_CALLBACK_URL` (browser flow not needed for mount)
4. Remove `SUBSCRIPTION_MODELS` (model catalog is in provider.py)
5. Remove `start_browser_flow()` function entirely (not needed)
6. Keep `DEVICE_CODE_CALLBACK_URL` — it's the redirect_uri used in the token exchange step
7. Update the User-Agent string from `"amplifier-openai-provider/1.0"` to `"amplifier-openai-chatgpt-provider/1.0"`
8. In `login()`: the existing code on the branch already only uses device code flow — keep that behavior

Keep these functions unchanged (they work):
- `save_tokens()`, `load_tokens()`, `is_token_valid()`
- `refresh_tokens()`
- `exchange_code_for_tokens()`
- `extract_account_id()`
- `generate_pkce_pair()`
- `start_device_code_flow()`
- `login()`

**Step 4: Run test to verify it passes**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_oauth.py::TestConstants -v
```
Expected: PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/oauth.py tests/test_oauth.py
git commit -m "feat: add oauth.py with constants, PKCE, and token storage"
```

---

### Task 2: Test PKCE helpers and token storage functions

**Files:**
- Modify: `tests/test_oauth.py` (add test classes)

**Step 1: Add PKCE and token storage tests**

Append to `tests/test_oauth.py`:

```python
import base64
import hashlib


class TestPKCE:
    """Verify PKCE helper functions per RFC 7636."""

    def test_returns_tuple_of_two_strings(self):
        result = generate_pkce_pair()
        assert isinstance(result, tuple)
        assert len(result) == 2
        verifier, challenge = result
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length_in_range(self):
        verifier, _ = generate_pkce_pair()
        assert 43 <= len(verifier) <= 128

    def test_verifier_is_url_safe(self):
        import re
        verifier, _ = generate_pkce_pair()
        assert re.match(r"^[A-Za-z0-9\-._~]+$", verifier)

    def test_challenge_is_sha256_of_verifier(self):
        verifier, challenge = generate_pkce_pair()
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_each_call_returns_unique_pair(self):
        pair1 = generate_pkce_pair()
        pair2 = generate_pkce_pair()
        assert pair1[0] != pair2[0]


class TestSaveTokens:
    """Verify save_tokens writes tokens to disk correctly."""

    def test_creates_file_with_correct_content(self, tmp_path):
        tokens = {"access_token": "abc", "refresh_token": "xyz"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == tokens

    def test_file_has_0600_permissions(self, tmp_path):
        tokens = {"access_token": "abc"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path)
        file_stat = (tmp_path / "tokens.json").stat()
        permissions = stat.S_IMODE(file_stat.st_mode)
        assert permissions == 0o600

    def test_creates_parent_directory_if_missing(self, tmp_path):
        tokens = {"access_token": "abc"}
        path = str(tmp_path / "nested" / "dir" / "tokens.json")
        save_tokens(tokens, path)
        assert (tmp_path / "nested" / "dir" / "tokens.json").exists()


class TestLoadTokens:
    """Verify load_tokens reads tokens from disk correctly."""

    def test_returns_dict_for_valid_file(self, tmp_path):
        tokens = {"access_token": "abc", "refresh_token": "xyz"}
        path = str(tmp_path / "tokens.json")
        with open(path, "w") as f:
            json.dump(tokens, f)
        result = load_tokens(path)
        assert result == tokens

    def test_returns_none_for_missing_file(self, tmp_path):
        result = load_tokens(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        assert load_tokens(path) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        (tmp_path / "tokens.json").touch()
        assert load_tokens(path) is None


class TestIsTokenValid:
    """Verify is_token_valid checks token existence and expiry."""

    def _future_expires_at(self) -> str:
        return (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()

    def _past_expires_at(self) -> str:
        return (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()

    def test_valid_token_not_expired_returns_true(self):
        tokens = {"access_token": "tok_abc", "expires_at": self._future_expires_at()}
        assert is_token_valid(tokens) is True

    def test_expired_token_returns_false(self):
        tokens = {"access_token": "tok_abc", "expires_at": self._past_expires_at()}
        assert is_token_valid(tokens) is False

    def test_none_tokens_returns_false(self):
        assert is_token_valid(None) is False

    def test_missing_access_token_returns_false(self):
        assert is_token_valid({"expires_at": self._future_expires_at()}) is False

    def test_missing_expires_at_returns_false(self):
        assert is_token_valid({"access_token": "tok"}) is False
```

**Step 2: Run tests**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_oauth.py -v
```
Expected: All PASS

**Step 3: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add tests/test_oauth.py
git commit -m "test: add PKCE, token storage, and token validation tests"
```

---

### Task 3: Test token refresh and JWT extraction

**Files:**
- Modify: `tests/test_oauth.py` (add test classes)

**Step 1: Add tests**

Append to `tests/test_oauth.py`:

```python
import asyncio
from io import BytesIO
from unittest.mock import MagicMock, patch

from amplifier_module_provider_openai_chatgpt.oauth import (
    extract_account_id,
    refresh_tokens,
)


def _mock_urlopen_response(data: dict) -> MagicMock:
    """Create a mock urllib response that returns JSON-encoded data."""
    encoded = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_jwt(payload: dict) -> str:
    """Create a minimal fake JWT with the given payload (unsigned)."""
    header_b64 = (
        base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}')
        .rstrip(b"=")
        .decode("ascii")
    )
    payload_bytes = json.dumps(payload).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode("ascii")
    return f"{header_b64}.{payload_b64}.fakesignature"


class TestExtractAccountId:
    """Verify extract_account_id decodes JWT payload and extracts account ID."""

    def test_extracts_from_openai_profile_claim(self):
        token = _make_jwt({
            "sub": "some-sub",
            "https://api.openai.com/profile": {"account_id": "acct_123"},
        })
        assert extract_account_id(token) == "acct_123"

    def test_falls_back_to_sub_claim(self):
        token = _make_jwt({"sub": "sub-fallback-id"})
        assert extract_account_id(token) == "sub-fallback-id"

    def test_returns_empty_for_invalid_jwt(self):
        assert extract_account_id("invalid.jwt.token") == ""

    def test_returns_empty_for_empty_string(self):
        assert extract_account_id("") == ""


class TestRefreshTokens:
    """Verify refresh_tokens exchanges a refresh token for new credentials."""

    def test_successful_refresh_returns_new_tokens(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        save_tokens({"account_id": "acct_123"}, path)

        mock_resp = _mock_urlopen_response({
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "id_token": "new_id",
            "expires_in": 3600,
        })
        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            return_value=mock_resp,
        ):
            result = asyncio.run(refresh_tokens("old_refresh", path=path))

        assert result is not None
        assert result["auth_mode"] == "oauth"
        assert result["access_token"] == "new_access"
        assert result["account_id"] == "acct_123"

    def test_refresh_failure_returns_none(self, tmp_path):
        from urllib.error import HTTPError

        path = str(tmp_path / "tokens.json")
        http_error = HTTPError(
            url="https://auth.openai.com/oauth/token",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b'{"error": "invalid_grant"}'),
        )
        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=http_error,
        ):
            result = asyncio.run(refresh_tokens("bad_refresh", path=path))

        assert result is None
```

**Step 2: Run tests**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_oauth.py -v
```
Expected: All PASS

**Step 3: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add tests/test_oauth.py
git commit -m "test: add JWT extraction and token refresh tests"
```

---

### Task 4: Test device code flow

**Files:**
- Modify: `tests/test_oauth.py` (add test class)

**Step 1: Add test**

Append to `tests/test_oauth.py`:

```python
from unittest.mock import AsyncMock

from amplifier_module_provider_openai_chatgpt.oauth import (
    start_device_code_flow,
)


class TestDeviceCodeFlow:
    """Verify start_device_code_flow performs device code authorization."""

    def test_requests_device_code_and_returns_auth_code(self):
        usercode_response = _mock_urlopen_response({
            "user_code": "ABCD-EFGH",
            "device_code": "dev_code_xyz",
            "interval": 5,
        })
        token_response = _mock_urlopen_response({
            "authorization_code": "auth_code_123",
        })

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, token_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(start_device_code_flow())

        assert result["authorization_code"] == "auth_code_123"
        assert "code_verifier" in result

    def test_handles_authorization_pending_then_success(self):
        usercode_response = _mock_urlopen_response({
            "user_code": "WXYZ-1234",
            "device_code": "dev_code_abc",
            "interval": 5,
        })
        pending = _mock_urlopen_response({"error": "authorization_pending"})
        token_response = _mock_urlopen_response({"authorization_code": "auth_456"})

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, pending, token_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = asyncio.run(start_device_code_flow())

        assert mock_sleep.call_count == 2
        assert result["authorization_code"] == "auth_456"

    def test_expired_device_code_raises(self):
        import pytest

        usercode_response = _mock_urlopen_response({
            "user_code": "ABCD-EFGH",
            "device_code": "dev_code_xyz",
            "interval": 5,
        })
        expired = _mock_urlopen_response({"error": "expired_token"})

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, expired],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="Device code expired"):
                    asyncio.run(start_device_code_flow())
```

**Note:** The device code poll uses `HTTPError` with specific `error.code` values in the response body. The existing test pattern from the source branch mocks `urlopen` to return successful responses or raise `HTTPError`. The `pending` responses work because the source code on the branch handles 403/404 HTTP errors with `{"error": {"code": "deviceauth_authorization_unknown"}}` in the body. However, looking at the source code more closely, it catches `HTTPError` and checks `body.get("error", {}).get("code", "")`. Make sure the mock raises `HTTPError` for pending states. **Wait** — re-reading the source, for authorization_pending the source catches `HTTPError` responses. But the test on the original branch creates mock responses that return via `urlopen()` (not raise). Looking more carefully at the source code flow:

The source `start_device_code_flow()` has a `try/except HTTPError` block. The `pending` responses come as HTTP errors (403/404), not 200 OK. So the pending mock should raise `HTTPError`, not return a successful response.

**CORRECTION to Step 1:** The pending responses in device code polling come as HTTP errors. Use this pattern instead:

```python
from urllib.error import HTTPError as _HTTPError


class TestDeviceCodeFlow:
    """Verify start_device_code_flow performs device code authorization."""

    def _make_http_error(self, body: dict, code: int = 403) -> _HTTPError:
        """Create an HTTPError with a JSON body."""
        return _HTTPError(
            url=DEVICE_CODE_TOKEN_URL,
            code=code,
            msg="Forbidden",
            hdrs={},
            fp=BytesIO(json.dumps(body).encode("utf-8")),
        )

    def test_requests_device_code_and_returns_auth_code(self):
        usercode_response = _mock_urlopen_response({
            "user_code": "ABCD-EFGH",
            "device_code": "dev_code_xyz",
            "interval": 5,
        })
        token_response = _mock_urlopen_response({
            "authorization_code": "auth_code_123",
        })

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, token_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(start_device_code_flow())

        assert result["authorization_code"] == "auth_code_123"
        assert "code_verifier" in result

    def test_handles_authorization_pending_then_success(self):
        usercode_response = _mock_urlopen_response({
            "user_code": "WXYZ-1234",
            "device_code": "dev_code_abc",
            "interval": 5,
        })
        pending = self._make_http_error(
            {"error": {"code": "deviceauth_authorization_unknown"}}
        )
        token_response = _mock_urlopen_response({"authorization_code": "auth_456"})

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, pending, token_response],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = asyncio.run(start_device_code_flow())

        assert mock_sleep.call_count == 2
        assert result["authorization_code"] == "auth_456"

    def test_expired_device_code_raises(self):
        import pytest

        usercode_response = _mock_urlopen_response({
            "user_code": "ABCD-EFGH",
            "device_code": "dev_code_xyz",
            "interval": 5,
        })
        expired = self._make_http_error(
            {"error": {"code": "deviceauth_expired"}}
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen",
            side_effect=[usercode_response, expired],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="Device code expired"):
                    asyncio.run(start_device_code_flow())
```

**Step 2: Run tests**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_oauth.py::TestDeviceCodeFlow -v
```
Expected: All PASS

**Step 3: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add tests/test_oauth.py
git commit -m "test: add device code flow tests"
```

---

### Task 5: SSE parser — parse data lines into typed events

**Files:**
- Create: `amplifier_module_provider_openai_chatgpt/_sse.py`
- Create: `tests/test_sse.py`

**Context:** The SSE parser is a pure function that takes raw SSE `data:` lines and extracts typed events. It does NOT depend on httpx or any async code — it operates on strings. The canonical accumulation event is `response.output_item.done`. Must handle both `output_text` and `text` content types. Must detect error events (`error`, `response.failed`, `response.incomplete`) inside 200 streams.

Reference: Letta's `_accumulate_sse_response()` at lines 451-544 of `/home/robotdad/Work/openaisub/letta/letta/llm_api/chatgpt_oauth_client.py`.

**Step 1: Write the failing test**

Create `tests/test_sse.py`:

```python
"""Tests for SSE event parsing."""

from __future__ import annotations

import json

from amplifier_module_provider_openai_chatgpt._sse import (
    ParsedResponse,
    SSEError,
    parse_sse_events,
)


def _sse_line(event: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(event)}"


class TestParseSSEEvents:
    """Verify parse_sse_events accumulates SSE lines into a ParsedResponse."""

    def test_simple_text_response(self):
        """Accumulates text from response.output_item.done with output_text type."""
        lines = [
            _sse_line({
                "type": "response.created",
                "response": {"id": "resp_123", "model": "gpt-5.4"},
            }),
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello, world!"}],
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5.4",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert isinstance(result, ParsedResponse)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []
        assert result.response_id == "resp_123"
        assert result.model == "gpt-5.4"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_handles_text_content_type(self):
        """Handles 'text' content type in addition to 'output_text'."""
        lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "text", "text": "Alt text type"}],
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.content == "Alt text type"

    def test_function_call_accumulation(self):
        """Accumulates function calls from response.output_item.done."""
        lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/test.txt"}',
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == '{"path": "/tmp/test.txt"}'

    def test_mixed_text_and_tool_calls(self):
        """Handles both text content and tool calls in one response."""
        lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
            }),
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_xyz",
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.content == "Let me check."
        assert len(result.tool_calls) == 1

    def test_skips_non_data_lines(self):
        """Ignores non-data lines (empty lines, comments, event type lines)."""
        lines = [
            "",
            ": comment",
            "event: message",
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.content == "OK"

    def test_skips_malformed_json(self):
        """Skips lines with invalid JSON gracefully."""
        lines = [
            "data: {not valid json",
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Still works"}],
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.content == "Still works"

    def test_usage_from_response_done(self):
        """Extracts usage from the response.done event's response object."""
        lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hi"}],
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {
                    "id": "resp_456",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            }),
            "data: [DONE]",
        ]
        result = parse_sse_events(lines)
        assert result.input_tokens == 100
        assert result.output_tokens == 50


class TestSSEErrors:
    """Verify SSE error event detection."""

    def test_error_event_raises(self):
        import pytest

        lines = [
            _sse_line({
                "type": "error",
                "error": {"message": "context_length_exceeded", "code": "invalid_request_error"},
            }),
        ]
        with pytest.raises(SSEError, match="context_length_exceeded"):
            parse_sse_events(lines)

    def test_response_failed_raises(self):
        import pytest

        lines = [
            _sse_line({
                "type": "response.failed",
                "response": {
                    "error": {"message": "Server error", "code": "server_error"},
                },
            }),
        ]
        with pytest.raises(SSEError, match="Server error"):
            parse_sse_events(lines)

    def test_response_incomplete_raises(self):
        import pytest

        lines = [
            _sse_line({
                "type": "response.incomplete",
                "response": {
                    "error": {"message": "Incomplete response"},
                },
            }),
        ]
        with pytest.raises(SSEError, match="Incomplete response"):
            parse_sse_events(lines)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_sse.py -v
```
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Create `amplifier_module_provider_openai_chatgpt/_sse.py`:

```python
"""SSE event parser for ChatGPT backend API responses.

Parses raw Server-Sent Events (SSE) `data:` lines from the ChatGPT backend
into a structured ParsedResponse. Uses `response.output_item.done` as the
canonical accumulation event.

This module is pure (no async, no httpx dependency) and can be tested
with plain string fixtures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SSEError(Exception):
    """Error event received inside an SSE stream.

    The ChatGPT backend API can return error events (type: "error",
    "response.failed", "response.incomplete") inside a 200 OK stream.
    """

    def __init__(self, message: str, code: str = "", event_type: str = ""):
        self.message = message
        self.code = code
        self.event_type = event_type
        super().__init__(message)


@dataclass
class ParsedResponse:
    """Accumulated result from parsing an SSE stream."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    response_id: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_events: list[dict] = field(default_factory=list)


def parse_sse_events(lines: list[str]) -> ParsedResponse:
    """Parse a list of SSE lines into a ParsedResponse.

    Args:
        lines: Raw SSE lines (including "data: " prefix).

    Returns:
        ParsedResponse with accumulated content, tool calls, and metadata.

    Raises:
        SSEError: If an error event is encountered in the stream.
    """
    result = ParsedResponse()

    for line in lines:
        if not line.startswith("data: "):
            continue

        data_str = line[6:]  # Remove "data: " prefix
        if data_str == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE event: %s", data_str[:100])
            continue

        result.raw_events.append(event)

        # Extract response metadata
        if not result.response_id and event.get("id"):
            result.response_id = event["id"]
        if not result.model and event.get("model"):
            result.model = event["model"]

        event_type = event.get("type", "")

        # --- Error detection (these come inside 200 streams) ---
        if event_type == "error":
            error_info = event.get("error", {})
            raise SSEError(
                message=error_info.get("message", str(event)),
                code=error_info.get("code", ""),
                event_type="error",
            )

        if event_type in ("response.failed", "response.incomplete"):
            resp_obj = event.get("response", {})
            error_info = resp_obj.get("error", {})
            msg = error_info.get("message", f"Request {event_type}")
            raise SSEError(
                message=msg,
                code=error_info.get("code", ""),
                event_type=event_type,
            )

        # --- Canonical accumulation: response.output_item.done ---
        if event_type == "response.output_item.done":
            item = event.get("item", {})
            item_type = item.get("type")

            if item_type == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") in ("output_text", "text"):
                        result.content += content_part.get("text", "")

            elif item_type == "function_call":
                result.tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })

        # --- Metadata from response.created ---
        elif event_type == "response.created":
            resp = event.get("response", {})
            if not result.response_id:
                result.response_id = resp.get("id", "")
            if not result.model:
                result.model = resp.get("model", "")

        # --- Usage from response.done ---
        elif event_type == "response.done":
            resp = event.get("response", {})
            if not result.response_id:
                result.response_id = resp.get("id", "")
            usage = resp.get("usage", {})
            if usage:
                result.input_tokens = usage.get("input_tokens", 0)
                result.output_tokens = usage.get("output_tokens", 0)

        # --- Direct usage on top-level event ---
        if event.get("usage") and event_type != "response.done":
            usage = event["usage"]
            result.input_tokens = usage.get("input_tokens", result.input_tokens)
            result.output_tokens = usage.get("output_tokens", result.output_tokens)

    return result
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_sse.py -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/_sse.py tests/test_sse.py
git commit -m "feat: add SSE event parser with error detection"
```

---

### Task 6: Provider — model catalog and get_info()

**Files:**
- Create: `amplifier_module_provider_openai_chatgpt/provider.py`
- Create: `tests/test_provider.py`

**Context:** The provider needs a hardcoded model catalog (ChatGPT backend doesn't have a list-models endpoint). Model data comes from Letta's `CHATGPT_MODELS` list at `/home/robotdad/Work/openaisub/letta/letta/schemas/providers/chatgpt_oauth.py` lines 38-65.

**Step 1: Write the failing test**

Create `tests/test_provider.py`:

```python
"""Tests for ChatGPTProvider."""

from __future__ import annotations

import pytest

from amplifier_module_provider_openai_chatgpt.provider import (
    CHATGPT_MODELS,
    ChatGPTProvider,
)


class TestModelCatalog:
    """Verify hardcoded model catalog."""

    def test_catalog_is_nonempty_list(self):
        assert isinstance(CHATGPT_MODELS, list)
        assert len(CHATGPT_MODELS) > 0

    def test_each_model_has_required_fields(self):
        for model in CHATGPT_MODELS:
            assert "name" in model, f"Model missing 'name': {model}"
            assert "context_window" in model, f"Model missing 'context_window': {model}"
            assert isinstance(model["name"], str)
            assert isinstance(model["context_window"], int)
            assert model["context_window"] > 0

    def test_catalog_contains_known_models(self):
        names = [m["name"] for m in CHATGPT_MODELS]
        assert "gpt-5.4" in names
        assert "o4-mini" in names
        assert "gpt-4o" in names


class TestGetInfo:
    """Verify get_info returns correct provider metadata."""

    def test_get_info_returns_provider_info(self):
        provider = ChatGPTProvider(config={}, coordinator=None)
        info = provider.get_info()
        assert info.id == "openai-chatgpt"
        assert info.display_name == "OpenAI ChatGPT"
        assert "streaming" in info.capabilities
        assert "tools" in info.capabilities


class TestListModels:
    """Verify list_models returns ModelInfo objects."""

    @pytest.mark.asyncio(strict=True)
    async def test_list_models_returns_model_info_list(self):
        provider = ChatGPTProvider(config={}, coordinator=None)
        models = await provider.list_models()
        assert len(models) == len(CHATGPT_MODELS)
        for m in models:
            assert m.id
            assert m.display_name
            assert m.context_window > 0
            assert m.max_output_tokens > 0
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py -v
```
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Create `amplifier_module_provider_openai_chatgpt/provider.py`:

```python
"""ChatGPT subscription provider for Amplifier.

Uses raw httpx + manual SSE against the ChatGPT backend API
(chatgpt.com/backend-api/codex/responses) with OAuth device code authentication.
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
    ToolCall,
    ToolCallBlock,
    Usage,
)
from amplifier_core.utils import redact_secrets

from ._sse import ParsedResponse, SSEError, parse_sse_events
from .oauth import (
    CHATGPT_CODEX_BASE_URL,
    is_token_valid,
    load_tokens,
    refresh_tokens,
)

logger = logging.getLogger(__name__)

# ChatGPT Backend API endpoint
CHATGPT_CODEX_ENDPOINT = f"{CHATGPT_CODEX_BASE_URL}/responses"

# Hardcoded models available via ChatGPT backend
# Based on Letta's model catalog + OpenAI Codex CLI presets
CHATGPT_MODELS = [
    # GPT-5.4
    {"name": "gpt-5.4", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.4-pro", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.4-fast", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.4-mini", "context_window": 400000, "max_output": 128000},
    # GPT-5.3 codex
    {"name": "gpt-5.3-codex", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.3-codex-spark", "context_window": 128000, "max_output": 128000},
    # GPT-5.2
    {"name": "gpt-5.2", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.2-codex", "context_window": 272000, "max_output": 128000},
    # GPT-5.1
    {"name": "gpt-5.1", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.1-codex", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.1-codex-mini", "context_window": 272000, "max_output": 128000},
    {"name": "gpt-5.1-codex-max", "context_window": 272000, "max_output": 128000},
    # GPT-5 Codex
    {"name": "gpt-5-codex-mini", "context_window": 272000, "max_output": 128000},
    # GPT-4 models
    {"name": "gpt-4o", "context_window": 128000, "max_output": 16384},
    {"name": "gpt-4o-mini", "context_window": 128000, "max_output": 16384},
    # Reasoning models
    {"name": "o1", "context_window": 200000, "max_output": 100000},
    {"name": "o1-pro", "context_window": 200000, "max_output": 100000},
    {"name": "o3", "context_window": 200000, "max_output": 100000},
    {"name": "o3-mini", "context_window": 200000, "max_output": 100000},
    {"name": "o4-mini", "context_window": 200000, "max_output": 100000},
]


class ChatGPTProvider:
    """Amplifier provider for ChatGPT subscription authentication.

    Sends requests to the ChatGPT backend API using raw httpx + manual SSE.
    Does NOT use the OpenAI Python SDK (its streaming accumulator doesn't
    work against this API).
    """

    name = "openai-chatgpt"

    def __init__(
        self,
        config: dict[str, Any],
        coordinator: Any | None = None,
        tokens: dict | None = None,
    ):
        self.config = config or {}
        self.coordinator = coordinator
        self._tokens = tokens
        self.raw = self.config.get("raw", False)
        self.default_model = self.config.get("default_model", "gpt-5.4")
        self.timeout = float(self.config.get("timeout", 120.0))
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialized httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="openai-chatgpt",
            display_name="OpenAI ChatGPT",
            credential_env_vars=[],
            capabilities=["streaming", "tools"],
            defaults={
                "model": self.default_model,
                "timeout": self.timeout,
            },
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available models (hardcoded catalog)."""
        return [
            ModelInfo(
                id=m["name"],
                display_name=m["name"],
                context_window=m["context_window"],
                max_output_tokens=m["max_output"],
                capabilities=["streaming", "tools"],
            )
            for m in CHATGPT_MODELS
        ]

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from ChatResponse."""
        if not response.tool_calls:
            return []
        return [tc for tc in response.tool_calls if tc.arguments is not None]

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Generate completion via ChatGPT backend API.

        Builds request payload, POSTs with httpx streaming, parses SSE,
        and converts to Amplifier ChatResponse.
        """
        raise NotImplementedError("complete() implemented in Task 8")
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/provider.py tests/test_provider.py
git commit -m "feat: add provider skeleton with model catalog, get_info, list_models"
```

---

### Task 7: Provider — request payload construction

**Files:**
- Modify: `amplifier_module_provider_openai_chatgpt/provider.py`
- Modify: `tests/test_provider.py`

**Context:** The request payload construction is the most critical part. The ChatGPT backend rejects many standard OpenAI parameters. Must NOT send: `max_output_tokens`, `temperature`, `truncation`, `parallel_tool_calls`, `include`, or native tool types. Must always send `stream: True` and `store: False`. The `-fast` model suffix maps to `service_tier: "priority"`.

Reference: Letta's `build_request_data()` at lines 172-281 of `/home/robotdad/Work/openaisub/letta/letta/llm_api/chatgpt_oauth_client.py`.

**Step 1: Write the failing test**

Add to `tests/test_provider.py`:

```python
from amplifier_core.message_models import Message, ToolSpec


class TestBuildPayload:
    """Verify request payload construction."""

    def _make_provider(self, **overrides):
        config = {"default_model": "gpt-5.4", **overrides}
        return ChatGPTProvider(config=config, coordinator=None)

    def test_basic_payload_structure(self):
        provider = self._make_provider()
        request = ChatRequest(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ],
        )
        payload = provider._build_payload(request)

        assert payload["model"] == "gpt-5.4"
        assert payload["stream"] is True
        assert payload["store"] is False
        assert "input" in payload
        assert "instructions" in payload
        # Must NOT contain rejected params
        assert "max_output_tokens" not in payload
        assert "temperature" not in payload
        assert "truncation" not in payload
        assert "parallel_tool_calls" not in payload
        assert "include" not in payload

    def test_system_message_becomes_instructions(self):
        provider = self._make_provider()
        request = ChatRequest(
            messages=[
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hi"),
            ],
        )
        payload = provider._build_payload(request)

        assert payload["instructions"] == "Be concise."
        # System message should not appear in input
        for msg in payload["input"]:
            assert msg.get("role") != "system"

    def test_fast_suffix_maps_to_priority_service_tier(self):
        provider = self._make_provider(default_model="gpt-5.4-fast")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
        )
        payload = provider._build_payload(request)

        assert payload["model"] == "gpt-5.4"  # Suffix stripped
        assert payload["service_tier"] == "priority"

    def test_non_fast_model_has_no_service_tier(self):
        provider = self._make_provider(default_model="gpt-5.4")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
        )
        payload = provider._build_payload(request)

        assert "service_tier" not in payload

    def test_tools_converted_to_function_format(self):
        provider = self._make_provider()
        request = ChatRequest(
            messages=[Message(role="user", content="Read a file")],
            tools=[
                ToolSpec(
                    name="read_file",
                    description="Read a file from disk",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
            ],
        )
        payload = provider._build_payload(request)

        assert "tools" in payload
        assert len(payload["tools"]) == 1
        tool = payload["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "read_file"
        assert tool["description"] == "Read a file from disk"
        assert "parameters" in tool
        assert payload["tool_choice"] == "auto"

    def test_model_override_from_request(self):
        provider = self._make_provider(default_model="gpt-5.4")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="o4-mini",
        )
        payload = provider._build_payload(request)
        assert payload["model"] == "o4-mini"

    def test_user_message_with_structured_content(self):
        """Messages with TextBlock content are converted to input format."""
        from amplifier_core.message_models import TextBlock

        provider = self._make_provider()
        request = ChatRequest(
            messages=[
                Message(role="user", content=[TextBlock(text="Hello")]),
            ],
        )
        payload = provider._build_payload(request)
        assert len(payload["input"]) == 1
        msg = payload["input"][0]
        assert msg["role"] == "user"

    def test_tool_result_message_converted(self):
        """Tool result messages are converted to function_call_output format."""
        from amplifier_core.message_models import ToolResultBlock

        provider = self._make_provider()
        request = ChatRequest(
            messages=[
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_call_id="call_abc", output="file contents")],
                    tool_call_id="call_abc",
                ),
            ],
        )
        payload = provider._build_payload(request)
        assert len(payload["input"]) == 1
        msg = payload["input"][0]
        assert msg["type"] == "function_call_output"
        assert msg["call_id"] == "call_abc"
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestBuildPayload -v
```
Expected: FAIL with AttributeError (`_build_payload` doesn't exist)

**Step 3: Write the implementation**

Add to `ChatGPTProvider` in `amplifier_module_provider_openai_chatgpt/provider.py`:

```python
    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        """Build request payload for ChatGPT backend API.

        Converts Amplifier ChatRequest to the Responses API format used
        by the ChatGPT backend. Key differences from standard OpenAI API:
        - Uses ``input`` array instead of ``messages``
        - System message becomes ``instructions`` (top-level, not in input)
        - ``stream: True`` and ``store: False`` are mandatory
        - Does NOT support: max_output_tokens, temperature, truncation,
          parallel_tool_calls, include, or native tool types
        - ``-fast`` model suffix → strip suffix + add ``service_tier: "priority"``

        Args:
            request: Amplifier ChatRequest.

        Returns:
            Dict payload for the ChatGPT backend API POST body.
        """
        # Determine model (request override takes precedence)
        model = request.model or self.default_model

        # Handle "-fast" suffix → service_tier: "priority"
        service_tier = None
        if model.endswith("-fast"):
            model = model.removesuffix("-fast")
            service_tier = "priority"

        # Convert messages to Responses API input format
        instructions = None
        input_messages: list[dict[str, Any]] = []

        for msg in request.messages:
            role = msg.role

            # System message → instructions (first one only)
            if role == "system":
                if instructions is None:
                    if isinstance(msg.content, str):
                        instructions = msg.content
                    elif isinstance(msg.content, list) and msg.content:
                        # Extract text from content blocks
                        texts = []
                        for block in msg.content:
                            if hasattr(block, "text"):
                                texts.append(block.text)
                        instructions = "\n".join(texts)
                continue  # Don't include system messages in input

            # Tool result message → function_call_output
            if role == "tool":
                call_id = msg.tool_call_id or ""
                output_text = ""
                if isinstance(msg.content, str):
                    output_text = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "output"):
                            output_text = str(block.output)
                            if not call_id and hasattr(block, "tool_call_id"):
                                call_id = block.tool_call_id
                        elif hasattr(block, "text"):
                            output_text = block.text
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                })
                continue

            # Assistant message — may contain tool calls
            if role == "assistant":
                # Check for tool call blocks
                if isinstance(msg.content, list):
                    has_tool_calls = any(
                        hasattr(block, "type") and block.type == "tool_call"
                        for block in msg.content
                    )
                    if has_tool_calls:
                        for block in msg.content:
                            if hasattr(block, "type") and block.type == "tool_call":
                                input_messages.append({
                                    "type": "function_call",
                                    "call_id": block.id,
                                    "name": block.name,
                                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                                })
                            elif hasattr(block, "text"):
                                input_messages.append({
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": block.text}],
                                })
                        continue

                # Plain assistant text
                content = self._convert_content(msg.content)
                input_messages.append({"role": "assistant", "content": content})
                continue

            # User and developer messages
            api_role = "developer" if role == "developer" else "user"
            content = self._convert_content(msg.content)
            input_messages.append({"role": api_role, "content": content})

        # Build payload — NEVER include rejected params
        payload: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "stream": True,
            "store": False,
        }

        if service_tier:
            payload["service_tier"] = service_tier

        if instructions:
            payload["instructions"] = instructions

        # Convert tools to Responses API function format
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in request.tools
            ]
            # Tool choice
            if request.tool_choice:
                if isinstance(request.tool_choice, str):
                    payload["tool_choice"] = request.tool_choice
                elif isinstance(request.tool_choice, dict):
                    payload["tool_choice"] = request.tool_choice
            else:
                payload["tool_choice"] = "auto"

        # Reasoning effort for reasoning models
        if request.reasoning_effort:
            payload["reasoning"] = {
                "effort": request.reasoning_effort,
                "summary": "detailed",
            }

        return payload

    def _convert_content(
        self, content: str | list,
    ) -> list[dict[str, Any]]:
        """Convert message content to Responses API format.

        Args:
            content: String or list of content blocks.

        Returns:
            List of content dicts in Responses API format.
        """
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        result = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    result.append({"type": "input_text", "text": block.text})
                elif block.type == "thinking":
                    # Preserve thinking blocks for round-trip
                    result.append({"type": "input_text", "text": block.thinking})
                else:
                    # Fallback: try to extract text
                    if hasattr(block, "text"):
                        result.append({"type": "input_text", "text": block.text})
            elif isinstance(block, dict):
                result.append(block)
        return result or [{"type": "input_text", "text": ""}]
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/provider.py tests/test_provider.py
git commit -m "feat: add request payload construction with tool conversion"
```

---

### Task 8: Provider — headers and token management

**Files:**
- Modify: `amplifier_module_provider_openai_chatgpt/provider.py`
- Modify: `tests/test_provider.py`

**Step 1: Write the failing test**

Add to `tests/test_provider.py`:

```python
class TestBuildHeaders:
    """Verify HTTP header construction."""

    def test_headers_contain_required_fields(self):
        provider = ChatGPTProvider(
            config={},
            coordinator=None,
            tokens={
                "access_token": "test_token_abc",
                "account_id": "acct_123",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
        )
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer test_token_abc"
        assert headers["ChatGPT-Account-Id"] == "acct_123"
        assert headers["OpenAI-Beta"] == "responses=v1"
        assert headers["OpenAI-Originator"] == "codex"
        assert headers["Content-Type"] == "application/json"
        assert headers["accept"] == "text/event-stream"

    def test_missing_token_raises(self):
        provider = ChatGPTProvider(config={}, coordinator=None, tokens=None)
        with pytest.raises(ValueError, match="No valid OAuth tokens"):
            provider._build_headers()

    def test_missing_account_id_raises(self):
        provider = ChatGPTProvider(
            config={},
            coordinator=None,
            tokens={"access_token": "tok", "expires_at": "2099-01-01T00:00:00+00:00"},
        )
        with pytest.raises(ValueError, match="account_id"):
            provider._build_headers()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestBuildHeaders -v
```
Expected: FAIL

**Step 3: Write the implementation**

Add to `ChatGPTProvider` in `provider.py`:

```python
    def _build_headers(self) -> dict[str, str]:
        """Build required HTTP headers for ChatGPT backend API.

        Returns:
            Dict of HTTP headers.

        Raises:
            ValueError: If tokens are missing or incomplete.
        """
        tokens = self._tokens
        if not tokens or not tokens.get("access_token"):
            raise ValueError(
                "No valid OAuth tokens available. Run device code login first."
            )

        account_id = tokens.get("account_id")
        if not account_id:
            raise ValueError(
                "OAuth tokens missing account_id. Re-authenticate to obtain it."
            )

        return {
            "Authorization": f"Bearer {tokens['access_token']}",
            "ChatGPT-Account-Id": account_id,
            "OpenAI-Beta": "responses=v1",
            "OpenAI-Originator": "codex",
            "Content-Type": "application/json",
            "accept": "text/event-stream",
        }

    async def _ensure_valid_tokens(self) -> dict:
        """Ensure tokens are loaded and valid, refreshing if needed.

        Returns:
            Valid token dict.

        Raises:
            ValueError: If no valid tokens can be obtained.
        """
        if self._tokens and is_token_valid(self._tokens):
            return self._tokens

        # Try to load from disk
        tokens = load_tokens()
        if tokens and is_token_valid(tokens):
            self._tokens = tokens
            return tokens

        # Try to refresh
        if self._tokens and self._tokens.get("refresh_token"):
            refreshed = await refresh_tokens(self._tokens["refresh_token"])
            if refreshed:
                self._tokens = refreshed
                return refreshed

        if tokens and tokens.get("refresh_token"):
            refreshed = await refresh_tokens(tokens["refresh_token"])
            if refreshed:
                self._tokens = refreshed
                return refreshed

        raise ValueError(
            "No valid OAuth tokens. Run device code login to authenticate."
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/provider.py tests/test_provider.py
git commit -m "feat: add header construction and token management"
```

---

### Task 9: Provider — complete() with event emission

**Files:**
- Modify: `amplifier_module_provider_openai_chatgpt/provider.py`
- Modify: `tests/test_provider.py`

**Context:** The `complete()` method is the core of the provider. It must: (1) ensure valid tokens, (2) build payload, (3) emit `llm:request` event, (4) POST to the endpoint with httpx streaming, (5) parse SSE response, (6) emit `llm:response` event, (7) convert to ChatResponse. On error, emit `llm:response` with `status: "error"`.

**Step 1: Write the failing test**

Add to `tests/test_provider.py`:

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch


def _make_sse_response(lines: list[str]) -> MagicMock:
    """Create a mock httpx streaming response."""
    response = MagicMock()
    response.status_code = 200

    async def aiter_lines():
        for line in lines:
            yield line

    response.aiter_lines = aiter_lines
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


def _sse_line(event: dict) -> str:
    return f"data: {json.dumps(event)}"


class TestComplete:
    """Verify complete() sends request and returns ChatResponse."""

    def _make_provider_with_tokens(self, raw: bool = False):
        tokens = {
            "access_token": "test_token",
            "account_id": "acct_test",
            "refresh_token": "refresh_test",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider(
            config={"raw": raw},
            coordinator=coordinator,
            tokens=tokens,
        )

    @pytest.mark.asyncio(strict=True)
    async def test_simple_text_completion(self):
        provider = self._make_provider_with_tokens()
        sse_lines = [
            _sse_line({
                "type": "response.created",
                "response": {"id": "resp_1", "model": "gpt-5.4"},
            }),
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            }),
            "data: [DONE]",
        ]
        mock_response = _make_sse_response(sse_lines)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with patch.object(provider, "client", mock_client):
            request = ChatRequest(
                messages=[Message(role="user", content="Say hello")],
            )
            response = await provider.complete(request)

        assert len(response.content) >= 1
        assert response.content[0].text == "Hello!"
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 3
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio(strict=True)
    async def test_tool_call_response(self):
        provider = self._make_provider_with_tokens()
        sse_lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": '{"path": "/test.txt"}',
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            }),
            "data: [DONE]",
        ]
        mock_response = _make_sse_response(sse_lines)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with patch.object(provider, "client", mock_client):
            request = ChatRequest(
                messages=[Message(role="user", content="Read a file")],
            )
            response = await provider.complete(request)

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "read_file"
        assert tc.id == "call_abc"
        assert tc.arguments == {"path": "/test.txt"}
        assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio(strict=True)
    async def test_emits_llm_request_and_response_events(self):
        provider = self._make_provider_with_tokens()
        sse_lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
            }),
            "data: [DONE]",
        ]
        mock_response = _make_sse_response(sse_lines)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with patch.object(provider, "client", mock_client):
            request = ChatRequest(
                messages=[Message(role="user", content="Test")],
            )
            await provider.complete(request)

        # Check that llm:request and llm:response were emitted
        emit_calls = provider.coordinator.hooks.emit.call_args_list
        event_names = [call.args[0] for call in emit_calls]
        assert "llm:request" in event_names
        assert "llm:response" in event_names

        # Check llm:response payload
        response_call = next(c for c in emit_calls if c.args[0] == "llm:response")
        response_payload = response_call.args[1]
        assert response_payload["provider"] == "openai-chatgpt"
        assert response_payload["status"] == "ok"
        assert "duration_ms" in response_payload

    @pytest.mark.asyncio(strict=True)
    async def test_raw_mode_includes_raw_payloads(self):
        provider = self._make_provider_with_tokens(raw=True)
        sse_lines = [
            _sse_line({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
            }),
            _sse_line({
                "type": "response.done",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
            }),
            "data: [DONE]",
        ]
        mock_response = _make_sse_response(sse_lines)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with patch.object(provider, "client", mock_client):
            request = ChatRequest(
                messages=[Message(role="user", content="Test")],
            )
            await provider.complete(request)

        emit_calls = provider.coordinator.hooks.emit.call_args_list
        request_call = next(c for c in emit_calls if c.args[0] == "llm:request")
        assert "raw" in request_call.args[1]

        response_call = next(c for c in emit_calls if c.args[0] == "llm:response")
        assert "raw" in response_call.args[1]
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestComplete -v
```
Expected: FAIL (complete() raises NotImplementedError)

**Step 3: Write the implementation**

Replace the placeholder `complete()` in `provider.py`:

```python
    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Generate completion via ChatGPT backend API.

        Builds request payload, POSTs with httpx streaming, parses SSE,
        and converts to Amplifier ChatResponse. Emits llm:request and
        llm:response events per the Amplifier convention.

        Args:
            request: Amplifier ChatRequest.
            **kwargs: Additional provider-specific options.

        Returns:
            ChatResponse with content blocks, tool calls, and usage.

        Raises:
            ValueError: If no valid tokens are available.
            SSEError: If the API returns an error in the SSE stream.
        """
        # Ensure valid tokens
        await self._ensure_valid_tokens()

        # Build request
        payload = self._build_payload(request)
        headers = self._build_headers()

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_event: dict[str, Any] = {
                "provider": self.name,
                "model": payload["model"],
                "message_count": len(request.messages),
            }
            if self.raw:
                request_event["raw"] = redact_secrets(payload)
            await self.coordinator.hooks.emit("llm:request", request_event)

        start_time = time.time()

        try:
            # Stream SSE response
            sse_lines: list[str] = []
            async with self.client.stream(
                "POST",
                CHATGPT_CODEX_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise ValueError(
                        f"ChatGPT API error ({response.status_code}): "
                        f"{error_body.decode('utf-8', errors='replace')[:500]}"
                    )
                async for line in response.aiter_lines():
                    sse_lines.append(line)

            # Parse SSE events
            parsed = parse_sse_events(sse_lines)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Emit llm:response event (success)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                response_event: dict[str, Any] = {
                    "provider": self.name,
                    "model": payload["model"],
                    "usage": {
                        "input": parsed.input_tokens,
                        "output": parsed.output_tokens,
                    },
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                if self.raw:
                    response_event["raw"] = redact_secrets({
                        "events": parsed.raw_events,
                        "response_id": parsed.response_id,
                    })
                await self.coordinator.hooks.emit("llm:response", response_event)

            # Convert to ChatResponse
            return self._to_chat_response(parsed, payload["model"])

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] ChatGPT API error: %s", error_msg)

            # Emit llm:response event (error)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": payload.get("model", self.default_model),
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise

    def _to_chat_response(self, parsed: ParsedResponse, model: str) -> ChatResponse:
        """Convert ParsedResponse to Amplifier ChatResponse.

        Args:
            parsed: Accumulated SSE response.
            model: Model name used for the request.

        Returns:
            ChatResponse with content blocks, tool calls, and usage.
        """
        content_blocks = []

        # Text content
        if parsed.content:
            content_blocks.append(TextBlock(text=parsed.content))

        # Tool call content blocks
        tool_calls = []
        for tc in parsed.tool_calls:
            func = tc.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"_raw": func.get("arguments", "")}

            call_id = tc.get("id", "")
            name = func.get("name", "")

            content_blocks.append(
                ToolCallBlock(id=call_id, name=name, input=arguments)
            )
            tool_calls.append(
                ToolCall(id=call_id, name=name, arguments=arguments)
            )

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else "stop"

        # Usage
        usage = Usage(
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            total_tokens=parsed.input_tokens + parsed.output_tokens,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=finish_reason,
            metadata={
                "response_id": parsed.response_id,
                "model": model,
            },
        )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/provider.py tests/test_provider.py
git commit -m "feat: implement complete() with SSE streaming and event emission"
```

---

### Task 10: Provider — error handling in complete()

**Files:**
- Modify: `tests/test_provider.py`

**Step 1: Write error handling tests**

Add to `tests/test_provider.py`:

```python
from amplifier_module_provider_openai_chatgpt._sse import SSEError


class TestCompleteErrors:
    """Verify error handling in complete()."""

    def _make_provider_with_tokens(self):
        tokens = {
            "access_token": "test_token",
            "account_id": "acct_test",
            "refresh_token": "refresh_test",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider(
            config={},
            coordinator=coordinator,
            tokens=tokens,
        )

    @pytest.mark.asyncio(strict=True)
    async def test_sse_error_event_emits_error_response(self):
        """SSE error events emit llm:response with status=error."""
        provider = self._make_provider_with_tokens()
        sse_lines = [
            _sse_line({
                "type": "error",
                "error": {"message": "context too long", "code": "invalid_request"},
            }),
        ]
        mock_response = _make_sse_response(sse_lines)
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with patch.object(provider, "client", mock_client):
            request = ChatRequest(
                messages=[Message(role="user", content="Test")],
            )
            with pytest.raises(SSEError, match="context too long"):
                await provider.complete(request)

        # Verify error event was emitted
        emit_calls = provider.coordinator.hooks.emit.call_args_list
        response_calls = [c for c in emit_calls if c.args[0] == "llm:response"]
        assert len(response_calls) == 1
        assert response_calls[0].args[1]["status"] == "error"

    @pytest.mark.asyncio(strict=True)
    async def test_no_tokens_raises_value_error(self):
        """Missing tokens raise ValueError."""
        provider = ChatGPTProvider(config={}, coordinator=None, tokens=None)

        with patch.object(
            provider, "_ensure_valid_tokens",
            side_effect=ValueError("No valid OAuth tokens"),
        ):
            request = ChatRequest(
                messages=[Message(role="user", content="Test")],
            )
            with pytest.raises(ValueError, match="No valid OAuth tokens"):
                await provider.complete(request)
```

**Step 2: Run tests**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestCompleteErrors -v
```
Expected: All PASS (error handling is already in complete() from Task 9)

**Step 3: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add tests/test_provider.py
git commit -m "test: add error handling tests for complete()"
```

---

### Task 11: Mount function — wire everything together

**Files:**
- Modify: `amplifier_module_provider_openai_chatgpt/__init__.py`
- Modify: `tests/test_provider.py` (add mount tests)

**Step 1: Write the failing test**

Add to `tests/test_provider.py`:

```python
from amplifier_module_provider_openai_chatgpt import mount


class TestMount:
    """Verify mount() wires the provider correctly."""

    @pytest.mark.asyncio(strict=True)
    async def test_mount_with_tokens_file(self, tmp_path):
        """Mount succeeds when a token file exists with valid tokens."""
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens
        from datetime import datetime, timedelta, timezone

        token_path = str(tmp_path / "tokens.json")
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        save_tokens(
            {
                "access_token": "test_tok",
                "refresh_token": "test_ref",
                "account_id": "acct_mount",
                "expires_at": expires_at,
            },
            token_path,
        )

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        cleanup = await mount(
            coordinator,
            config={"token_file_path": token_path},
        )

        # Provider should have been mounted
        coordinator.mount.assert_called_once()
        call_args = coordinator.mount.call_args
        assert call_args.args[0] == "providers"
        assert call_args.kwargs.get("name") == "openai-chatgpt"

        # Cleanup should be callable
        assert callable(cleanup)

    @pytest.mark.asyncio(strict=True)
    async def test_mount_returns_none_when_no_tokens(self, tmp_path):
        """Mount returns None gracefully when no tokens are available."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        result = await mount(
            coordinator,
            config={"token_file_path": str(tmp_path / "nonexistent.json"), "login_on_mount": False},
        )

        assert result is None
        coordinator.mount.assert_not_called()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestMount -v
```
Expected: FAIL (mount raises NotImplementedError)

**Step 3: Write the implementation**

Replace the content of `amplifier_module_provider_openai_chatgpt/__init__.py`:

```python
"""Amplifier ChatGPT subscription auth provider module.

Uses raw httpx + manual SSE against the ChatGPT backend API
(chatgpt.com/backend-api/codex/responses) with OAuth device code authentication.
"""

from __future__ import annotations

__all__ = ["mount", "ChatGPTProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import logging
from typing import TYPE_CHECKING, Any

from .oauth import is_token_valid, load_tokens, login
from .provider import ChatGPTProvider

if TYPE_CHECKING:
    from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)


async def mount(
    coordinator: ModuleCoordinator,
    config: dict[str, Any] | None = None,
) -> Any:
    """Mount the ChatGPT subscription provider.

    Loads OAuth tokens from disk. If no valid tokens exist and
    ``login_on_mount`` is True (default), starts the device code flow
    interactively. If no tokens can be obtained, returns None for
    graceful degradation.

    Args:
        coordinator: Amplifier module coordinator.
        config: Provider configuration dict. Supported keys:
            - ``token_file_path``: Override path for token storage.
            - ``login_on_mount``: If True (default), run device code login
              when no valid tokens exist. Set False to skip.
            - ``raw``: If True, include raw payloads in llm:request/llm:response events.
            - ``default_model``: Default model ID (default: ``gpt-5.4``).
            - ``timeout``: HTTP timeout in seconds (default: 120).

    Returns:
        Async cleanup callable, or None if provider cannot be mounted.
    """
    config = config or {}
    token_file_path = config.get("token_file_path")
    login_on_mount = config.get("login_on_mount", True)

    # Try to load existing tokens
    tokens = load_tokens(token_file_path)

    if not tokens or not is_token_valid(tokens):
        if login_on_mount:
            logger.info("No valid ChatGPT OAuth tokens — starting device code login")
            try:
                tokens = await login(token_file_path=token_file_path)
            except Exception as e:
                logger.warning("ChatGPT OAuth login failed: %s", e)
                return None
        else:
            logger.warning(
                "No valid ChatGPT OAuth tokens and login_on_mount=False — "
                "provider not mounted"
            )
            return None

    if not tokens or not is_token_valid(tokens):
        logger.warning("Could not obtain valid ChatGPT OAuth tokens — provider not mounted")
        return None

    provider = ChatGPTProvider(
        config=config,
        coordinator=coordinator,
        tokens=tokens,
    )
    await coordinator.mount("providers", provider, name="openai-chatgpt")
    logger.info("Mounted ChatGPTProvider (model: %s)", provider.default_model)

    async def cleanup():
        await provider.close()

    return cleanup
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/test_provider.py::TestMount -v
```
Expected: All PASS

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add amplifier_module_provider_openai_chatgpt/__init__.py tests/test_provider.py
git commit -m "feat: implement mount() with token loading and device code login"
```

---

### Task 12: Run full test suite and fix any issues

**Files:**
- All test files

**Step 1: Install dependencies**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv sync
```

**Step 2: Run full test suite**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run pytest tests/ -v
```
Expected: All PASS

**Step 3: Run linting**

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run ruff check amplifier_module_provider_openai_chatgpt/ tests/
uv run ruff format --check amplifier_module_provider_openai_chatgpt/ tests/
```

**Step 4: Fix any lint or format issues**

If `ruff` reports issues, fix them. Common ones:
- Import ordering
- Unused imports
- Line length
- Missing `from __future__ import annotations`

```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
uv run ruff check --fix amplifier_module_provider_openai_chatgpt/ tests/
uv run ruff format amplifier_module_provider_openai_chatgpt/ tests/
```

**Step 5: Commit**
```bash
cd /home/robotdad/Work/openaisub/amplifier-module-provider-openai-chatgpt
git add -A
git commit -m "chore: lint fixes and full test suite passing"
```

---

## File Summary

After all tasks, the repo should contain:

```
amplifier_module_provider_openai_chatgpt/
  __init__.py        # mount(), cleanup, imports
  oauth.py           # Device code flow, token refresh, PKCE, token persistence
  _sse.py            # SSE line parser → ParsedResponse
  provider.py        # ChatGPTProvider: complete(), list_models(), get_info(),
                     #   _build_payload(), _build_headers(), _to_chat_response(),
                     #   parse_tool_calls()
tests/
  __init__.py
  test_oauth.py      # Constants, PKCE, token storage/validation, JWT, refresh,
                     #   device code flow
  test_sse.py        # SSE parsing: text, tools, mixed, errors, malformed JSON
  test_provider.py   # Model catalog, get_info, list_models, payload construction,
                     #   headers, complete(), error handling, mount()
```

## Key Constraints Checklist

- [x] Raw httpx, NOT the OpenAI SDK
- [x] Streaming is mandatory (`stream: True` always)
- [x] Rejected params never sent (`max_output_tokens`, `temperature`, `truncation`, `parallel_tool_calls`, `include`)
- [x] Handles both `output_text` and `text` content types
- [x] `response.output_item.done` is canonical accumulation event
- [x] Error events detected inside 200 streams (`error`, `response.failed`, `response.incomplete`)
- [x] `-fast` suffix → strip + `service_tier: "priority"`
- [x] `raw: bool` config flag with `llm:request`/`llm:response` emission + `redact_secrets()`
- [x] Emission guard: `if self.coordinator and hasattr(self.coordinator, "hooks")`
- [x] All 5 Provider protocol methods implemented: `name`, `get_info()`, `list_models()`, `complete()`, `parse_tool_calls()`
- [x] `mount()` with graceful degradation (returns None on missing tokens)
- [x] OAuth device code flow for interactive authentication
- [x] Token refresh and local file persistence
