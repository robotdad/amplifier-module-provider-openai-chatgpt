"""Tests for oauth.py constants and structure."""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import HTTPError


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_jwt(payload_dict: dict) -> str:
    """Create an unsigned JWT string (header.payload.signature) for testing.

    The signature part is left empty — this mirrors how unsigned tokens are
    structured and is sufficient for decode-only tests.
    """
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps(payload_dict).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


def _mock_urlopen_response(body_dict: dict) -> MagicMock:
    """Return a mock urlopen callable that yields the given JSON response body.

    Usage::

        mock_urlopen = _mock_urlopen_response({"access_token": "tok"})
        with patch("...urlopen", mock_urlopen):
            ...
    """
    body = json.dumps(body_dict).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = body
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    return MagicMock(return_value=mock_response)


class TestConstants:
    """Verify all OAuth constant values."""

    def test_oauth_issuer(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_ISSUER

        assert OAUTH_ISSUER == "https://auth.openai.com"

    def test_oauth_token_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_TOKEN_URL

        assert OAUTH_TOKEN_URL == "https://auth.openai.com/oauth/token"

    def test_oauth_client_id(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_CLIENT_ID

        assert OAUTH_CLIENT_ID == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_oauth_scopes(self):
        from amplifier_module_provider_openai_chatgpt.oauth import OAUTH_SCOPES

        assert OAUTH_SCOPES == "openid profile email offline_access"

    def test_device_code_usercode_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_USERCODE_URL,
        )

        assert (
            DEVICE_CODE_USERCODE_URL
            == "https://auth.openai.com/api/accounts/deviceauth/usercode"
        )

    def test_device_code_token_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import DEVICE_CODE_TOKEN_URL

        assert (
            DEVICE_CODE_TOKEN_URL
            == "https://auth.openai.com/api/accounts/deviceauth/token"
        )

    def test_device_code_verification_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_VERIFICATION_URL,
        )

        assert DEVICE_CODE_VERIFICATION_URL == "https://auth.openai.com/codex/device"

    def test_device_code_poll_interval(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_POLL_INTERVAL,
        )

        assert DEVICE_CODE_POLL_INTERVAL == 5

    def test_chatgpt_codex_base_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            CHATGPT_CODEX_BASE_URL,
        )

        assert CHATGPT_CODEX_BASE_URL == "https://chatgpt.com/backend-api/codex"

    def test_token_file_path(self):
        from amplifier_module_provider_openai_chatgpt.oauth import TOKEN_FILE_PATH

        assert TOKEN_FILE_PATH == "~/.amplifier/openai-chatgpt-oauth.json"

    def test_device_code_callback_url(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            DEVICE_CODE_CALLBACK_URL,
        )

        assert DEVICE_CODE_CALLBACK_URL == "https://auth.openai.com/deviceauth/callback"

    def test_no_oauth_authorize_url(self):
        """OAUTH_AUTHORIZE_URL must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_AUTHORIZE_URL"), (
            "OAUTH_AUTHORIZE_URL should have been removed"
        )

    def test_no_oauth_callback_port(self):
        """OAUTH_CALLBACK_PORT must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_CALLBACK_PORT"), (
            "OAUTH_CALLBACK_PORT should have been removed"
        )

    def test_no_oauth_callback_url(self):
        """OAUTH_CALLBACK_URL must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "OAUTH_CALLBACK_URL"), (
            "OAUTH_CALLBACK_URL should have been removed"
        )

    def test_no_subscription_models(self):
        """SUBSCRIPTION_MODELS must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "SUBSCRIPTION_MODELS"), (
            "SUBSCRIPTION_MODELS should have been removed"
        )

    def test_no_start_browser_flow(self):
        """start_browser_flow() must not exist (removed from chatgpt variant)."""
        import amplifier_module_provider_openai_chatgpt.oauth as oauth_module

        assert not hasattr(oauth_module, "start_browser_flow"), (
            "start_browser_flow should have been removed"
        )


class TestPKCE:
    """Verify generate_pkce_pair() behavior per RFC 7636."""

    def test_returns_tuple_of_two_strings(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        result = generate_pkce_pair()
        assert isinstance(result, tuple)
        assert len(result) == 2
        verifier, challenge = result
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, _ = generate_pkce_pair()
        assert 43 <= len(verifier) <= 128

    def test_verifier_url_safe_characters(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, _ = generate_pkce_pair()
        assert re.fullmatch(r"[A-Za-z0-9\-._~]+", verifier), (
            f"Verifier contains non-URL-safe characters: {verifier!r}"
        )

    def test_challenge_is_sha256_of_verifier(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        verifier, challenge = generate_pkce_pair()
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_each_call_returns_unique_pair(self):
        from amplifier_module_provider_openai_chatgpt.oauth import generate_pkce_pair

        pair1 = generate_pkce_pair()
        pair2 = generate_pkce_pair()
        assert pair1[0] != pair2[0], "verifiers should be unique per call"
        assert pair1[1] != pair2[1], "challenges should be unique per call"


class TestSaveTokens:
    """Verify save_tokens() file-writing behavior."""

    def test_creates_file_with_correct_json_content(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc", "expires_at": "2099-01-01T00:00:00+00:00"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path=path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == tokens

    def test_file_has_0600_permissions(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc"}
        path = str(tmp_path / "tokens.json")
        save_tokens(tokens, path=path)
        mode = os.stat(path).st_mode & 0o777
        assert mode == 0o600

    def test_creates_parent_directories_if_missing(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import save_tokens

        tokens = {"access_token": "abc"}
        path = str(tmp_path / "nested" / "deep" / "tokens.json")
        save_tokens(tokens, path=path)
        assert os.path.isfile(path)


class TestLoadTokens:
    """Verify load_tokens() file-reading behavior."""

    def test_returns_dict_for_valid_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        tokens = {"access_token": "abc", "expires_at": "2099-01-01T00:00:00+00:00"}
        path = str(tmp_path / "tokens.json")
        with open(path, "w") as f:
            json.dump(tokens, f)
        result = load_tokens(path=path)
        assert result == tokens

    def test_returns_none_for_missing_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "nonexistent.json")
        result = load_tokens(path=path)
        assert result is None

    def test_returns_none_for_malformed_json(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        result = load_tokens(path=path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import load_tokens

        path = str(tmp_path / "empty.json")
        with open(path, "w") as f:
            f.write("")
        result = load_tokens(path=path)
        assert result is None


class TestIsTokenValid:
    """Verify is_token_valid() token validation behavior."""

    def test_valid_future_token_returns_true(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {"access_token": "abc", "expires_at": future}
        assert is_token_valid(tokens) is True

    def test_expired_token_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        past = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        tokens = {"access_token": "abc", "expires_at": past}
        assert is_token_valid(tokens) is False

    def test_none_tokens_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        assert is_token_valid(None) is False

    def test_missing_access_token_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {"expires_at": future}
        assert is_token_valid(tokens) is False

    def test_missing_expires_at_returns_false(self):
        from amplifier_module_provider_openai_chatgpt.oauth import is_token_valid

        tokens = {"access_token": "abc"}
        assert is_token_valid(tokens) is False


class TestExtractAccountId:
    """Verify extract_account_id() JWT decoding behavior."""

    def test_extracts_account_id_from_profile_claim(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        jwt = _make_jwt({"https://api.openai.com/profile": {"account_id": "acc_123"}})
        assert extract_account_id(jwt) == "acc_123"

    def test_falls_back_to_sub_claim(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        jwt = _make_jwt({"sub": "sub_456"})
        assert extract_account_id(jwt) == "sub_456"

    def test_returns_empty_string_for_invalid_jwt(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        assert extract_account_id("not_a_valid_jwt") == ""

    def test_returns_empty_string_for_empty_string(self):
        from amplifier_module_provider_openai_chatgpt.oauth import extract_account_id

        assert extract_account_id("") == ""


class TestRefreshTokens:
    """Verify refresh_tokens() HTTP behavior."""

    def test_successful_refresh_returns_new_tokens_with_account_id(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        # Write existing tokens so that refresh_tokens can preserve account_id.
        token_path = str(tmp_path / "tokens.json")
        with open(token_path, "w") as f:
            json.dump({"account_id": "existing_acc", "access_token": "old"}, f)

        mock_urlopen = _mock_urlopen_response(
            {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            }
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen", mock_urlopen
        ):
            result = asyncio.run(refresh_tokens("test_refresh", path=token_path))

        assert result is not None
        assert result["auth_mode"] == "oauth"
        assert result["access_token"] == "new_access"
        assert result["account_id"] == "existing_acc"

    def test_refresh_failure_http_401_returns_none(self, tmp_path):
        from amplifier_module_provider_openai_chatgpt.oauth import refresh_tokens

        mock_urlopen = MagicMock(
            side_effect=HTTPError(
                url="https://auth.openai.com/oauth/token",
                code=401,
                msg="Unauthorized",
                hdrs={},  # type: ignore[arg-type]
                fp=io.BytesIO(b"Unauthorized"),
            )
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.urlopen", mock_urlopen
        ):
            result = asyncio.run(
                refresh_tokens("bad_refresh", path=str(tmp_path / "tokens.json"))
            )

        assert result is None


def _make_http_error(body_dict: dict, code: int = 403) -> HTTPError:
    """Create an HTTPError with a JSON body for testing.

    Device code polling pending/expired responses arrive as HTTP errors
    (403/404), NOT as 200 OK responses.  Use this helper to build those
    errors in tests.
    """
    body = json.dumps(body_dict).encode("utf-8")
    return HTTPError(
        url="https://auth.openai.com/api/accounts/deviceauth/token",
        code=code,
        msg="Error",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


def _make_urlopen_cm(body_dict: dict) -> MagicMock:
    """Create a context-manager mock that urlopen returns for a successful call."""
    body = json.dumps(body_dict).encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.read.return_value = body
    mock_cm.__enter__.return_value = mock_cm
    mock_cm.__exit__.return_value = False
    return mock_cm


class TestDeviceCodeFlow:
    """Verify start_device_code_flow() behavior."""

    def test_requests_device_code_and_returns_auth_code(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        usercode_cm = _make_urlopen_cm(
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5}
        )
        poll_success_cm = _make_urlopen_cm({"authorization_code": "auth_code_xyz"})
        mock_urlopen = MagicMock(side_effect=[usercode_cm, poll_success_cm])

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.urlopen", mock_urlopen
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(start_device_code_flow())

        assert "authorization_code" in result
        assert result["authorization_code"] == "auth_code_xyz"
        assert "code_verifier" in result

    def test_handles_authorization_pending_then_success(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        usercode_cm = _make_urlopen_cm(
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5}
        )
        poll_success_cm = _make_urlopen_cm({"authorization_code": "auth_code_xyz"})
        mock_urlopen = MagicMock(
            side_effect=[
                usercode_cm,
                _make_http_error(
                    {"error": {"code": "deviceauth_authorization_unknown"}}
                ),
                poll_success_cm,
            ]
        )
        mock_sleep = AsyncMock()

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.urlopen", mock_urlopen
            ),
            patch("asyncio.sleep", mock_sleep),
        ):
            result = asyncio.run(start_device_code_flow())

        assert result["authorization_code"] == "auth_code_xyz"
        assert mock_sleep.call_count == 2

    def test_expired_device_code_raises(self):
        from amplifier_module_provider_openai_chatgpt.oauth import (
            start_device_code_flow,
        )

        usercode_cm = _make_urlopen_cm(
            {"user_code": "ABC-123", "device_auth_id": "dev_001", "interval": 5}
        )
        mock_urlopen = MagicMock(
            side_effect=[
                usercode_cm,
                _make_http_error({"error": {"code": "deviceauth_expired"}}),
            ]
        )

        with (
            patch(
                "amplifier_module_provider_openai_chatgpt.oauth.urlopen", mock_urlopen
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="Device code expired"):
                asyncio.run(start_device_code_flow())
