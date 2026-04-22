"""Tests for oauth.py constants and structure."""

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone


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
