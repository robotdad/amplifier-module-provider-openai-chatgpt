"""Tests for oauth.py constants and structure."""


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
