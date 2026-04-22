"""Amplifier ChatGPT subscription auth provider module.

Uses raw httpx + manual SSE against the ChatGPT backend API
(chatgpt.com/backend-api/codex/responses) with OAuth device code authentication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amplifier_core import Coordinator


async def mount(coordinator: Coordinator, config: dict | None = None) -> None:
    """Mount the ChatGPT subscription provider."""
    raise NotImplementedError("Provider not yet implemented")
