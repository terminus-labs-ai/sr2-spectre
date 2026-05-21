"""Relay client — thin wrapper around RelayLLMCallable."""
from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from sr2.protocols.llm import CompletionRequest, CompletionResponse

__all__ = ["RelayClient"]


class RelayClient:
    """Wraps RelayLLMCallable for spectre's agent loop.

    Lazy import avoids requiring sr2-relay at spectre import time.
    """

    def __init__(self, model: str, base_url: str) -> None:
        self._model = model
        self._base_url = base_url
        self._llm: Any | None = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from sr2_relay.llm import RelayLLMCallable
            self._llm = RelayLLMCallable(model=self._model, base_url=self._base_url)
        return self._llm

    async def complete(self, request: Any) -> Any:
        """Send a completion request to relay."""
        return await self._get_llm().complete(request)

    async def stream(self, request: Any) -> AsyncIterator:
        """Stream completion events from relay."""
        return self._get_llm().stream(request)
