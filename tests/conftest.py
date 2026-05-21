"""Pytest configuration for sr2-spectre."""
import pytest


@pytest.fixture
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
