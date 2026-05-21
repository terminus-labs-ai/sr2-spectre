"""CLI entry point for sr2-spectre.

Usage:
    sr2-spectre --plugin single_shot "What is the weather?"
    sr2-spectre --plugin tui
    sr2-spectre config.yaml --plugin tui
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
from typing import Any

from sr2_spectre.agent import Agent
from sr2_spectre.config import AgentConfig, SpectreConfig, load_config

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sr2-spectre",
        description="SR2 Spectre — full agent runtime powered by sr2-relay",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="single_shot",
        help="Plugin to run: single_shot, tui (default: single_shot)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Override session ID",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text (for single_shot mode)",
    )
    return parser.parse_args(argv)


def _load_plugin(plugin_name: str, **kwargs: Any) -> Any:
    """Load a plugin class by name."""
    known_plugins = {
        "single_shot": "sr2_spectre.plugins.single_shot:SingleShotPlugin",
        "tui": "sr2_spectre.plugins.tui:TUIPlugin",
    }

    class_path = known_plugins.get(plugin_name)
    if class_path is None:
        # Try direct import (class_path from config)
        class_path = plugin_name

    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


async def run_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )

    logger.info("SR2 Spectre starting")

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = SpectreConfig(agent=AgentConfig())

    agent_config = config.agent
    logger.info(
        f"Agent: {agent_config.name}, model: {agent_config.model}, "
        f"relay: {agent_config.relay_base_url}"
    )

    # Create agent
    agent = Agent(
        config=agent_config,
        session_id=args.session_id,
    )

    # Load and run plugin
    plugin_kwargs: dict[str, Any] = {}
    if args.plugin == "single_shot" and args.prompt:
        plugin_kwargs["prompt"] = " ".join(args.prompt)

    plugin = _load_plugin(args.plugin, **plugin_kwargs)

    await plugin.start(agent)
    await plugin.run(agent)
    await plugin.stop()

    logger.info("SR2 Spectre shutdown")


def main() -> None:
    asyncio.run(run_async())


if __name__ == "__main__":
    main()
