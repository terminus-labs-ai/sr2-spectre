"""CLI entry point for sr2-spectre.

Usage:
    sr2-spectre config.yaml --plugin single_shot "What is the weather?"
    sr2-spectre config.yaml --plugin tui
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOG", "ERROR")

from sr2_spectre.agent import Agent
from sr2_spectre.config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_LOG_FILE = Path.home() / ".sr2-spectre" / "spectre.log"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sr2-spectre",
        description="SR2 Spectre — full agent runtime powered by SR2",
    )
    parser.add_argument(
        "config",
        help="Path to YAML config file (required)",
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
        "--log-file",
        default=str(_DEFAULT_LOG_FILE),
        help=f"Log file path (default: {_DEFAULT_LOG_FILE})",
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
        "single_shot": "sr2_spectre.plugins.single_shot.SingleShotPlugin",
        "tui": "sr2_spectre.plugins.tui.TUIPlugin",
    }

    class_path = known_plugins.get(plugin_name, plugin_name)
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _configure_logging(level: str, log_file: str) -> None:
    os.environ.setdefault("LITELLM_LOG", "WARNING")

    numeric = getattr(logging, level)
    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(numeric)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logging.basicConfig(level=numeric, handlers=[file_handler, console_handler], force=True)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


async def run_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level, args.log_file)

    logger.info("SR2 Spectre starting")

    config = load_config(args.config)
    logger.info(
        "Agent: %s | model: %s",
        config.agent.name,
        config.models.get("default", {}).model if hasattr(config.models.get("default", {}), "model") else "unknown",
    )

    agent = Agent(
        config=config,
        session_id=args.session_id,
    )

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
