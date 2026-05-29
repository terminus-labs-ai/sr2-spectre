"""CLI entry point for sr2-spectre.

Usage:
    sr2-spectre config.yaml --plugin single_shot "What is the weather?"
    sr2-spectre config.yaml --plugin tui
    sr2-spectre config show [--no-provenance] [--sr2-home PATH]
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

from sr2.pipeline.tracing import CollectingTracer, render_compiled_request, render_trace
from sr2_spectre.agent import Agent
from sr2_spectre.config import load_config

logger = logging.getLogger(__name__)

_DEFAULT_LOG_FILE = Path.home() / ".sr2-spectre" / "spectre.log"


def _parse_config_show_args(argv: list[str]) -> argparse.Namespace:
    """Parse arguments for 'spectre config show' subcommand."""
    parser = argparse.ArgumentParser(
        prog="sr2-spectre config show",
        description="Show the resolved configuration with provenance annotations.",
        add_help=True,
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        default=False,
        help="Output plain YAML without source annotations",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        default=False,
        help="Include raw file content in the output (reserved)",
    )
    parser.add_argument(
        "--sr2-home",
        type=str,
        default=None,
        help="Override SR2_HOME directory",
    )
    return parser.parse_args(argv)


def _run_config_show(argv: list[str]) -> int:
    """Execute the 'config show' dry-run command. Returns exit code (0 or 1)."""
    from sr2_spectre.config import (
        format_dry_run,
        load_config_with_provenance,
        validate_config,
    )

    args = _parse_config_show_args(argv)

    env = dict(os.environ)
    if args.sr2_home:
        env["SR2_HOME"] = args.sr2_home

    try:
        config, provenance = load_config_with_provenance(cwd=Path.cwd(), env=env)
    except Exception as exc:
        print(f"# Error loading config: {exc}", file=sys.stdout)
        return 1

    errors = validate_config(config)

    output = format_dry_run(
        config=config,
        provenance=provenance,
        errors=errors,
        include_content=args.include_content,
        show_provenance=not args.no_provenance,
    )
    print(output, end="")

    return 1 if errors else 0


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
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Print pipeline firing timeline after reply",
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

    tracer = CollectingTracer() if args.trace else None

    if tracer is not None:
        agent = Agent(
            config=config,
            session_id=args.session_id,
            tracer=tracer,
        )
    else:
        agent = Agent(
            config=config,
            session_id=args.session_id,
        )

    plugin_kwargs: dict[str, Any] = {}
    if args.plugin == "single_shot" and args.prompt:
        plugin_kwargs["prompt"] = " ".join(args.prompt)

    await agent.initialize()

    plugin = _load_plugin(args.plugin, **plugin_kwargs)

    await plugin.start(agent)
    await plugin.run(agent)
    await plugin.stop()

    if tracer is not None:
        print(render_trace(tracer.get_trace()))
        if tracer.compiled_request is not None:
            print(render_compiled_request(tracer.compiled_request))

    logger.info("SR2 Spectre shutdown")


def main(argv: list[str] | None = None) -> None:
    """Entry point. Dispatches 'config show' before agent startup."""
    args = argv if argv is not None else sys.argv[1:]

    # Dispatch 'config show' subcommand before agent startup
    if len(args) >= 2 and args[0] == "config" and args[1] == "show":
        exit_code = _run_config_show(args[2:])
        sys.exit(exit_code)

    asyncio.run(run_async(argv))


if __name__ == "__main__":
    main()
