#!/usr/bin/env python3
"""Main entry point for game automation framework."""

import sys
import argparse
import logging
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_global_config, load_game_config
from utils.logger import setup_logger
from core.ai_client import GeminiVisionClient, LocalVisionClient
from core.game_loop import GameLoop


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered WeChat game automation"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="endless_winter",
        help="Game to play (default: endless_winter)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum number of rounds (overrides config)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Interval between rounds (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute actions, just analyze",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        default=True,
        help="Use local model instead of Gemini API (default: True)",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini API instead of local model",
    )
    parser.add_argument(
        "--local-api-url",
        type=str,
        default="http://192.168.1.156:1234",
        help="URL of local API server (default: http://192.168.1.156:1234)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start live web dashboard at http://127.0.0.1:8765/",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
    )

    args = parser.parse_args()

    use_local = args.use_local and not args.use_gemini

    setup_logger(args.log_level)
    logger = logging.getLogger(__name__)

    if args.dashboard:
        from utils.dashboard_server import start as _dash_start
        _dash_start(host="127.0.0.1", port=args.dashboard_port, project_root=Path(__file__).parent)

    logger.info(f"Starting autoplay with game: {args.game}")

    try:
        global_config = load_global_config()
        game_config = load_game_config(args.game)

        if use_local:
            logger.info(f"Using local model at {args.local_api_url}")
            ai_client = LocalVisionClient(
                api_base=args.local_api_url,
                model="qwen/qwen3-vl-8b",
            )
        else:
            api_key = global_config.gemini.api_key
            if not api_key:
                logger.error(
                    "Gemini API key not found. Please set GEMINI_API_KEY environment variable"
                )
                sys.exit(1)

            logger.info("Using Gemini API")
            ai_client = GeminiVisionClient(
                api_key=api_key,
                model=global_config.gemini.model,
            )

        game_module_name = f"games.{args.game}.game"
        try:
            game_module = importlib.import_module(game_module_name)
        except ImportError as e:
            logger.error(f"Failed to import game module: {game_module_name}")
            logger.error(f"Error: {e}")
            sys.exit(1)

        game_class_name = "".join(
            word.capitalize() for word in args.game.split("_")
        ) + "Game"

        if not hasattr(game_module, game_class_name):
            logger.error(f"Game class {game_class_name} not found in {game_module_name}")
            sys.exit(1)

        GameClass = getattr(game_module, game_class_name)
        game_cfg_dict = game_config.model_dump()
        if args.use_gemini:
            s1 = game_cfg_dict.setdefault("system1", {})
            if s1.get("enabled", True):
                logger.info("--use-gemini: disabling System 1 (local Qwen) for this run")
                s1["enabled"] = False
        try:
            game = GameClass(game_cfg_dict, ai_client=ai_client)
        except TypeError:
            game = GameClass(game_cfg_dict)

        loop_interval = args.interval or global_config.loop.interval
        max_rounds = args.max_rounds or global_config.loop.max_rounds

        game_loop = GameLoop(
            game=game,
            ai_client=ai_client,
            loop_interval=loop_interval,
            max_rounds=max_rounds,
            dry_run=args.dry_run,
        )

        game_loop.run()

        if args.dashboard:
            logger.info(
                f"Game loop ended. Dashboard still serving at http://127.0.0.1:{args.dashboard_port}/ — press Ctrl+C to exit."
            )
            try:
                import time
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                logger.info("Dashboard shutting down.")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
