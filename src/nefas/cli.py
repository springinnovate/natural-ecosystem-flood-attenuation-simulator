from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ConfigError
from .logging import configure_logging
from .runner import prepare_run

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a NEFAS flood simulation run.")
    parser.add_argument("config", type=Path, help="Path to the simulation YAML file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    try:
        prepare_run(args.config)
    except (ConfigError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Configuration is valid. No simulation was run.")
    return 0
