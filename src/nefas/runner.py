from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import SimulationConfig, load_config
from .preprocessing import prepare_intermediates

LOGGER = logging.getLogger(__name__)


def prepare_run(config_path: Path) -> SimulationConfig:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for progress reporting.") from exc

    steps = (
        "load configuration",
        "prepare output directory",
        "prepare geospatial intermediates",
    )
    with tqdm(steps, desc="Preparing run", unit="step", disable=not sys.stderr.isatty()) as progress:
        config = load_config(config_path)
        LOGGER.info("Loaded configuration from %s", config_path)
        progress.update()

        config.output.directory.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Output directory is ready at %s", config.output.directory)
        progress.update()

        intermediates = prepare_intermediates(config, config_path)
        LOGGER.info("Intermediate workspace is ready at %s", intermediates.workspace)
        progress.update()

    return config
