"""Load and access the pipeline configuration."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "config/params.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("raw", "interim", "processed", "artifacts", "outputs", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def cli_config(description: str) -> dict[str, Any]:
    """Parse the common --config argument used by every stage script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to params.yaml")
    args, _ = parser.parse_known_args()
    return load_config(args.config)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
