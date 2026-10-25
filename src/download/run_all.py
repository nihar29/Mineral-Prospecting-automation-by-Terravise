"""Stage 1: download every enabled data source, then check that manual layers exist.

Usage:  python -m src.download.run_all --config config/params.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.download.fetch_stac import fetch_collection
from src.utils.config import cli_config, get_logger

log = get_logger("stage1.download")


def main() -> None:
    cfg = cli_config(__doc__)
    for source in ("landsat", "dem", "sentinel1"):
        if cfg["sources"].get(source, {}).get("enabled"):
            fetch_collection(cfg, source)
    if cfg["sources"].get("aster", {}).get("enabled"):
        from src.download.fetch_aster import fetch_aster
        fetch_aster(cfg)

    missing = [p for group in cfg["manual_layers"].values() for p in group.values()
               if not Path(p).exists()]
    if missing:
        log.warning("Manual layers not found (they will be skipped): %s", missing)
    if not Path(cfg["labels"]["positives"]).exists():
        raise FileNotFoundError(f"Label file missing: {cfg['labels']['positives']}")

    Path(cfg["paths"]["raw"], ".download_done").touch()
    log.info("Stage 1 complete")


if __name__ == "__main__":
    main()
