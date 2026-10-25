"""Download ASTER surface reflectance (AST_07XT) from NASA Earthdata via `earthaccess`.

Requires a free Earthdata account. Run `earthaccess.login(persist=True)` once, or set
EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables.

Note: ASTER SWIR bands (B4-B9) are unusable after April 2008, so the clay, carbonate
and silica ratios must come from pre-2008 scenes.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from src.utils.config import get_logger

log = get_logger("download.aster")


def fetch_aster(cfg: dict) -> list[Path]:
    import earthaccess

    scfg = cfg["sources"]["aster"]
    out_dir = Path(cfg["paths"]["raw"]) / "aster"
    out_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()  # tries environment variables, then ~/.netrc, then prompts
    bounds = tuple(gpd.read_file(cfg["aoi"]).to_crs(4326).total_bounds)
    start, end = scfg["dates"].split("/")
    results = earthaccess.search_data(
        short_name=scfg["short_name"], bounding_box=bounds, temporal=(start, end), count=20
    )
    if not results:
        raise RuntimeError("No ASTER granules found; check dates and AOI.")
    log.info("ASTER: %d granules found, downloading the first", len(results))
    files = earthaccess.download(results[:1], local_path=str(out_dir))
    return [Path(f) for f in files]
