"""Download Landsat, Copernicus DEM and Sentinel-1 RTC from Microsoft Planetary Computer (STAC).

Each asset is clipped to the AOI and saved as a compressed GeoTIFF under data/raw/<source>/.
A manifest (URL, date, SHA-256) is appended to data/raw/manifest.csv for reproducibility.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

from src.utils.config import get_logger

log = get_logger("download.stac")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_manifest(raw_dir: Path, source: str, item_id: str, asset: str, url: str, out: Path) -> None:
    manifest = raw_dir / "manifest.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["downloaded_utc", "source", "item_id", "asset", "url", "file", "sha256"])
        w.writerow([datetime.now(timezone.utc).isoformat(), source, item_id, asset,
                    url.split("?")[0], str(out), sha256(out)])


def _client():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def _save_asset(href: str, aoi: gpd.GeoDataFrame, out: Path) -> None:
    import rioxarray

    da = rioxarray.open_rasterio(href, masked=True)
    da = da.rio.clip(aoi.to_crs(da.rio.crs).geometry, from_disk=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(out, compress="deflate")


def fetch_collection(cfg: dict, source: str) -> list[Path]:
    """Search a STAC collection and download the least-cloudy (or first) item's assets."""
    scfg = cfg["sources"][source]
    raw_dir = Path(cfg["paths"]["raw"])
    aoi = gpd.read_file(cfg["aoi"]).to_crs(4326)
    geom = aoi.geometry.union_all().__geo_interface__

    search_kwargs = {"collections": [scfg["collection"]], "intersects": geom}
    if "dates" in scfg:
        search_kwargs["datetime"] = scfg["dates"]
    if "max_cloud" in scfg:
        search_kwargs["query"] = {"eo:cloud_cover": {"lt": scfg["max_cloud"]}}

    items = list(_client().search(**search_kwargs).items())
    if not items:
        raise RuntimeError(f"No {source} items found for AOI/date range; widen the search.")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 0))
    item = items[0]
    log.info("%s: using item %s (%d candidates)", source, item.id, len(items))

    assets = scfg.get("bands") or scfg.get("pol") or ["data"]
    written = []
    for asset in assets:
        if asset not in item.assets:
            log.warning("%s: asset '%s' not in item, skipping", source, asset)
            continue
        out = raw_dir / source / f"{asset}.tif"
        _save_asset(item.assets[asset].href, aoi, out)
        log_manifest(raw_dir, source, item.id, asset, item.assets[asset].href, out)
        written.append(out)
        log.info("  saved %s", out)
    return written
