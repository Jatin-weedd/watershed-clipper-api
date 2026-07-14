"""
Watershed COG Clipping Service
===============================

FastAPI service (deployable on Render) that:
  1. Authenticates to Hugging Face Hub with HF_TOKEN.
  2. Streams a single watershed geometry out of a large FlatGeobuf
     (JS2512/hydrology-data-vault/Watershed.fgb) using an attribute
     filter pushed down via pyogrio/GDAL -- no full-file download.
  3. Determines which of the COG tiles in JS2512/india-tri intersect
     that geometry (cheap metadata-only reads over /vsicurl/).
  4. Opens only the intersecting tiles with rioxarray (masked=True,
     lazy/windowed), mosaics them if more than one is needed, and
     clips to the exact watershed boundary using rio.clip(..., from_disk=True)
     so only the required window is ever pulled over the network.
  5. Returns the clipped raster as an in-memory GeoTIFF byte stream.

Env vars:
  HF_TOKEN                 - Hugging Face access token (required)
  WATERSHED_DATASET_REPO   - default "JS2512/hydrology-data-vault"
  WATERSHED_FILE           - default "Watershed.fgb"
  WATERSHED_ID_FIELD       - default "watershed_id"
  COG_DATASET_REPO         - default "JS2512/india-tri"
"""

import io
import os
import logging
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import geopandas as gpd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.io import MemoryFile
from rioxarray.merge import merge_arrays
from shapely.geometry import box
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("watershed-clip-service")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning(
        "HF_TOKEN is not set. Requests to gated/private Hugging Face "
        "datasets will fail."
    )

WATERSHED_DATASET_REPO = os.environ.get(
    "WATERSHED_DATASET_REPO", "JS2512/hydrology-data-vault"
)
WATERSHED_FILE = os.environ.get("WATERSHED_FILE", "Watershed.fgb")
WATERSHED_ID_FIELD = os.environ.get("WATERSHED_ID_FIELD", "watershed_id")
COG_DATASET_REPO = os.environ.get("COG_DATASET_REPO", "JS2512/india-tri")

# Server-side API key that clients must send back via the X-API-KEY header.
# Set this in Render's Environment tab. If it's left unset, the endpoint
# stays open (useful for local dev) but a warning is logged so it's never
# silently unprotected in production.
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logger.warning(
        "API_KEY is not set. /clip-watershed will accept requests from "
        "anyone with no key check. Set API_KEY in the environment to lock "
        "this down."
    )

# GDAL tuning: keep vsicurl chatty operations to a minimum and only allow
# the extensions we actually expect, which avoids extra HEAD/LIST calls.
GDAL_BASE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.fgb",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_MULTIPLEX": "YES",
    # rasterio/GDAL will use this only when byte-range reads are supported
    # by the server (Hugging Face's CDN does support Range requests).
    "GDAL_HTTP_VERSION": "2",
}


def _auth_header_string() -> str:
    """GDAL wants raw 'Header: value' pairs, newline separated."""
    if not HF_TOKEN:
        return ""
    return f"Authorization: Bearer {HF_TOKEN}"


@contextmanager
def gdal_hf_env():
    """
    Context manager that configures GDAL/rasterio to authenticate against
    Hugging Face's resolve endpoints over /vsicurl/, so gated/private
    datasets can be streamed without a local download.
    """
    env = dict(GDAL_BASE_ENV)
    header = _auth_header_string()
    if header:
        env["GDAL_HTTP_HEADERS"] = header
    with rasterio.Env(**env):
        yield


def vsicurl(url: str) -> str:
    return f"/vsicurl/{url}"


# --------------------------------------------------------------------------
# Hugging Face helpers
# --------------------------------------------------------------------------

_hf_api = HfApi(token=HF_TOKEN)

# Cached at process start: list of (tif_url, bounds, crs) for the 9 COGs.
_tile_index: Optional[List[dict]] = None


def _resolve_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    return hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)


def build_tile_index() -> List[dict]:
    """
    List every COG in COG_DATASET_REPO and open each one just far enough
    to read its bounds/CRS (a handful of HTTP range requests against the
    header/metadata of the file, not the pixel data).
    """
    global _tile_index
    if _tile_index is not None:
        return _tile_index

    try:
        repo_files = _hf_api.list_repo_files(
            repo_id=COG_DATASET_REPO, repo_type="dataset"
        )
    except (RepositoryNotFoundError, EntryNotFoundError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not list COG repo: {exc}"
        )

    tif_files = [f for f in repo_files if f.lower().endswith((".tif", ".tiff"))]
    if not tif_files:
        raise HTTPException(
            status_code=502, detail=f"No COG tiles found in {COG_DATASET_REPO}"
        )

    index = []
    with gdal_hf_env():
        for filename in tif_files:
            url = _resolve_url(COG_DATASET_REPO, filename)
            try:
                with rasterio.open(vsicurl(url)) as ds:
                    index.append(
                        {
                            "filename": filename,
                            "url": url,
                            "bounds": ds.bounds,  # left, bottom, right, top
                            "crs": ds.crs,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not open tile metadata for %s: %s", filename, exc)

    if not index:
        raise HTTPException(
            status_code=502, detail="Could not read metadata for any COG tile."
        )

    _tile_index = index
    logger.info("Indexed %d COG tiles from %s", len(index), COG_DATASET_REPO)
    return _tile_index


def find_intersecting_tiles(geom_wgs84_or_native, geom_crs) -> List[dict]:
    """
    Given a shapely geometry + its CRS, return the tile index entries whose
    bounding box intersects it. Reprojects the geometry into each tile's
    CRS only if necessary (COGs in a single collection are usually all in
    the same CRS, but we don't assume that).
    """
    tiles = build_tile_index()
    matches = []

    gdf = gpd.GeoDataFrame(geometry=[geom_wgs84_or_native], crs=geom_crs)

    for tile in tiles:
        tile_geom = gdf
        if tile["crs"] is not None and tile["crs"] != geom_crs:
            tile_geom = gdf.to_crs(tile["crs"])
        target_geom = tile_geom.geometry.iloc[0]
        tile_box = box(*tile["bounds"])
        if tile_box.intersects(target_geom):
            matches.append(tile)

    return matches


# --------------------------------------------------------------------------
# Watershed lookup
# --------------------------------------------------------------------------

def fetch_watershed_geometry(watershed_id: str) -> gpd.GeoDataFrame:
    """
    Pull only the requested watershed feature out of the FlatGeobuf using
    an OGR SQL attribute filter (pyogrio `where=`), so the rest of the
    file is never streamed.
    """
    fgb_url = _resolve_url(WATERSHED_DATASET_REPO, WATERSHED_FILE)

    where_clause = f"{WATERSHED_ID_FIELD} = '{watershed_id}'"

    with gdal_hf_env():
        try:
            gdf = gpd.read_file(
                vsicurl(fgb_url),
                engine="pyogrio",
                where=where_clause,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Failed to read watershed vault: {exc}",
            )

    if gdf.empty:
        raise HTTPException(
            status_code=404,
            detail=f"watershed_id '{watershed_id}' not found in {WATERSHED_FILE}",
        )

    return gdf


# --------------------------------------------------------------------------
# Raster clipping
# --------------------------------------------------------------------------

def open_tile_lazy(tile: dict) -> xr.DataArray:
    """Open a single COG tile with rioxarray, masked + lazily chunked."""
    da = rioxarray.open_rasterio(
        vsicurl(tile["url"]),
        masked=True,
        chunks=True,  # dask-backed -> no eager full read
    )
    if da.ndim == 3 and da.shape[0] == 1:
        # keep band dimension; downstream clip works fine with it
        pass
    return da


def build_mosaic_for_geometry(watershed_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    geom = watershed_gdf.geometry.iloc[0]
    geom_crs = watershed_gdf.crs

    with gdal_hf_env():
        tiles = find_intersecting_tiles(geom, geom_crs)

        if not tiles:
            raise HTTPException(
                status_code=404,
                detail="No COG tile intersects this watershed's geometry.",
            )

        logger.info(
            "Watershed intersects %d/%d tiles: %s",
            len(tiles),
            len(build_tile_index()),
            [t["filename"] for t in tiles],
        )

        arrays = [open_tile_lazy(t) for t in tiles]

        if len(arrays) == 1:
            mosaic = arrays[0]
        else:
            # merge_arrays requires matching CRS/resolution; reproject_match
            # is used defensively in case tiles differ slightly.
            ref_crs = arrays[0].rio.crs
            aligned = []
            for da in arrays:
                if da.rio.crs != ref_crs:
                    da = da.rio.reproject(ref_crs)
                aligned.append(da)
            mosaic = merge_arrays(aligned)

        return mosaic


def clip_to_watershed(mosaic: xr.DataArray, watershed_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    geom_crs = watershed_gdf.crs
    if mosaic.rio.crs != geom_crs:
        watershed_gdf = watershed_gdf.to_crs(mosaic.rio.crs)

    geometry = [watershed_gdf.geometry.iloc[0].__geo_interface__]

    with gdal_hf_env():
        try:
            clipped = mosaic.rio.clip(
                geometry,
                crs=mosaic.rio.crs,
                all_touched=True,
                from_disk=True,   # stream only the required window
                drop=True,
            )
        except rioxarray.exceptions.NoDataInBounds:
            raise HTTPException(
                status_code=404,
                detail="Watershed geometry does not overlap any valid raster data.",
            )

    return clipped


def clipped_array_to_tiff_bytes(clipped: xr.DataArray) -> bytes:
    """Materialize the (small, already-clipped) array and write a GeoTIFF to memory."""
    clipped = clipped.rio.write_nodata(clipped.rio.nodata, inplace=False)

    data = clipped.values  # triggers the (now small) dask computation, if any
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    transform = clipped.rio.transform()
    crs = clipped.rio.crs
    nodata = clipped.rio.nodata

    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }

    buffer = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(data)
        buffer.write(memfile.read())

    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title="Watershed COG Clipping Service",
    description="Clips India-TRI COG rasters to a requested watershed boundary.",
    version="1.0.0",
)


class ClipRequest(BaseModel):
    watershed_id: str = Field(..., description="ID of the watershed to clip against")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency that enforces the X-API-KEY header when API_KEY is
    configured on the server. Raises 401 on a missing/incorrect key.
    """
    if not API_KEY:
        # No key configured server-side -> auth check is a no-op (dev mode).
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-KEY.")


@app.on_event("startup")
def _startup() -> None:
    if not HF_TOKEN:
        return
    try:
        build_tile_index()
    except HTTPException as exc:
        logger.error("Startup tile indexing failed: %s", exc.detail)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "hf_token_present": bool(HF_TOKEN),
        "api_key_protection_enabled": bool(API_KEY),
    }


@app.post("/clip-watershed", dependencies=[Depends(verify_api_key)])
def clip_watershed(payload: ClipRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing HF_TOKEN.")

    watershed_gdf = fetch_watershed_geometry(payload.watershed_id)
    mosaic = build_mosaic_for_geometry(watershed_gdf)
    clipped = clip_to_watershed(mosaic, watershed_gdf)
    tiff_bytes = clipped_array_to_tiff_bytes(clipped)

    filename = f"{payload.watershed_id}_clipped.tif"
    return StreamingResponse(
        io.BytesIO(tiff_bytes),
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
