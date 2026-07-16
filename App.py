"""
India TRI Data Gateway — Byte-Range Proxy (Render deployment)
─────────────────────────────────────────────────────────────
This is NOT a rendering API. It does no cropping, reprojecting, decoding,
or PNG encoding. It's a thin authenticated gatekeeper in front of the raw
private COG tiles:

  Client (their own GIS lib) ──(X-API-Key + HTTP Range)──►  This gateway
                                        │
                                        └── forwards the SAME Range request,
                                            server-side, to the private HF
                                            URL with Authorization: Bearer
                                            <HF_TOKEN> attached, and streams
                                            the response straight back.

The client never sees the Hugging Face URL or the HF_TOKEN. They only ever
see this gateway's own URL and their own X-API-Key.

WHY THIS IS LIGHTWEIGHT
─────────────────────────
There is no GDAL raster decoding, no numpy array math, no matplotlib, no
Pillow, at request time. The only per-request work is: check the API key,
check/record a usage quota, forward an HTTP Range request, forward the
response back. This is why a tiny CPU allocation (e.g. Render's free 0.1
vCPU instance) is genuinely fine for this design, unlike the earlier
server-side-rendering version.

The tile FOOTPRINT INDEX (bounding boxes) is still built at startup exactly
like the rendering-API version — reading just the COG headers via
/vsicurl/, not the data — because clients still need to know which tile(s)
cover the area they care about. That's exposed via /api/v1/manifest.

WHAT STOPS SOMEONE FROM RECONSTRUCTING THE WHOLE DATASET
────────────────────────────────────────────────────────
Being "dumb" about geospatial concepts doesn't mean being dumb about abuse.
Two real controls are enforced here, not just theoretical ones:

  1. MAX_RANGE_BYTES — every single Range request is capped (default 8 MB).
     Open-ended ranges ("bytes=1000-", no explicit end) are rejected outright,
     since their size can't be checked before fetching.
  2. Per-API-key daily byte quota — cumulative bytes actually served are
     tracked in memory per key, in a rolling 24h window. Once a key exceeds
     DAILY_BYTE_QUOTA_PER_KEY, further requests get 429 until the window
     rolls over.

This state is in-memory only (see the note in the Usage tracking section
below) — acceptable for a first version, with a clearly marked upgrade path
to a persistent store (e.g. Redis) noted in the README if you need the quota
to survive restarts/scale beyond one instance.

WHY CLIENTS CAN USE THEIR OWN GIS TOOLS DIRECTLY
──────────────────────────────────────────────────
Because this gateway speaks plain HTTP Range semantics (like any normal
static file server that supports Accept-Ranges: bytes), any COG-aware
client — GDAL's own /vsicurl/, rasterio, QGIS, geotiff.js — can point
directly at this gateway's /tiles/<filename> URL as if it were an ordinary
remote COG, and it will "just work": these libraries already read COGs via
byte-range requests as their default behavior. The only addition needed on
the client side is passing the X-API-Key header.
"""

import os
import time
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import httpx
import rasterio
from rasterio.vrt import WarpedVRT
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from huggingface_hub import hf_hub_url

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN")
CLIENT_API_KEY = os.environ.get("CLIENT_API_KEY")  # if unset, gateway is unauthenticated (dev only)

TRI_REPO   = "J2003S/india-tri"
TRI_TILES  = [f"TRI_Tile{i}_COG.tif" for i in range(1, 10)]
REPO_TYPE  = "dataset"
WEB_CRS    = "EPSG:4326"
NODATA     = -9999.0

# Hard caps — tune to your actual commercial terms.
MAX_RANGE_BYTES         = 8 * 1024 * 1024          # 8 MB per single Range request
DAILY_BYTE_QUOTA_PER_KEY = 500 * 1024 * 1024        # 500 MB / rolling 24h per API key

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. Add it as an Environment Variable on this Render "
        "Web Service (Dashboard → your service → Environment)."
    )

app = FastAPI(
    title="India TRI Data Gateway",
    description="Authenticated, metered byte-range access to raw TRI COG tiles. No rendering — clients bring their own GIS tooling.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    # Browsers won't let client JS read these response headers unless the
    # server explicitly exposes them — needed for geotiff.js / any fetch-based
    # client to correctly interpret partial-content responses.
    expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
)


def _tile_url(filename: str) -> str:
    return hf_hub_url(repo_id=TRI_REPO, filename=filename, repo_type=REPO_TYPE)


def _auth_headers() -> str:
    return f"Authorization: Bearer {HF_TOKEN}"


GDAL_OPTS = {
    "GDAL_HTTP_HEADERS": _auth_headers(),
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "16000000",
    "GDAL_HTTP_MULTIRANGE": "YES",
}


@contextmanager
def open_tile_as_4326(filename: str):
    """
    Used ONLY at startup to read tile headers for the footprint index — not
    used per-request. This is the same header-only pattern as the earlier
    rendering-API version.
    """
    url = f"/vsicurl/{_tile_url(filename)}"
    with rasterio.Env(**GDAL_OPTS):
        with rasterio.open(url) as src:
            with WarpedVRT(src, crs=WEB_CRS, src_nodata=NODATA, nodata=NODATA) as vrt:
                yield vrt


# ─────────────────────────────────────────────────────────────────────────────
# Tile footprint index — built once at startup, read-only after that
# ─────────────────────────────────────────────────────────────────────────────
_index_lock: threading.Lock = threading.Lock()
_tile_bounds: Dict[str, Tuple[float, float, float, float]] = {}
_index_ready: bool = False


def _build_tile_index() -> None:
    global _index_ready
    with _index_lock:
        if _index_ready:
            return
        for fname in TRI_TILES:
            try:
                with open_tile_as_4326(fname) as vrt:
                    b = vrt.bounds
                    _tile_bounds[fname] = (b.left, b.bottom, b.right, b.top)
                    print(f"[startup] indexed {fname}: {b.left:.4f},{b.bottom:.4f},{b.right:.4f},{b.top:.4f}")
            except Exception as e:
                print(f"[startup] WARNING: failed to read bounds for {fname}: {e}")
        _index_ready = True


@app.on_event("startup")
def _on_startup():
    _build_tile_index()


def _ensure_index() -> None:
    if not _index_ready:
        _build_tile_index()


def check_api_key(x_api_key: Optional[str]) -> str:
    """Returns the key to use for usage tracking (falls back to 'anonymous' if auth is disabled)."""
    if CLIENT_API_KEY and x_api_key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return x_api_key or "anonymous"


# ─────────────────────────────────────────────────────────────────────────────
# Per-key usage tracking — in-memory, rolling 24h window
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: this resets whenever the process restarts (including Render's free-tier
# spin-down/spin-up cycle). Fine for a first version / low-stakes abuse
# prevention. If you need quotas to survive restarts or to hold across
# multiple instances, swap this dict for a Redis INCR-with-TTL pattern —
# the call sites (_check_and_record_usage) are the only place that'd change.
_usage_lock = threading.Lock()
_usage: Dict[str, Tuple[float, int]] = {}   # api_key -> (window_start_ts, bytes_served)


def _check_and_record_usage(api_key: str, nbytes: int) -> None:
    now = time.time()
    with _usage_lock:
        window_start, used = _usage.get(api_key, (now, 0))
        if now - window_start > 86400:
            window_start, used = now, 0
        if used + nbytes > DAILY_BYTE_QUOTA_PER_KEY:
            raise HTTPException(status_code=429, detail="Daily data quota exceeded for this API key.")
        _usage[api_key] = (window_start, used + nbytes)


# ─────────────────────────────────────────────────────────────────────────────
# Range header parsing / enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _parse_explicit_range_span(range_header: str) -> Optional[int]:
    """
    Returns the byte span of a 'bytes=START-END' range header, or None if
    the header is malformed, open-ended, or a multi-range request (all of
    which we reject — every COG-aware client we expect to support sends a
    single, explicit, closed range per request).
    """
    try:
        unit, _, spec = range_header.partition("=")
        if unit.strip().lower() != "bytes" or "," in spec:
            return None
        start_str, _, end_str = spec.partition("-")
        if not start_str.strip() or not end_str.strip():
            return None
        start, end = int(start_str), int(end_str)
        if end < start:
            return None
        return end - start + 1
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Unauthenticated — set this path as Render's Health Check Path."""
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "India TRI Data Gateway",
        "note": "This gateway forwards raw COG bytes only. It does not render, crop, or reproject anything — bring your own GIS tooling (rasterio, GDAL, geotiff.js, QGIS, etc).",
        "endpoints": {
            "GET /api/v1/manifest": "List of tiles with bounding boxes and their gateway URLs.",
            "GET /tiles/{filename}": "Byte-range access to one tile. Requires an explicit 'Range: bytes=start-end' header.",
        },
        "auth": "All routes below / require header: X-API-Key: <your key>",
    }


@app.get("/api/v1/manifest")
def manifest(request: Request, x_api_key: Optional[str] = Header(None)):
    """
    Tells the client which tiles exist, where they are, and what URL to
    fetch each one through. Clients use this to pick the right tile(s) for
    their area of interest before making any Range requests.
    """
    check_api_key(x_api_key)
    _ensure_index()
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "crs": "EPSG:4326",
        "nodata": NODATA,
        "tiles": [
            {
                "filename": fname,
                "bounds": {"west": l, "south": b, "east": r, "north": t},
                "url": f"{base}/tiles/{fname}",
            }
            for fname, (l, b, r, t) in _tile_bounds.items()
        ],
    })


@app.get("/tiles/{filename}")
async def get_tile_bytes(filename: str, request: Request, x_api_key: Optional[str] = Header(None)):
    """
    The gatekeeper itself. Requires an explicit, single, closed Range header
    — this is how GDAL/rasterio/geotiff.js read COGs by default, so no
    special client behavior is needed beyond adding the X-API-Key header.
    """
    api_key = check_api_key(x_api_key)

    if filename not in TRI_TILES:
        raise HTTPException(status_code=404, detail="Unknown tile filename.")

    range_header = request.headers.get("range")
    if not range_header:
        raise HTTPException(
            status_code=416,
            detail="This endpoint requires an explicit 'Range: bytes=start-end' header — full-file downloads are not permitted.",
        )

    span = _parse_explicit_range_span(range_header)
    if span is None:
        raise HTTPException(
            status_code=416,
            detail="Range header must be a single explicit 'bytes=start-end' span (no open-ended or multi-range requests).",
        )
    if span > MAX_RANGE_BYTES:
        raise HTTPException(
            status_code=416,
            detail=f"Requested range too large ({span} bytes; max {MAX_RANGE_BYTES} bytes per request).",
        )

    upstream_url = _tile_url(filename)
    upstream_headers = {"Authorization": f"Bearer {HF_TOKEN}", "Range": range_header}

    async with httpx.AsyncClient(timeout=30.0) as client:
        upstream_resp = await client.get(upstream_url, headers=upstream_headers, follow_redirects=True)

    if upstream_resp.status_code not in (200, 206):
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed with status {upstream_resp.status_code}.")

    body = upstream_resp.content
    if len(body) > MAX_RANGE_BYTES:
        # Defensive second check — in case an upstream response ever returns
        # more than requested (shouldn't happen with a well-formed range, but
        # don't trust upstream blindly for something abuse-relevant).
        raise HTTPException(status_code=502, detail="Upstream returned more data than requested.")

    _check_and_record_usage(api_key, len(body))

    passthrough_headers = {"Accept-Ranges": "bytes"}
    for h in ("Content-Range", "Content-Length"):
        if h in upstream_resp.headers:
            passthrough_headers[h] = upstream_resp.headers[h]

    return Response(
        content=body,
        status_code=upstream_resp.status_code,
        headers=passthrough_headers,
        media_type=upstream_resp.headers.get("content-type", "application/octet-stream"),
    )
