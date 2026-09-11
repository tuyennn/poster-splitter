"""Stateless double-DVD poster splitter API."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from app.fetcher import FetcherError, fetch_image
from app.splitter import SplitterError, decode_image, encode_png, split_poster

app = FastAPI(
    title="Poster Splitter",
    description=(
        "Stateless service: fetch a landscape double-DVD combo poster, "
        "crop one portrait panel, stream PNG bytes back. Nothing is written to disk."
    ),
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/poster")
async def poster(
    url: Optional[str] = Query(
        None, description="Full-size source image URL to split"
    ),
    side: Literal["left", "right", "auto"] = Query(
        "auto",
        description="Which half to return: left, right, or auto (face/interest detection)",
    ),
    midline: float = Query(
        0.5,
        description="Vertical split position as a fraction of width (0–1)",
    ),
) -> Response:
    if not url or not url.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_url",
                "detail": "Missing required query parameter: url",
            },
        )

    if midline <= 0.0 or midline >= 1.0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_param",
                "detail": "midline must be a float strictly between 0 and 1.",
            },
        )

    try:
        data = await fetch_image(url)
        img = decode_image(data)
        cropped = split_poster(img, side=side, midline=midline)
        png_bytes = encode_png(cropped)
        return Response(content=png_bytes, media_type="image/png")

    except FetcherError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
    except SplitterError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
