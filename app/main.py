"""Stateless double-DVD poster splitter API."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.fetcher import FetcherError, fetch_image
from app.splitter import (
    SplitterError,
    classify_layout,
    decode_image,
    encode_png,
    load_face_detector,
    parse_ratio,
    split_poster,
)

logger = logging.getLogger("poster_splitter")

# How many posters may be decoded/split/encoded at once. Each one can hold a
# 40 MP image (~120 MB) plus copies, so this bounds memory as well as CPU.
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", os.cpu_count() or 2)))
_job_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load and warm the face detector before taking traffic, off the event loop.
    await run_in_threadpool(load_face_detector)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Poster Splitter",
    description=(
        "Stateless service: fetch a landscape double-DVD combo poster, "
        "crop one portrait panel, stream PNG bytes back. A single portrait "
        "cover (any non double-poster ratio) is cropped to the single-cover "
        "ratio around the largest face. "
        "Nothing is written to disk."
    ),
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Safety net: any bug that isn't already turned into a FetcherError/
    # SplitterError still gets logged with a full traceback server-side
    # and returns a structured JSON body instead of a bare 500 page.
    logger.exception("Unhandled exception while processing %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


def _render(
    data: bytes,
    side: Literal["left", "right", "auto"],
    midline: float,
    spine_trim: float,
    target_ratio: Optional[float],
) -> tuple[bytes, str]:
    img = decode_image(data)
    layout = classify_layout(img)
    cropped = split_poster(
        img,
        side=side,
        midline=midline,
        spine_trim=spine_trim,
        target_ratio=target_ratio,
        layout=layout,
    )
    return encode_png(cropped), layout


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
        description=(
            "left or right half, or auto: the half holding the largest "
            "detected face; if that face sits across the middle, one cover's "
            "ratio cropped around it instead; with no face, the right half"
        ),
    ),
    midline: float = Query(
        0.5,
        description="Vertical split position as a fraction of width (0–1)",
    ),
    spine_trim: float = Query(
        0.05,
        description=(
            "Fraction of the returned half's width to trim off its inner "
            "edge (nearest the midline), to drop the spine/gutter divider "
            "between the two DVD panels. 0 disables trimming."
        ),
    ),
    target_ratio: Optional[str] = Query(
        None,
        description=(
            "Desired output aspect ratio as 'W:H' (e.g. '2:3') or a decimal "
            "(e.g. '0.6667'). Center-crops the panel to this exact ratio "
            "after spine trim. Only ever crops, never pads/upscales. "
            "Omit to keep the natural split width."
        ),
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

    if spine_trim < 0.0 or spine_trim >= 1.0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_param",
                "detail": "spine_trim must be a float between 0 (inclusive) and 1 (exclusive).",
            },
        )

    parsed_ratio: Optional[float] = None
    if target_ratio is not None and target_ratio.strip():
        try:
            parsed_ratio = parse_ratio(target_ratio)
        except SplitterError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
        if not 0.1 <= parsed_ratio <= 5.0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_param",
                    "detail": "target_ratio must resolve to a width/height between 0.1 and 5.0.",
                },
            )

    try:
        data = await fetch_image(url)
        # Decode, detection and PNG encode are CPU-bound; running them on the
        # event loop would stall every other request (and /health) meanwhile.
        async with _job_slots:
            png_bytes, layout = await run_in_threadpool(
                _render, data, side, midline, spine_trim, parsed_ratio
            )
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"X-Poster-Layout": layout},
        )

    except FetcherError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
    except SplitterError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
