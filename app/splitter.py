"""In-memory OpenCV poster splitting: midline crop + auto face/interest pick."""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

Side = Literal["left", "right", "auto"]

# Landscape / near-square double-poster aspect range
MIN_ASPECT = 0.9
MAX_ASPECT = 2.2

# Decompression-bomb guard (~40 megapixels)
MAX_MEGAPIXELS = 40_000_000


class SplitterError(Exception):
    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(detail)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


_face_cascade: cv2.CascadeClassifier | None = None
_face_detection_unavailable = False  # sticky flag once we've confirmed it's broken


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise SplitterError(
                500,
                "internal_error",
                "Failed to load Haar cascade face detector.",
            )
        _face_cascade = cascade
    return _face_cascade


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes in memory; enforce megapixel + aspect checks."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise SplitterError(
            400,
            "invalid_image",
            "Content is not a valid JPEG/PNG/WEBP image.",
        )

    height, width = img.shape[:2]
    if width * height > MAX_MEGAPIXELS:
        raise SplitterError(
            413,
            "invalid_image",
            f"Decoded image exceeds maximum of {MAX_MEGAPIXELS // 1_000_000} megapixels.",
        )

    if height == 0:
        raise SplitterError(400, "invalid_image", "Image has zero height.")

    aspect = width / height
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        raise SplitterError(
            422,
            "invalid_aspect",
            f"Image aspect ratio {aspect:.2f} is not a landscape double-poster "
            f"(expected between {MIN_ASPECT} and {MAX_ASPECT}).",
        )

    return img


def _largest_face_area(img: np.ndarray) -> float:
    """Returns the area (px²) of the largest detected face, 0.0 if none
    found, or -1.0 if face detection isn't usable in this environment."""
    global _face_detection_unavailable
    if _face_detection_unavailable:
        return -1.0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = _get_face_cascade().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return 0.0
        return max(float(w * h) for (_x, _y, w, h) in faces)
    except (AttributeError, cv2.error) as exc:
        # Broken/incompatible OpenCV build (e.g. conflicting opencv-python /
        # opencv-python-headless installs leave cv2.CascadeClassifier missing).
        # Don't fail every request over an optional heuristic — log once and
        # fall back to the visual-interest comparison instead.
        logging.getLogger(__name__).error(
            "Face detection unavailable, falling back to visual-interest "
            "heuristic for side='auto': %s",
            exc,
        )
        _face_detection_unavailable = True
        return -1.0


def _visual_interest(img: np.ndarray) -> float:
    """Edge density + Laplacian variance — prefer art over blank spine/margin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return edge_density * 1000.0 + lap_var


def pick_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Literal["left", "right"]]:
    left_area = _largest_face_area(left)
    right_area = _largest_face_area(right)

    if left_area > 0.0 or right_area > 0.0:
        # At least one side has a detected face — prefer whichever side's
        # biggest face is larger (a -1 "unavailable" side loses to any
        # real, even small, detected face).
        if left_area >= right_area:
            return left, "left"
        return right, "right"

    # No faces detected on either side (or detection is unavailable) →
    # visual interest fallback.
    if _visual_interest(left) >= _visual_interest(right):
        return left, "left"
    return right, "right"


def _trim_spine(img: np.ndarray, side: Literal["left", "right"], spine_trim: float) -> np.ndarray:
    """Trim a sliver off the inner edge (nearest the midline) to drop the
    spine/gutter divider between the two DVD cover panels."""
    if spine_trim <= 0.0:
        return img

    width = img.shape[1]
    trim_px = int(round(width * spine_trim))
    trim_px = max(0, min(trim_px, width - 1))  # always keep at least 1px
    if trim_px == 0:
        return img

    if side == "left":
        # Spine sits along this half's RIGHT edge.
        return img[:, : width - trim_px]
    # side == "right": spine sits along this half's LEFT edge.
    return img[:, trim_px:]


def split_poster(
    img: np.ndarray,
    side: Side = "right",
    midline: float = 0.5,
    spine_trim: float = 0.0,
) -> np.ndarray:
    if not 0.0 < midline < 1.0:
        raise SplitterError(
            400,
            "invalid_param",
            "midline must be a float strictly between 0 and 1.",
        )

    if not 0.0 <= spine_trim < 1.0:
        raise SplitterError(
            400,
            "invalid_param",
            "spine_trim must be a float between 0 (inclusive) and 1 (exclusive).",
        )

    height, width = img.shape[:2]
    split_x = int(width * midline)
    # Ensure both halves have at least 1 pixel
    split_x = max(1, min(split_x, width - 1))

    left = img[:, :split_x]
    right = img[:, split_x:]

    if side == "left":
        chosen, chosen_side = left, "left"
    elif side == "right":
        chosen, chosen_side = right, "right"
    else:
        chosen, chosen_side = pick_side(left, right)

    return _trim_spine(chosen, chosen_side, spine_trim)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SplitterError(500, "internal_error", "Failed to encode PNG.")
    return buf.tobytes()
