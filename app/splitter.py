"""In-memory OpenCV poster splitting: midline crop + auto face/interest pick."""

from __future__ import annotations

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


def _count_faces(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )
    return len(faces)


def _visual_interest(img: np.ndarray) -> float:
    """Edge density + Laplacian variance — prefer art over blank spine/margin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return edge_density * 1000.0 + lap_var


def pick_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_faces = _count_faces(left)
    right_faces = _count_faces(right)

    if left_faces > 0 and right_faces == 0:
        return left
    if right_faces > 0 and left_faces == 0:
        return right

    # Both or neither have faces → visual interest fallback
    if _visual_interest(left) >= _visual_interest(right):
        return left
    return right


def split_poster(
    img: np.ndarray,
    side: Side = "right",
    midline: float = 0.5,
) -> np.ndarray:
    if not 0.0 < midline < 1.0:
        raise SplitterError(
            400,
            "invalid_param",
            "midline must be a float strictly between 0 and 1.",
        )

    height, width = img.shape[:2]
    split_x = int(width * midline)
    # Ensure both halves have at least 1 pixel
    split_x = max(1, min(split_x, width - 1))

    left = img[:, :split_x]
    right = img[:, split_x:]

    if side == "left":
        return left
    if side == "right":
        return right
    return pick_side(left, right)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SplitterError(500, "internal_error", "Failed to encode PNG.")
    return buf.tobytes()
