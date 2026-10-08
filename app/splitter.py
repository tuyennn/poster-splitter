"""In-memory OpenCV poster splitting: midline crop + auto face pick."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Side = Literal["left", "right", "auto"]
Layout = Literal["single", "double"]

# Width/height of one portrait cover. Single-cover inputs are center-cropped
# to this when the caller doesn't pass target_ratio.
SINGLE_COVER_RATIO = 2 / 3

# Aspect bands (width/height). Two 2:3 covers side by side make 4:3 (1.33),
# one makes 0.67; SINGLE_MAX_ASPECT sits near the geometric midpoint (0.94)
# so anything closer to one cover than to two is treated as a single cover.
SINGLE_MIN_ASPECT = 0.45
SINGLE_MAX_ASPECT = 0.95
MAX_ASPECT = 2.2

# Decompression-bomb guard (~40 megapixels). app/__init__.py also hands this
# limit to OpenCV so it is enforced from the image header, before decoding.
MAX_MEGAPIXELS = 40_000_000

# --- YuNet face detector (cv2.FaceDetectorYN, ONNX) ---
FACE_MODEL_PATH = os.environ.get(
    "FACE_MODEL_PATH",
    str(Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"),
)
FACE_CONF_THRESH = 0.6
FACE_NMS_THRESH = 0.3
# Faces are detected on a copy downscaled to this longest side; a face
# smaller than ~10px there is too small to matter for picking a cover.
FACE_MAX_SIDE = 960

class SplitterError(Exception):
    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(detail)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


_face_det: cv2.FaceDetectorYN | None = None
_face_unavailable = False
_face_lock = threading.Lock()  # FaceDetectorYN holds per-call input size


def load_face_detector() -> bool:
    """Load and warm up the face detector once. Returns False if it can't be
    used; side='auto' then returns the right half. Safe to call from
    several threads."""
    global _face_det, _face_unavailable
    if _face_det is not None:
        return True
    with _face_lock:
        if _face_det is not None:
            return True
        if _face_unavailable:
            return False
        try:
            det = cv2.FaceDetectorYN.create(
                FACE_MODEL_PATH, "", (320, 320), FACE_CONF_THRESH, FACE_NMS_THRESH
            )
            det.detect(np.zeros((320, 320, 3), np.uint8))  # warm-up
        except Exception as exc:
            logger.error("Face detection unavailable for side='auto': %s", exc)
            _face_unavailable = True
            return False
        _face_det = det
        return True


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes in memory; enforce the megapixel cap."""
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        # Raised by OpenCV's OPENCV_IO_MAX_IMAGE_PIXELS check, from the image
        # header, before any pixel buffer is allocated.
        if "CV_IO_MAX_IMAGE_PIXELS" in str(exc):
            raise SplitterError(
                413,
                "invalid_image",
                f"Decoded image exceeds maximum of {MAX_MEGAPIXELS // 1_000_000} megapixels.",
            ) from exc
        raise SplitterError(
            400, "invalid_image", "Content is not a valid JPEG/PNG/WEBP image."
        ) from exc
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

    return img


def classify_layout(img: np.ndarray) -> Layout:
    """'double' for a two-cover combo, 'single' for one portrait cover;
    422 for anything outside both aspect bands."""
    height, width = img.shape[:2]
    aspect = width / height
    if SINGLE_MIN_ASPECT <= aspect < SINGLE_MAX_ASPECT:
        return "single"
    if SINGLE_MAX_ASPECT <= aspect <= MAX_ASPECT:
        return "double"
    raise SplitterError(
        422,
        "invalid_aspect",
        f"Image aspect ratio {aspect:.2f} is neither a single cover "
        f"({SINGLE_MIN_ASPECT}-{SINGLE_MAX_ASPECT}) nor a double-poster "
        f"({SINGLE_MAX_ASPECT}-{MAX_ASPECT}).",
    )


def _face_areas(img: np.ndarray, split_x: int) -> tuple[float, float] | None:
    """Area (px²) of the largest face in each half, as (left, right), with
    0.0 for a half with no face. Each face counts for the half its centre is
    in. Returns None if the detector can't be used."""
    if not load_face_detector():
        return None

    h, w = img.shape[:2]
    factor = min(1.0, FACE_MAX_SIDE / max(h, w))
    small = img
    if factor < 1.0:
        small = cv2.resize(
            img, (max(1, round(w * factor)), max(1, round(h * factor))),
            interpolation=cv2.INTER_AREA,
        )
    try:
        with _face_lock:
            _face_det.setInputSize((small.shape[1], small.shape[0]))
            _, faces = _face_det.detect(small)
    except Exception as exc:
        logger.warning("Face detection failed for this image, skipping it: %s", exc)
        return None
    if faces is None or len(faces) == 0:
        return 0.0, 0.0

    x, fw, fh = faces[:, 0] / factor, faces[:, 2] / factor, faces[:, 3] / factor
    area = fw * fh
    on_left = (x + fw / 2) < split_x
    left = float(area[on_left].max()) if on_left.any() else 0.0
    right = float(area[~on_left].max()) if (~on_left).any() else 0.0
    return left, right


def pick_side(img: np.ndarray, split_x: int) -> Literal["left", "right"]:
    """Pick the half with the biggest face in it, or the right half if no
    face is found (or face detection is unavailable)."""
    faces = _face_areas(img, split_x)
    if faces is not None and max(faces) > 0.0:
        return "left" if faces[0] >= faces[1] else "right"
    return "right"


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


def parse_ratio(value: str) -> float:
    """Parse a target aspect ratio given as 'W:H' (e.g. '2:3') or a plain
    decimal (e.g. '0.6667'). Returns width/height as a float."""
    text = value.strip()
    for sep in (":", "x", "/"):
        if sep in text:
            parts = text.split(sep)
            if len(parts) != 2:
                raise SplitterError(
                    400, "invalid_param", f"Could not parse ratio '{value}'."
                )
            try:
                w, h = float(parts[0]), float(parts[1])
            except ValueError as exc:
                raise SplitterError(
                    400, "invalid_param", f"Could not parse ratio '{value}'."
                ) from exc
            if h <= 0:
                raise SplitterError(
                    400, "invalid_param", f"Invalid ratio '{value}': height must be > 0."
                )
            return w / h
    try:
        return float(text)
    except ValueError as exc:
        raise SplitterError(
            400, "invalid_param", f"Could not parse ratio '{value}'."
        ) from exc


def _crop_to_ratio(img: np.ndarray, target_ratio: float) -> np.ndarray:
    """Center-crop img (width/height) down to target_ratio. Only ever crops
    (never pads/upscales), so the result is always a true subset of img."""
    height, width = img.shape[:2]
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # Wider than target -> crop width, keep full height
        new_width = max(1, min(width, round(height * target_ratio)))
        x0 = (width - new_width) // 2
        return img[:, x0 : x0 + new_width]

    # Taller than target -> crop height, keep full width
    new_height = max(1, min(height, round(width / target_ratio)))
    y0 = (height - new_height) // 2
    return img[y0 : y0 + new_height, :]


def split_poster(
    img: np.ndarray,
    side: Side = "auto",
    midline: float = 0.5,
    spine_trim: float = 0.0,
    target_ratio: float | None = None,
    layout: Layout = "double",
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

    if target_ratio is not None and not 0.1 <= target_ratio <= 5.0:
        raise SplitterError(
            400,
            "invalid_param",
            "target_ratio must resolve to a width/height between 0.1 and 5.0.",
        )

    if layout == "single":
        # Already one cover: no midline split, no spine to trim — just bring
        # it to the single-cover ratio (or the caller's target_ratio).
        return _crop_to_ratio(img, target_ratio or SINGLE_COVER_RATIO)

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
        chosen_side = pick_side(img, split_x)
        chosen = left if chosen_side == "left" else right

    result = _trim_spine(chosen, chosen_side, spine_trim)
    if target_ratio is not None:
        result = _crop_to_ratio(result, target_ratio)
    return result


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SplitterError(500, "internal_error", "Failed to encode PNG.")
    return buf.tobytes()
