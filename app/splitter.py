"""In-memory OpenCV poster splitting: midline crop + auto person/interest pick."""

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

# --- YOLOv8n person detector (cv2.dnn, ONNX — no torch at runtime) ---
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH", str(Path(__file__).parent / "models" / "yolov8n.onnx")
)
# The bundled ONNX export has a fixed 1x3x640x640 input.
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESH = 0.4
YOLO_NMS_THRESH = 0.45
COCO_PERSON_CLASS_ID = 0

# The visual-interest fallback only compares the two halves against each
# other, so it runs on a downscaled copy instead of the full-size image.
INTEREST_MAX_SIDE = 800


class SplitterError(Exception):
    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(detail)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


_yolo_net: cv2.dnn.Net | None = None
_yolo_unavailable = False  # sticky only for load failures (missing/corrupt model)
# cv2.dnn.Net is not safe for concurrent setInput()/forward() calls, and
# requests now run in a thread pool. forward() is already multi-threaded
# internally, so serialising calls costs little throughput.
_yolo_lock = threading.Lock()


def load_detector() -> bool:
    """Load and warm up the person detector once. Returns False if the model
    can't be used in this environment; the caller then falls back to the
    visual-interest heuristic. Safe to call from several threads."""
    global _yolo_net, _yolo_unavailable
    if _yolo_net is not None:
        return True
    with _yolo_lock:
        if _yolo_net is not None:
            return True
        if _yolo_unavailable:
            return False
        try:
            net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH)
            # The first forward() pays for graph setup; do it here so the
            # first real request doesn't.
            net.setInput(np.zeros((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), np.float32))
            net.forward()
        except Exception as exc:
            # Missing/corrupt model file, opencv build without dnn support,
            # bad ONNX opset, etc. This won't fix itself, so stop retrying.
            logger.error(
                "Person detection unavailable, falling back to visual-interest "
                "heuristic for side='auto': %s",
                exc,
            )
            _yolo_unavailable = True
            return False
        _yolo_net = net
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


def _person_areas(img: np.ndarray, split_x: int) -> tuple[float, float] | None:
    """Area (px²) of the largest detected person in each half, as
    (left, right), with 0.0 for a half with nobody in it. Returns None if
    the detector can't be used, so the caller falls back.

    Runs one forward pass over the whole poster rather than one per half,
    then clips each person box to the half (or halves) it falls in."""
    if not load_detector():
        return None

    h, w = img.shape[:2]
    scale = YOLO_INPUT_SIZE / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, swapRB=True)

    try:
        with _yolo_lock:
            _yolo_net.setInput(blob)
            out = _yolo_net.forward()  # (1, 84, 8400)
    except Exception as exc:
        # One odd image shouldn't switch detection off for every later
        # request; fall back for this one only.
        logger.warning("Person detection failed for this image, using fallback: %s", exc)
        return None

    preds = out[0].T  # (8400, 84): [cx, cy, w, h, class0..class79]
    class_scores = preds[:, 4:]
    person_conf = class_scores[:, COCO_PERSON_CLASS_ID]
    mask = (person_conf >= YOLO_CONF_THRESH) & (
        class_scores.argmax(axis=1) == COCO_PERSON_CLASS_ID
    )
    if not mask.any():
        return 0.0, 0.0

    # Letterboxed 640x640 space -> original image pixels.
    cx, cy, bw, bh = (preds[mask, :4] / scale).T
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), person_conf[mask].tolist(), YOLO_CONF_THRESH, YOLO_NMS_THRESH
    )
    if len(keep) == 0:
        return 0.0, 0.0

    x0, y0, bw, bh = boxes[np.asarray(keep).flatten()].T
    x1, y1 = x0 + bw, y0 + bh
    box_h = np.clip(np.minimum(y1, h) - np.maximum(y0, 0), 0, None)
    left_w = np.clip(np.minimum(x1, split_x) - np.maximum(x0, 0), 0, None)
    right_w = np.clip(np.minimum(x1, w) - np.maximum(x0, split_x), 0, None)
    return float((left_w * box_h).max()), float((right_w * box_h).max())


def _visual_interest(gray: np.ndarray) -> float:
    """Edge density + Laplacian variance — prefer art over blank spine/margin."""
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return edge_density * 1000.0 + lap_var


def pick_side(img: np.ndarray, split_x: int) -> Literal["left", "right"]:
    """Pick the half with the biggest person in it, or the busier half if
    nobody is detected on either side (or detection is unavailable)."""
    areas = _person_areas(img, split_x)
    if areas is not None and max(areas) > 0.0:
        return "left" if areas[0] >= areas[1] else "right"

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    factor = min(1.0, INTEREST_MAX_SIDE / max(h, w))
    if factor < 1.0:
        gray = cv2.resize(
            gray, (max(2, round(w * factor)), max(1, round(h * factor))),
            interpolation=cv2.INTER_AREA,
        )
    gx = min(max(1, round(split_x * factor)), gray.shape[1] - 1)
    if _visual_interest(gray[:, :gx]) >= _visual_interest(gray[:, gx:]):
        return "left"
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
