"""In-memory OpenCV poster splitting: midline crop + auto person/interest pick."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

Side = Literal["left", "right", "auto"]

# Landscape / near-square double-poster aspect range
MIN_ASPECT = 0.9
MAX_ASPECT = 2.2

# Decompression-bomb guard (~40 megapixels)
MAX_MEGAPIXELS = 40_000_000

# --- YOLOv8n person detector (cv2.dnn, ONNX — no torch at runtime) ---
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH", str(Path(__file__).parent / "models" / "yolov8n.onnx")
)
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRESH = 0.4
YOLO_NMS_THRESH = 0.45
COCO_PERSON_CLASS_ID = 0


class SplitterError(Exception):
    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(detail)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


_yolo_net: cv2.dnn.Net | None = None
_yolo_unavailable = False  # sticky flag once we've confirmed it's broken/missing


def _get_yolo_net() -> cv2.dnn.Net:
    global _yolo_net
    if _yolo_net is None:
        net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH)
        _yolo_net = net
    return _yolo_net


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


def _largest_person_area(img: np.ndarray) -> float:
    """Returns the area (px²) of the largest detected person, 0.0 if none
    found, or -1.0 if the detector isn't usable in this environment."""
    global _yolo_unavailable
    if _yolo_unavailable:
        return -1.0
    try:
        h, w = img.shape[:2]
        scale = YOLO_INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1 / 255.0, size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True
        )
        net = _get_yolo_net()
        net.setInput(blob)
        out = net.forward()  # (1, 84, 8400)
        out = out[0].T  # (8400, 84): [cx, cy, w, h, class0..class79]

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_scores = out[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), class_ids]
        person_mask = (class_ids == COCO_PERSON_CLASS_ID) & (confs >= YOLO_CONF_THRESH)

        for cx, cy, bw, bh in out[person_mask, :4]:
            boxes.append([float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)])
        scores = [float(c) for c in confs[person_mask]]

        if not boxes:
            return 0.0

        idxs = cv2.dnn.NMSBoxes(boxes, scores, YOLO_CONF_THRESH, YOLO_NMS_THRESH)
        if len(idxs) == 0:
            return 0.0

        best_area = 0.0
        for i in np.array(idxs).flatten():
            bw, bh = boxes[i][2], boxes[i][3]
            # boxes are in the 640x640 letterboxed space — undo scale to
            # get area back in the original image's pixel space.
            area = (bw / scale) * (bh / scale)
            best_area = max(best_area, area)
        return best_area
    except Exception as exc:
        # Missing/corrupt model file, incompatible opencv build without dnn
        # support, bad ONNX opset, etc. Don't fail every request over an
        # optional heuristic — log once and fall back to visual-interest.
        logging.getLogger(__name__).error(
            "Person detection unavailable, falling back to visual-interest "
            "heuristic for side='auto': %s",
            exc,
        )
        _yolo_unavailable = True
        return -1.0


def _visual_interest(img: np.ndarray) -> float:
    """Edge density + Laplacian variance — prefer art over blank spine/margin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return edge_density * 1000.0 + lap_var


def pick_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, Literal["left", "right"]]:
    left_area = _largest_person_area(left)
    right_area = _largest_person_area(right)

    if left_area > 0.0 or right_area > 0.0:
        # At least one side has a detected person — prefer whichever side's
        # biggest person is larger (a -1 "unavailable" side loses to any
        # real, even small, detection).
        if left_area >= right_area:
            return left, "left"
        return right, "right"

    # No people detected on either side (or detection is unavailable) →
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

    result = _trim_spine(chosen, chosen_side, spine_trim)
    if target_ratio is not None:
        result = _crop_to_ratio(result, target_ratio)
    return result


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SplitterError(500, "internal_error", "Failed to encode PNG.")
    return buf.tobytes()
