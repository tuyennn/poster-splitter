"""side=auto person detection, run against the real bundled YOLOv8n model."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import splitter

# Eileen Collins, NASA (public domain); see tests/data/README.md.
PHOTO = cv2.imread(str(Path(__file__).parent / "data" / "astronaut.jpg"))
H, COVER_W = 512, 341  # one 2:3 cover


@pytest.fixture(autouse=True)
def detector():
    if not splitter.load_detector():
        pytest.fail("bundled YOLO model failed to load")


def person_cover():
    return PHOTO[:, 85 : 85 + COVER_W]


def small_person_cover():
    small = cv2.resize(person_cover(), (COVER_W // 2, H // 2))
    return cv2.copyMakeBorder(
        small, H // 4, H // 4, COVER_W // 4, COVER_W - COVER_W // 2 - COVER_W // 4,
        cv2.BORDER_CONSTANT, value=(60, 60, 60),
    )


def busy_cover():
    # Dense noise: wins the edge/Laplacian fallback against any photo.
    return (np.random.default_rng(0).random((H, COVER_W, 3)) * 255).astype(np.uint8)


@pytest.mark.parametrize("person_side", ["left", "right"])
def test_detects_person_only_in_its_half(person_side):
    covers = [person_cover(), busy_cover()]
    if person_side == "right":
        covers.reverse()
    left, right = splitter._person_areas(np.hstack(covers), COVER_W)
    person, other = (left, right) if person_side == "left" else (right, left)
    # Most of the cover is the person.
    assert person > 0.4 * H * COVER_W
    assert other == 0.0


@pytest.mark.parametrize("person_side", ["left", "right"])
def test_person_beats_busier_half(person_side):
    # The noise half would win the visual-interest fallback, so picking the
    # photo shows the detector made the call.
    covers = [person_cover(), busy_cover()]
    if person_side == "right":
        covers.reverse()
    poster = np.hstack(covers)
    assert splitter.pick_side(poster, COVER_W) == person_side

    out = splitter.split_poster(poster, side="auto")
    expected = poster[:, :COVER_W] if person_side == "left" else poster[:, COVER_W:]
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("big_side", ["left", "right"])
def test_bigger_person_wins(big_side):
    covers = [person_cover(), small_person_cover()]
    if big_side == "right":
        covers.reverse()
    left, right = splitter._person_areas(np.hstack(covers), COVER_W)
    assert min(left, right) > 0.0  # both people found
    assert splitter.pick_side(np.hstack(covers), COVER_W) == big_side


def test_detection_works_on_large_poster():
    poster = np.hstack([busy_cover(), person_cover()])
    big = cv2.resize(poster, (poster.shape[1] * 6, poster.shape[0] * 6))
    assert splitter.pick_side(big, big.shape[1] // 2) == "right"


def test_no_person_falls_back_to_visual_interest():
    flat = np.full((H, COVER_W, 3), 60, np.uint8)
    poster = np.hstack([flat, busy_cover()])
    assert splitter._person_areas(poster, COVER_W) == (0.0, 0.0)
    assert splitter.pick_side(poster, COVER_W) == "right"


def test_missing_model_falls_back(monkeypatch):
    monkeypatch.setattr(splitter, "_yolo_net", None)
    monkeypatch.setattr(splitter, "_yolo_unavailable", False)
    monkeypatch.setattr(splitter, "YOLO_MODEL_PATH", "/nonexistent/model.onnx")
    poster = np.hstack([person_cover(), busy_cover()])
    assert not splitter.load_detector()
    assert splitter._person_areas(poster, COVER_W) is None
    # No detector: the busier half wins.
    assert splitter.pick_side(poster, COVER_W) == "right"
