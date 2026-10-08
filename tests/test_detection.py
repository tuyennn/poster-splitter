"""side=auto face/person detection, run against the real bundled models."""

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
    if not splitter.load_face_detector():
        pytest.fail("bundled face model failed to load")


def person_cover():
    return PHOTO[:, 85 : 85 + COVER_W]


def small_person_cover():
    small = cv2.resize(person_cover(), (COVER_W // 2, H // 2))
    return cv2.copyMakeBorder(
        small, H // 4, H // 4, COVER_W // 4, COVER_W - COVER_W // 2 - COVER_W // 4,
        cv2.BORDER_CONSTANT, value=(60, 60, 60),
    )


def faceless_person_cover():
    # Same photo with the face painted over: still a (big) person to YOLO,
    # but no face. Stands in for a figure seen from behind, a masked or
    # silhouetted character, or a person false positive.
    cover = person_cover().copy()
    cv2.rectangle(cover, (83, 50), (193, 190), (40, 40, 40), -1)
    return cover


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


@pytest.mark.parametrize("face_side", ["left", "right"])
def test_detects_face_only_in_its_half(face_side):
    covers = [person_cover(), busy_cover()]
    if face_side == "right":
        covers.reverse()
    left, right = splitter._face_areas(np.hstack(covers), COVER_W)
    face, other = (left, right) if face_side == "left" else (right, left)
    assert face > 0.0
    assert other == 0.0


@pytest.mark.parametrize("face_side", ["left", "right"])
def test_face_beats_bigger_faceless_person(face_side):
    # Reported case: auto picked the cover without a face because the other
    # cover's person box was bigger. The face side must win.
    covers = [small_person_cover(), faceless_person_cover()]
    if face_side == "right":
        covers.reverse()
    poster = np.hstack(covers)
    faces = splitter._face_areas(poster, COVER_W)
    persons = splitter._person_areas(poster, COVER_W)
    face_i = 0 if face_side == "left" else 1
    assert faces[face_i] > 0.0 and faces[1 - face_i] == 0.0
    assert persons[1 - face_i] > persons[face_i]  # person-only logic would lose
    assert splitter.pick_side(poster, COVER_W) == face_side


def test_face_beats_person_false_positive(monkeypatch):
    # Whatever the person detector claims, a face decides the side.
    monkeypatch.setattr(splitter, "_person_areas", lambda img, x: (0.0, 1e9))
    poster = np.hstack([person_cover(), busy_cover()])
    assert splitter.pick_side(poster, COVER_W) == "left"


def test_face_detection_works_on_large_poster():
    poster = np.hstack([busy_cover(), small_person_cover()])
    big = cv2.resize(poster, (poster.shape[1] * 6, poster.shape[0] * 6))
    assert splitter._face_areas(big, big.shape[1] // 2)[1] > 0.0
    assert splitter.pick_side(big, big.shape[1] // 2) == "right"


def test_missing_face_model_falls_back_to_person(monkeypatch):
    monkeypatch.setattr(splitter, "_face_det", None)
    monkeypatch.setattr(splitter, "_face_unavailable", False)
    monkeypatch.setattr(splitter, "FACE_MODEL_PATH", "/nonexistent/face.onnx")
    poster = np.hstack([person_cover(), busy_cover()])
    assert not splitter.load_face_detector()
    assert splitter._face_areas(poster, COVER_W) is None
    assert splitter.pick_side(poster, COVER_W) == "left"


def test_missing_model_falls_back(monkeypatch):
    monkeypatch.setattr(splitter, "_face_det", None)
    monkeypatch.setattr(splitter, "_face_unavailable", True)
    monkeypatch.setattr(splitter, "_yolo_net", None)
    monkeypatch.setattr(splitter, "_yolo_unavailable", False)
    monkeypatch.setattr(splitter, "YOLO_MODEL_PATH", "/nonexistent/model.onnx")
    poster = np.hstack([person_cover(), busy_cover()])
    assert not splitter.load_detector()
    assert splitter._person_areas(poster, COVER_W) is None
    # No detector: the busier half wins.
    assert splitter.pick_side(poster, COVER_W) == "right"
