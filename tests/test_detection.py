"""side=auto face detection, run against the real bundled model."""

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
    # Same photo with the face painted over: a big person but no face.
    # Stands in for a figure seen from behind or a masked character.
    cover = person_cover().copy()
    cv2.rectangle(cover, (83, 50), (193, 190), (40, 40, 40), -1)
    return cover


def busy_cover():
    return (np.random.default_rng(0).random((H, COVER_W, 3)) * 255).astype(np.uint8)


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
    # Reported case: auto used to pick the cover without a face because its
    # person was bigger. The face side must win.
    covers = [small_person_cover(), faceless_person_cover()]
    if face_side == "right":
        covers.reverse()
    poster = np.hstack(covers)
    faces = splitter._face_areas(poster, COVER_W)
    face_i = 0 if face_side == "left" else 1
    assert faces[face_i] > 0.0 and faces[1 - face_i] == 0.0
    assert splitter.pick_side(poster, COVER_W) == face_side


def test_face_detection_works_on_large_poster():
    poster = np.hstack([small_person_cover(), busy_cover()])
    big = cv2.resize(poster, (poster.shape[1] * 6, poster.shape[0] * 6))
    assert splitter._face_areas(big, big.shape[1] // 2)[0] > 0.0
    assert splitter.pick_side(big, big.shape[1] // 2) == "left"


@pytest.mark.parametrize("big_side", ["left", "right"])
def test_bigger_face_wins(big_side):
    covers = [person_cover(), small_person_cover()]
    if big_side == "right":
        covers.reverse()
    poster = np.hstack(covers)
    assert min(splitter._face_areas(poster, COVER_W)) > 0.0  # both faces found
    assert splitter.pick_side(poster, COVER_W) == big_side


def test_no_face_returns_right():
    # A faceless person on the left doesn't count; no face means right.
    poster = np.hstack([faceless_person_cover(), busy_cover()])
    assert splitter._face_areas(poster, COVER_W) == (0.0, 0.0)
    assert splitter.pick_side(poster, COVER_W) == "right"
    out = splitter.split_poster(poster, side="auto")
    assert np.array_equal(out, poster[:, COVER_W:])


def test_missing_model_returns_right(monkeypatch):
    monkeypatch.setattr(splitter, "_face_det", None)
    monkeypatch.setattr(splitter, "_face_unavailable", False)
    monkeypatch.setattr(splitter, "FACE_MODEL_PATH", "/nonexistent/face.onnx")
    poster = np.hstack([person_cover(), busy_cover()])
    assert not splitter.load_face_detector()
    assert splitter._face_areas(poster, COVER_W) is None
    assert splitter.pick_side(poster, COVER_W) == "right"


def _canvas(w, h):
    return np.full((h, w, 3), 90, np.uint8)


def _face_x_in(out):
    faces = splitter._detect_faces(out)
    assert faces is not None and len(faces) > 0
    x, _, fw, _ = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
    return x + fw / 2


def test_single_cover_crops_around_biggest_face():
    # A 3:1 banner: not a double-poster ratio, so it is handled as one cover.
    # The big face sits near the centre; a smaller one is at the far left.
    banner = _canvas(1536, 512)
    banner[:, 640:1152] = PHOTO
    banner[128:384, 0:256] = cv2.resize(PHOTO, (256, 256))
    assert splitter.classify_layout(banner) == "single"

    out = splitter.split_poster(banner, layout="single")
    assert out.shape[:2] == (512, 341)  # 2:3, full height
    # The big face is centred in the output (within a few pixels).
    assert abs(_face_x_in(out) - 341 / 2) < 8


def test_single_cover_follows_face_off_centre():
    # Face high up in a tall image: a plain centre crop would cut it off.
    tall = _canvas(512, 1100)
    tall[:512] = PHOTO
    out = splitter.split_poster(tall, layout="single")
    assert out.shape[:2] == (768, 512)
    assert np.array_equal(out, tall[:768])  # window clamped to the top edge
    assert splitter._detect_faces(out).shape[0] == 1


def test_single_cover_target_ratio_centres_on_face():
    banner = _canvas(1536, 512)
    banner[:, 300:812] = PHOTO  # face centre at x ~523
    out = splitter.split_poster(banner, layout="single", target_ratio=0.5)
    assert out.shape[:2] == (512, 256)
    assert abs(_face_x_in(out) - 128) < 8


def test_single_cover_without_face_is_centre_cropped():
    banner = (np.random.default_rng(1).random((300, 1200, 3)) * 255).astype(np.uint8)
    out = splitter.split_poster(banner, layout="single")
    assert np.array_equal(out, banner[:, 500:700])
