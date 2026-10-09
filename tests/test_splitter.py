import asyncio
import time

import cv2
import httpx
import numpy as np
import pytest

from app import main, splitter


def test_decompression_bomb_rejected_before_decode():
    # ~2.5 KB file that declares 8000x8000 (64 MP).
    pixels = np.zeros((8000, 8000, 3), np.uint8)
    bomb = cv2.imencode(".webp", pixels, [cv2.IMWRITE_WEBP_QUALITY, 101])[1]
    assert len(bomb) < 10_000
    # OpenCV itself refuses from the header, before allocating ~190 MB.
    with pytest.raises(cv2.error, match="CV_IO_MAX_IMAGE_PIXELS"):
        cv2.imdecode(bomb, cv2.IMREAD_COLOR)
    with pytest.raises(splitter.SplitterError) as exc:
        splitter.decode_image(bomb.tobytes())
    assert exc.value.status_code == 413


def _poster(busy_side):
    rng = np.random.default_rng(0)
    busy = (rng.random((600, 400, 3)) * 255).astype(np.uint8)
    flat = np.full((600, 400, 3), 60, np.uint8)
    return np.hstack([busy, flat] if busy_side == "left" else [flat, busy])


@pytest.mark.parametrize("busy_side", ["left", "right"])
def test_auto_without_face_returns_right(busy_side):
    poster = _poster(busy_side)
    out = splitter.split_poster(poster, side="auto")
    assert np.array_equal(out, poster[:, 400:])


def test_face_detection_failure_is_not_sticky(monkeypatch):
    assert splitter.load_face_detector()
    real_det = splitter._face_det

    class Broken:
        def setInputSize(self, size):
            pass

        def detect(self, img):
            raise RuntimeError("bad input")

    monkeypatch.setattr(splitter, "_face_det", Broken())
    assert splitter._detect_faces(_poster("left")) is None
    assert np.array_equal(splitter.split_poster(_poster("left")), _poster("left")[:, 400:])
    monkeypatch.setattr(splitter, "_face_det", real_det)
    assert len(splitter._detect_faces(_poster("left"))) == 0


def test_side_defaults_to_auto(monkeypatch):
    seen = {}

    async def fake_fetch(url):
        return cv2.imencode(".png", _poster("left"))[1].tobytes()

    def fake_render(data, side, *args):
        seen["side"] = side
        return b"png", "double"

    monkeypatch.setattr(main, "fetch_image", fake_fetch)
    monkeypatch.setattr(main, "_render", fake_render)

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.get("/poster", params={"url": "http://x/a.png"})

    assert asyncio.run(run()).status_code == 200
    assert seen["side"] == "auto"


def test_cpu_work_does_not_block_event_loop(monkeypatch):
    png = cv2.imencode(".png", _poster("left"))[1].tobytes()

    async def fake_fetch(url):
        return png

    def slow_render(*args):
        time.sleep(1.0)
        return png, "double"

    monkeypatch.setattr(main, "fetch_image", fake_fetch)
    monkeypatch.setattr(main, "_render", slow_render)

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            poster = asyncio.create_task(client.get("/poster", params={"url": "http://x/a.png"}))
            await asyncio.sleep(0.1)
            start = time.perf_counter()
            health = await client.get("/health")
            health_time = time.perf_counter() - start
            assert (await poster).status_code == 200
            return health.status_code, health_time

    status, elapsed = asyncio.run(run())
    assert status == 200
    assert elapsed < 0.5
