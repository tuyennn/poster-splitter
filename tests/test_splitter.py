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
def test_auto_falls_back_to_busier_half(busy_side):
    assert splitter.pick_side(_poster(busy_side), 400) == busy_side


def test_detection_failure_is_not_sticky(monkeypatch):
    assert splitter.load_detector()
    real_net = splitter._yolo_net

    class Broken:
        def setInput(self, blob):
            pass

        def forward(self):
            raise RuntimeError("bad input")

    monkeypatch.setattr(splitter, "_yolo_net", Broken())
    assert splitter._person_areas(_poster("left"), 400) is None
    monkeypatch.setattr(splitter, "_yolo_net", real_net)
    assert splitter._person_areas(_poster("left"), 400) == (0.0, 0.0)


def test_person_box_is_clipped_to_each_half(monkeypatch):
    # One fake detection: a 100x200 box (in 640 space) centred on x=300,
    # i.e. straddling the split at x=320 of an 800x800 image (scale 0.8).
    out = np.zeros((1, 84, 8400), np.float32)
    out[0, :4, 0] = [300, 300, 100, 200]
    out[0, 4, 0] = 0.9

    class Fake:
        def setInput(self, blob):
            pass

        def forward(self):
            return out

    monkeypatch.setattr(splitter, "_yolo_net", Fake())
    left, right = splitter._person_areas(np.zeros((800, 800, 3), np.uint8), 400)
    # Box spans x 312.5..437.5, height 250 in original pixels.
    assert left == pytest.approx(87.5 * 250)
    assert right == pytest.approx(37.5 * 250)


def test_cpu_work_does_not_block_event_loop(monkeypatch):
    png = cv2.imencode(".png", _poster("left"))[1].tobytes()

    async def fake_fetch(url):
        return png

    def slow_render(*args):
        time.sleep(1.0)
        return png

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
