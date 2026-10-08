# Poster Splitter

Stateless FastAPI service that fetches a landscape double-DVD combo poster, crops one portrait panel, and returns the PNG bytes in the response. Nothing is written to disk.

## Endpoints

### `GET /poster`

| Param          | Required | Default | Description                                                                                                                    |
|----------------|----------|---------|------------------------------------------------------------------------------------------------------------------------------------|
| `url`          | yes      | —       | Source image URL (`http` / `https`)                                                                                              |
| `side`         | no       | `right` | `left` \| `right` \| `auto` (largest detected face, else largest person, else visual-interest heuristic)                         |
| `midline`      | no       | `0.5`   | Vertical split position as a fraction of width (`0`–`1`, exclusive)                                                              |
| `spine_trim`   | no       | `0.02`  | Fraction of the returned half's width to trim off its inner edge (nearest the midline), to drop the spine/gutter divider. `0` disables it. |
| `target_ratio` | no       | — (off) | Desired output aspect ratio as `W:H` (e.g. `2:3`) or a decimal (e.g. `0.6667`). Center-crops the panel to this exact ratio after spine trim. Only ever crops — never pads or upscales. |

- **200** — `Content-Type: image/png` (cropped panel)
- **400** — bad/missing URL, invalid image / MIME mismatch, invalid `midline` / `spine_trim` / `target_ratio`
- **413** — file or decoded dimensions too large
- **422** — aspect ratio is neither a single cover nor a double-poster (see [Single covers](#single-covers))

The response carries `X-Poster-Layout: single` or `double` so callers can tell which path ran.
- **502** — could not fetch the source URL
- **500** — unexpected server error (logged with full traceback server-side; response body is a generic JSON error, never a bare error page)

### `GET /health`

Returns `{"status": "ok"}`.

## Run with Docker Compose

```bash
docker compose up --build -d
```

Service listens on `http://localhost:8000`.

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/poster?url=https%3A%2F%2Fexample.com%2Fcombo-cover.jpg&side=auto" -o poster.png
```

To pick up code changes, rebuild and recreate:

```bash
docker compose up --build -d
```

If it still 500s after a rebuild, check the traceback in `docker compose logs` — the global exception handler logs every unhandled error server-side even though the client just sees a generic JSON body.

Stop with:

```bash
docker compose down
```

## Run with Docker

```bash
docker build -t poster-splitter .
docker run -p 8000:8000 poster-splitter

curl "http://localhost:8000/poster?url=https%3A%2F%2Fexample.com%2Fcombo-cover.jpg&side=auto" -o poster.png
```

## Pull from GHCR

Images are published to GitHub Container Registry on pushes to `main`/`master` and on `v*` tags.

```bash
docker pull ghcr.io/<OWNER>/<REPO>:latest
docker run -p 8000:8000 ghcr.io/<OWNER>/<REPO>:latest
```

Or point Compose at the published image:

```yaml
services:
  poster-splitter:
    image: ghcr.io/<OWNER>/<REPO>:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
```

Replace `<OWNER>` / `<REPO>` with your GitHub user or org and repository name (lowercase).

## Local (without Docker)

Needs system `libmagic` (e.g. `libmagic1` on Debian/Ubuntu).

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Make sure only **one** OpenCV package is in `requirements.txt` (`opencv-python-headless` is the right one for a headless server). Having both `opencv-python` and `opencv-python-headless` installed at once corrupts the `cv2` module and breaks detection at runtime with errors like `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.

## Safeguards

- SSRF: rejects any address that isn't globally routable (private, loopback, link-local, CGNAT, IPv4-mapped IPv6, …), both on the original URL and on **every redirect hop** — hops are walked and validated manually (not via httpx's built-in `follow_redirects`) so a malicious redirect target is never connected to
- DNS rebinding: each host is resolved once and the connection goes to that vetted IP (Host header and TLS SNI/certificate check still use the hostname), so a second, different DNS answer is never used. Proxy environment variables are ignored for the same reason
- Dual MIME check: `Content-Type` header + magic-byte sniff
- Streaming download with 10 MB hard cap and a 30 s deadline for the whole fetch
- Pixel cap (~40 megapixels), enforced by OpenCV from the image header before decoding (`OPENCV_IO_MAX_IMAGE_PIXELS`), so a tiny compressed "bomb" is never expanded
- Image work runs in a thread pool, at most `MAX_CONCURRENT_JOBS` at a time, so one large poster doesn't stall other requests or `/health`
- Fetch errors return a generic message; details go to the server log only
- Fetches look like a browser loading an image: a random real-browser `User-Agent` (kept across redirects), image `Accept`, and a same-site `Referer`. A `429` is retried once with a different `User-Agent` after the server's `Retry-After` (capped at 5 s)
- The Docker image runs as a non-root user
- Aspect ratio gate: 0.45–0.95 is treated as a single cover, 0.95–2.2 as a double-poster, anything else is rejected
- Any unexpected exception anywhere in the app is caught by a global handler, logged with a full traceback, and returned as a structured JSON `500` — never a bare error page

## Auto side selection (`side=auto`)

1. Run YuNet face detection (`cv2.FaceDetectorYN`, ONNX) once over the whole poster. If any face is found, pick the half holding the **largest face** (by the face's centre)
2. Otherwise run YOLOv8n person detection (via `cv2.dnn`, ONNX — no `torch` at runtime) once over the whole poster, clip each person box to the half it falls in, and pick the half whose **largest detected person** has the bigger bounding-box area
3. If neither half has a face or a person (or the detectors are unavailable — e.g. missing model file), fall back to a visual-interest heuristic: Canny edge density + Laplacian variance

Requires the model files at `app/models/face_detection_yunet_2023mar.onnx` and `app/models/yolov8n.onnx` (or wherever `FACE_MODEL_PATH` / `YOLO_MODEL_PATH` point — see below). Both are loaded and warmed up at startup. If it's missing or unreadable, detection degrades gracefully to the visual-interest fallback rather than failing the request. If inference fails on one image, only that request falls back.

### Environment variables

| Var               | Default                    | Description                              |
|-------------------|-----------------------------|-------------------------------------------|
| `FACE_MODEL_PATH` | `app/models/face_detection_yunet_2023mar.onnx` | Path to the YuNet ONNX face-detection model (MIT, from opencv_zoo) |
| `YOLO_MODEL_PATH` | `app/models/yolov8n.onnx`  | Path to the ONNX person-detection model  |
| `MAX_CONCURRENT_JOBS` | CPU count | Posters decoded/split/encoded at once (bounds CPU and memory) |

## Cropping pipeline

For a given request, the panel goes through, in order:

1. **Split** at `midline` into left/right halves
2. **Select** a half (`side=left`/`right`/`auto`)
3. **Spine trim** — remove `spine_trim` fraction off the half's inner edge
4. **Ratio crop** (if `target_ratio` given) — center-crop to the exact requested aspect ratio

## Single covers

If the source image is already one portrait cover (width/height from `0.45` up to `0.95`) it is not split. `side`, `midline` and `spine_trim` are ignored, and the image is center-cropped to `target_ratio` if given, otherwise to the single-cover ratio `2:3`. The `0.95` boundary sits near the midpoint between one 2:3 cover (`0.67`) and two side by side (`1.33`), so an image is handled as whichever layout it is closer to.
