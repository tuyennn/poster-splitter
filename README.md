# Poster Splitter

Stateless FastAPI service that fetches a landscape double-DVD combo poster, crops one portrait panel, and returns the PNG bytes in the response. Nothing is written to disk.

## Endpoints

### `GET /poster`

| Param          | Required | Default | Description                                                                                                                    |
|----------------|----------|---------|------------------------------------------------------------------------------------------------------------------------------------|
| `url`          | yes      | —       | Source image URL (`http` / `https`)                                                                                              |
| `side`         | no       | `auto`  | `left` \| `right` \| `auto` (largest detected person, else visual-interest heuristic)                                            |
| `midline`      | no       | `0.5`   | Vertical split position as a fraction of width (`0`–`1`, exclusive)                                                              |
| `spine_trim`   | no       | `0.02`  | Fraction of the returned half's width to trim off its inner edge (nearest the midline), to drop the spine/gutter divider. `0` disables it. |
| `target_ratio` | no       | — (off) | Desired output aspect ratio as `W:H` (e.g. `2:3`) or a decimal (e.g. `0.6667`). Center-crops the panel to this exact ratio after spine trim. Only ever crops — never pads or upscales. |

- **200** — `Content-Type: image/png` (cropped panel)
- **400** — bad/missing URL, invalid image / MIME mismatch, invalid `midline` / `spine_trim` / `target_ratio`
- **413** — file or decoded dimensions too large
- **422** — aspect ratio is not a landscape double-poster
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

Make sure only **one** OpenCV package is in `requirements.txt` (`opencv-python-headless` is the right one for a headless server). Having both `opencv-python` and `opencv-python-headless` installed at once corrupts the `cv2` module and breaks detection at runtime with errors like `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.

## Safeguards

- SSRF: rejects private / loopback / link-local IPs, both on the original URL and on **every redirect hop** — hops are walked and validated manually (not via httpx's built-in `follow_redirects`) so a malicious redirect target is never connected to
- Dual MIME check: `Content-Type` header + magic-byte sniff
- Streaming download with 10 MB hard cap
- Decoded pixel cap (~40 megapixels)
- Aspect ratio gate (~0.9–2.2) for double-poster layout
- Any unexpected exception anywhere in the app is caught by a global handler, logged with a full traceback, and returned as a structured JSON `500` — never a bare error page

## Auto side selection (`side=auto`)

1. Run YOLOv8n person detection (via `cv2.dnn`, ONNX — no `torch` at runtime) on each half
2. Pick the half whose **largest detected person** has the bigger bounding-box area
3. If neither half has a detected person (or the detector is unavailable — e.g. missing model file), fall back to a visual-interest heuristic: Canny edge density + Laplacian variance

Requires the model file at `app/models/yolov8n.onnx` (or wherever `YOLO_MODEL_PATH` points — see below). If missing or unreadable, detection degrades gracefully to the visual-interest fallback rather than failing the request.

### Environment variables

| Var               | Default                    | Description                              |
|-------------------|-----------------------------|-------------------------------------------|
| `YOLO_MODEL_PATH` | `app/models/yolov8n.onnx`  | Path to the ONNX person-detection model  |

## Cropping pipeline

For a given request, the panel goes through, in order:

1. **Split** at `midline` into left/right halves
2. **Select** a half (`side=left`/`right`/`auto`)
3. **Spine trim** — remove `spine_trim` fraction off the half's inner edge
4. **Ratio crop** (if `target_ratio` given) — center-crop to the exact requested aspect ratio
