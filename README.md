# Poster Splitter

Stateless FastAPI service that fetches a landscape double-DVD combo poster, crops one portrait panel, and streams the PNG back in the response. Nothing is written to disk.

## Endpoints

### `GET /poster`

| Param     | Required | Default | Description                                      |
|-----------|----------|---------|--------------------------------------------------|
| `url`     | yes      | —       | Source image URL (`http` / `https`)              |
| `side`    | no       | `auto`  | `left` \| `right` \| `auto`                      |
| `midline` | no       | `0.5`   | Vertical split as fraction of width (`0`–`1`)    |

- **200** — `Content-Type: image/png` (cropped panel)
- **400** — bad/missing URL, invalid image / MIME mismatch
- **413** — file or decoded dimensions too large
- **422** — aspect ratio is not a landscape double-poster
- **502** — could not fetch the source URL

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

## Safeguards

- SSRF: rejects private / loopback / link-local IPs (including after redirects)
- Dual MIME check: `Content-Type` header + magic-byte sniff
- Streaming download with 10 MB hard cap
- Decoded pixel cap (~40 megapixels)
- Aspect ratio gate (~0.9–2.2) for double-poster layout

## Auto side selection

1. Haar cascade face detection on each half — prefer the half with faces
2. If both/neither have faces, pick by Canny edge density + Laplacian variance
