"""Download remote images with SSRF protection, MIME sniffing, and size limits."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import magic

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
FETCH_TIMEOUT = 10.0
MAX_REDIRECTS = 5

# Normalize aliases for comparison
CONTENT_TYPE_CANONICAL = {
    "image/jpg": "image/jpeg",
    "image/jpeg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}

MAGIC_TO_MIME = {
    "image/jpeg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}


class FetcherError(Exception):
    """Base error for fetch failures with HTTP status + error payload."""

    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(detail)

    def as_dict(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


def _normalize_content_type(header: str | None) -> str | None:
    if not header:
        return None
    # Strip parameters: "image/jpeg; charset=binary" → "image/jpeg"
    mime = header.split(";")[0].strip().lower()
    return CONTENT_TYPE_CANONICAL.get(mime)


def _sniff_mime(data: bytes) -> str | None:
    """Detect MIME from magic bytes; prefer python-magic, fall back to signatures."""
    try:
        detected = magic.from_buffer(data, mime=True)
        if detected in MAGIC_TO_MIME:
            return MAGIC_TO_MIME[detected]
    except Exception:
        pass

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_and_check_host(hostname: str) -> None:
    """Resolve hostname and reject private/loopback/link-local addresses (SSRF)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise FetcherError(
            502,
            "fetch_failed",
            f"Could not resolve host: {hostname}",
        ) from exc

    if not infos:
        raise FetcherError(502, "fetch_failed", f"Could not resolve host: {hostname}")

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if _is_private_ip(ip):
            raise FetcherError(
                400,
                "invalid_url",
                "URL resolves to a private or reserved IP address.",
            )


def validate_url(url: str) -> str:
    """Validate URL scheme/host and run SSRF IP checks. Returns the cleaned URL."""
    if not url or not url.strip():
        raise FetcherError(400, "invalid_url", "Missing required query parameter: url")

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise FetcherError(400, "invalid_url", "Malformed URL.") from exc

    if parsed.scheme not in ("http", "https"):
        raise FetcherError(
            400,
            "invalid_url",
            "Only http and https URLs are allowed.",
        )

    if not parsed.hostname:
        raise FetcherError(400, "invalid_url", "URL must include a hostname.")

    # Reject literal private IPs in the URL itself
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        if _is_private_ip(ip):
            raise FetcherError(
                400,
                "invalid_url",
                "URL targets a private or reserved IP address.",
            )
    except ValueError:
        # Hostname is not a literal IP — resolve and check
        _resolve_and_check_host(parsed.hostname)

    return url


async def fetch_image(url: str) -> bytes:
    """
    Stream-download an image with Content-Length + mid-stream size caps,
    dual MIME validation (header + magic sniff), and SSRF guards.
    """
    try:
        url = validate_url(url)

        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=False,
        ) as client:
            for hop in range(MAX_REDIRECTS + 1):
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        if hop == MAX_REDIRECTS:
                            raise FetcherError(
                                502, "fetch_failed", "Too many redirects."
                            )
                        location = response.headers.get("location")
                        if not location:
                            raise FetcherError(
                                502,
                                "fetch_failed",
                                "Redirect response missing Location header.",
                            )
                        # Resolve + SSRF-validate the NEXT hop's host
                        # BEFORE we ever open a connection to it.
                        url = validate_url(str(response.url.join(location)))
                        continue

                    if response.status_code < 200 or response.status_code >= 300:
                        raise FetcherError(
                            502,
                            "fetch_failed",
                            f"Source URL returned HTTP {response.status_code}.",
                        )

                    declared = _normalize_content_type(
                        response.headers.get("content-type")
                    )
                    if declared is None or declared not in CONTENT_TYPE_CANONICAL.values():
                        raise FetcherError(
                            400,
                            "invalid_image",
                            "Content-Type is not a valid JPEG/PNG/WEBP image.",
                        )

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            length = int(content_length)
                        except ValueError:
                            length = -1
                        if length > MAX_IMAGE_BYTES:
                            raise FetcherError(
                                413,
                                "invalid_image",
                                f"File exceeds maximum allowed size of "
                                f"{MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
                            )

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise FetcherError(
                                413,
                                "invalid_image",
                                f"File exceeds maximum allowed size of "
                                f"{MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
                            )
                        chunks.append(chunk)

                    data = b"".join(chunks)
                    break

    except FetcherError:
        raise
    except httpx.TimeoutException as exc:
        raise FetcherError(
            502,
            "fetch_failed",
            "Timed out while fetching the source URL.",
        ) from exc
    except httpx.HTTPError as exc:
        raise FetcherError(
            502,
            "fetch_failed",
            f"Could not fetch the source URL: {exc}",
        ) from exc
    except Exception as exc:
        # Catches httpx exceptions that don't inherit from HTTPError
        # (e.g. InvalidURL, CookieConflict, StreamConsumed/StreamClosed)
        # plus any other unexpected failure during the fetch, so callers
        # always get a structured FetcherError instead of a raw 500.
        raise FetcherError(
            502,
            "fetch_failed",
            f"Unexpected error while fetching the source URL: {exc}",
        ) from exc

    if not data:
        raise FetcherError(400, "invalid_image", "Downloaded file is empty.")

    sniffed = _sniff_mime(data)
    if sniffed is None:
        raise FetcherError(
            400,
            "invalid_image",
            "Content is not a valid JPEG/PNG/WEBP image.",
        )

    if sniffed != declared:
        raise FetcherError(
            400,
            "invalid_image",
            "Content-Type header does not match the sniffed image type.",
        )

    return data
