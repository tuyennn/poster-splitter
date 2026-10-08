"""Download remote images with SSRF protection, MIME sniffing, and size limits."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import random
import socket

import httpx
import magic

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
FETCH_TIMEOUT = 10.0  # per connect/read/write
FETCH_DEADLINE = 30.0  # whole fetch, all redirects included
MAX_REDIRECTS = 5
# One retry on HTTP 429, waiting the server's Retry-After up to this long.
MAX_RETRY_AFTER = 5.0

# Some image hosts throttle or block non-browser clients (the default
# "python-httpx/x.y" agent gets 429/403). Each fetch picks one of these at
# random and keeps it across redirects; a 429 retry picks another.
USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/26.0 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1",
)

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


_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Unwrap IPv6 forms that carry an IPv4 address (::ffff:127.0.0.1, or the
    # NAT64 prefix) so they get the IPv4 checks.
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64_PREFIX:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    # is_global also rules out shared/CGNAT space (100.64.0.0/10) and the
    # other IANA special-purpose ranges that is_private misses.
    return not ip.is_global or ip.is_multicast


async def _resolve_public_ip(hostname: str) -> str:
    """Resolve hostname once and return an address to connect to. Rejects
    the host if ANY of its addresses is private/loopback/link-local (SSRF)."""
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_private_ip(literal):
            raise FetcherError(
                400, "invalid_url", "URL targets a private or reserved IP address."
            )
        return hostname

    try:
        # The event loop's resolver runs in a thread, so a slow DNS server
        # doesn't block other requests.
        infos = await asyncio.get_running_loop().getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        )
    except (socket.gaierror, UnicodeError) as exc:
        raise FetcherError(
            502, "fetch_failed", f"Could not resolve host: {hostname}"
        ) from exc

    addresses: list[str] = []
    for info in infos:
        ip_str = info[4][0]
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
        addresses.append(ip_str)

    if not addresses:
        raise FetcherError(502, "fetch_failed", f"Could not resolve host: {hostname}")
    return addresses[0]


def validate_url(url: str) -> httpx.URL:
    """Validate URL scheme and host (no network access)."""
    if not url or not url.strip():
        raise FetcherError(400, "invalid_url", "Missing required query parameter: url")

    try:
        parsed = httpx.URL(url.strip())
    except Exception as exc:
        raise FetcherError(400, "invalid_url", "Malformed URL.") from exc

    if parsed.scheme not in ("http", "https"):
        raise FetcherError(
            400,
            "invalid_url",
            "Only http and https URLs are allowed.",
        )

    if not parsed.host:
        raise FetcherError(400, "invalid_url", "URL must include a hostname.")

    return parsed


def _browser_headers(previous_agent: str | None = None) -> dict[str, str]:
    """Headers a browser sends when loading an image, with a random
    User-Agent (a different one from previous_agent, if given)."""
    agents = [ua for ua in USER_AGENTS if ua != previous_agent] or list(USER_AGENTS)
    return {
        "User-Agent": random.choice(agents),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _retry_after_seconds(header: str | None) -> float:
    """Seconds to wait from a Retry-After header, capped at MAX_RETRY_AFTER.
    HTTP-date values and junk fall back to 1 second."""
    try:
        seconds = float(header) if header is not None else 1.0
    except ValueError:
        seconds = 1.0
    return min(max(seconds, 0.0), MAX_RETRY_AFTER)


async def _open_pinned(
    client: httpx.AsyncClient, url: httpx.URL, headers: dict[str, str]
) -> httpx.Response:
    """Send GET for url, connecting to the IP address that was just checked.

    Letting httpx resolve the name again would open a DNS-rebinding hole:
    the first lookup could return a public IP and the second 127.0.0.1 or
    169.254.169.254. So the URL is rewritten to the vetted IP, while the
    Host header and TLS SNI/certificate check keep the original name."""
    hostname = url.raw_host.decode("ascii")
    ip = await _resolve_public_ip(hostname)
    extensions = {"sni_hostname": hostname} if url.scheme == "https" else {}
    request = client.build_request(
        "GET",
        url.copy_with(host=ip),
        headers={
            **headers,
            "Host": url.netloc.decode("ascii"),
            # Hotlink protection often wants a same-site Referer.
            "Referer": f"{url.scheme}://{url.netloc.decode('ascii')}/",
        },
        extensions=extensions,
    )
    return await client.send(request, stream=True)


def _new_client() -> httpx.AsyncClient:
    # trust_env=False: an HTTP(S)_PROXY from the environment would resolve
    # the hostname itself and bypass the IP pinning above.
    return httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=False, trust_env=False)


async def fetch_image(url: str) -> bytes:
    """
    Stream-download an image with Content-Length + mid-stream size caps,
    dual MIME validation (header + magic sniff), and SSRF guards.
    """
    try:
        target = validate_url(url)

        # FETCH_TIMEOUT applies per network operation; this caps the whole
        # download so a server dripping one byte at a time can't hold a
        # request open indefinitely.
        async with asyncio.timeout(FETCH_DEADLINE), _new_client() as client:
            headers = _browser_headers()
            hop = 0
            retried = False
            while True:
                response = await _open_pinned(client, target, headers)
                try:
                    if response.status_code == 429 and not retried:
                        retried = True
                        delay = _retry_after_seconds(response.headers.get("retry-after"))
                        await asyncio.sleep(delay)
                        headers = _browser_headers(previous_agent=headers["User-Agent"])
                        continue

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
                        # Join against the original name, not the pinned IP;
                        # the next hop is validated and pinned the same way.
                        target = validate_url(str(target.join(location)))
                        hop += 1
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
                finally:
                    await response.aclose()

    except FetcherError:
        raise
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise FetcherError(
            502,
            "fetch_failed",
            "Timed out while fetching the source URL.",
        ) from exc
    except Exception as exc:
        # httpx errors and anything else unexpected. The exception text can
        # name internal hosts or library details, so it goes to the log, not
        # to the client.
        logger.warning("Fetch failed for %s: %r", url, exc)
        raise FetcherError(
            502,
            "fetch_failed",
            "Could not fetch the source URL.",
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
