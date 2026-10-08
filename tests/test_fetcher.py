import asyncio
import ipaddress
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import httpx
import numpy as np
import pytest

from app import fetcher

PNG = cv2.imencode(".png", np.zeros((30, 40, 3), np.uint8))[1].tobytes()
FAKE_HOST = "poster.test"
REBOUND_IP = "10.66.66.66"


@pytest.mark.parametrize(
    "addr, private",
    [
        ("8.8.8.8", False),
        ("2606:4700::1111", False),
        ("127.0.0.1", True),
        ("10.0.0.1", True),
        ("169.254.169.254", True),
        ("100.64.0.1", True),  # CGNAT, not covered by is_private
        ("0.0.0.0", True),
        ("224.0.0.1", True),
        ("::1", True),
        ("fd00::1", True),
        ("::ffff:127.0.0.1", True),
        ("::ffff:8.8.8.8", False),
        ("64:ff9b::7f00:1", True),  # NAT64 -> 127.0.0.1
    ],
)
def test_private_ip_check(addr, private):
    assert fetcher._is_private_ip(ipaddress.ip_address(addr)) is private


def test_literal_private_ip_rejected():
    with pytest.raises(fetcher.FetcherError) as exc:
        asyncio.run(fetcher.fetch_image("http://169.254.169.254/latest/meta-data"))
    assert exc.value.status_code == 400


class _Handler(BaseHTTPRequestHandler):
    seen_headers: list = []
    throttle_first = False

    def do_GET(self):
        _Handler.seen_headers.append(dict(self.headers))
        if _Handler.throttle_first and len(_Handler.seen_headers) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/img.png")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_allowed(monkeypatch):
    """Treat 127.0.0.1 as public so a local server can stand in for a real
    host, and make the fake hostname resolve to it exactly once: any later
    lookup gets a private IP, like a DNS-rebinding attack."""
    real_check = fetcher._is_private_ip
    monkeypatch.setattr(
        fetcher, "_is_private_ip", lambda ip: str(ip) != "127.0.0.1" and real_check(ip)
    )
    real_gai = socket.getaddrinfo
    lookups = []

    def fake_gai(host, *args, **kwargs):
        if host == FAKE_HOST:
            lookups.append(host)
            ip = "127.0.0.1" if len(lookups) == 1 else REBOUND_IP
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]
        return real_gai(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    return lookups


def _serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def test_connects_to_vetted_ip_not_a_second_lookup(loopback_allowed):
    server = _serve(ThreadingHTTPServer(("127.0.0.1", 0), _Handler))
    try:
        port = server.server_address[1]
        data = asyncio.run(fetcher.fetch_image(f"http://{FAKE_HOST}:{port}/img.png"))
        assert data == PNG
        assert loopback_allowed == [FAKE_HOST]
    finally:
        server.shutdown()


def test_redirect_hop_is_revalidated(loopback_allowed):
    server = _serve(ThreadingHTTPServer(("127.0.0.1", 0), _Handler))
    try:
        port = server.server_address[1]
        # The second hop resolves to the "rebound" private IP and is refused.
        with pytest.raises(fetcher.FetcherError) as exc:
            asyncio.run(fetcher.fetch_image(f"http://{FAKE_HOST}:{port}/redirect"))
        assert exc.value.status_code == 400
        assert exc.value.error == "invalid_url"
    finally:
        server.shutdown()


def test_https_keeps_hostname_for_sni_and_cert_check(loopback_allowed, tmp_path, monkeypatch):
    key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={FAKE_HOST}",
            "-addext", f"subjectAltName=DNS:{FAKE_HOST}",
        ],
        check=True,
        capture_output=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_ctx.load_cert_chain(cert, key)
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    _serve(server)

    client_ctx = ssl.create_default_context(cafile=str(cert))
    monkeypatch.setattr(
        fetcher,
        "_new_client",
        lambda: httpx.AsyncClient(verify=client_ctx, follow_redirects=False, trust_env=False),
    )
    try:
        port = server.server_address[1]
        # Succeeds only if the certificate is checked against the hostname,
        # not the 127.0.0.1 the connection was pinned to.
        data = asyncio.run(fetcher.fetch_image(f"https://{FAKE_HOST}:{port}/img.png"))
        assert data == PNG
    finally:
        server.shutdown()


@pytest.fixture
def local_server(loopback_allowed, monkeypatch):
    """Local image server; FAKE_HOST always resolves to it (no rebinding)."""
    real_gai = socket.getaddrinfo

    def fake_gai(host, *args, **kwargs):
        if host == FAKE_HOST:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        return real_gai(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    _Handler.seen_headers = []
    _Handler.throttle_first = False
    server = _serve(ThreadingHTTPServer(("127.0.0.1", 0), _Handler))
    yield f"http://{FAKE_HOST}:{server.server_address[1]}"
    server.shutdown()


def test_sends_browser_user_agent(local_server):
    asyncio.run(fetcher.fetch_image(f"{local_server}/redirect"))
    first, second = _Handler.seen_headers
    assert first["User-Agent"] in fetcher.USER_AGENTS
    assert "httpx" not in first["User-Agent"]
    assert first["Accept"].startswith("image/")
    assert first["Referer"] == f"{local_server}/"
    # Same identity across a redirect.
    assert second["User-Agent"] == first["User-Agent"]


def test_user_agent_varies_between_fetches(local_server):
    for _ in range(20):
        asyncio.run(fetcher.fetch_image(f"{local_server}/img.png"))
    assert len({h["User-Agent"] for h in _Handler.seen_headers}) > 1


def test_retries_429_once_with_another_user_agent(local_server):
    _Handler.throttle_first = True
    assert asyncio.run(fetcher.fetch_image(f"{local_server}/img.png")) == PNG
    first, second = _Handler.seen_headers
    assert second["User-Agent"] != first["User-Agent"]


@pytest.mark.parametrize(
    "header, seconds",
    [("2", 2.0), ("0", 0.0), ("600", fetcher.MAX_RETRY_AFTER), (None, 1.0),
     ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0), ("-3", 0.0)],
)
def test_retry_after_parsing(header, seconds):
    assert fetcher._retry_after_seconds(header) == seconds


def test_error_detail_does_not_leak_exception_text(monkeypatch):
    async def boom(*args, **kwargs):
        raise httpx.ConnectError("connect to internal-db.corp:5432 refused")

    monkeypatch.setattr(fetcher, "_open_pinned", boom)
    with pytest.raises(fetcher.FetcherError) as exc:
        asyncio.run(fetcher.fetch_image("http://example.com/a.png"))
    assert "internal-db" not in exc.value.detail
