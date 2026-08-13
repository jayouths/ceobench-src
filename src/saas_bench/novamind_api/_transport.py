"""Local HTTP transport shared by the NovaMind CLI and Python SDK."""

import http.client
import json
import os
import socket
from typing import Any, Dict, Optional


class HTTPTransportError(Exception):
    """HTTP response with a non-success status code."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body
        try:
            self.payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.payload = None
        super().__init__(f"HTTP {status}")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: Optional[float]):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def request_json(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[float] = None,
    socket_path: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """Send one JSON request over the configured local transport."""
    socket_path = socket_path or os.environ.get("NOVAMIND_API_SOCKET")
    if socket_path:
        connection = _UnixHTTPConnection(socket_path, timeout)
    else:
        port = port or int(os.environ.get("NOVAMIND_API_PORT", "0"))
        if not port:
            raise ConnectionError(
                "NOVAMIND_API_SOCKET or NOVAMIND_API_PORT must be configured"
            )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    payload = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
    finally:
        connection.close()

    if response.status >= 400:
        raise HTTPTransportError(response.status, response_body)
    try:
        return json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("API server returned invalid JSON") from exc
