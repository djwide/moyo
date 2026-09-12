"""HTTP wrapper so the same Docker image can serve /exposure previews.

Cloud Run Job entrypoint stays ``python cloud_worker.py``.
A Cloud Run Service can override the command to::

    python -m moyo.exposure_preview_http

The storefront calls this when ``EXPOSURE_PREVIEW_URL`` is set.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from moyo.exposure_preview import estimate_exposure_preview


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        origin = self.headers.get("Origin") or ""
        if origin.endswith("senteguard.com") or origin.startswith("http://localhost"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        origin = self.headers.get("Origin") or ""
        if origin.endswith("senteguard.com") or origin.startswith("http://localhost"):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._send(200, {"ok": True, "service": "moyo-exposure-preview"})
            return
        self._send(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/", "/preview", "/exposure-preview"}:
            self._send(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > 4096:
            self._send(413, {"error": "Topic is too long."})
            return
        try:
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            topic = str(body.get("topic") or "").strip()
            preview = estimate_exposure_preview(topic)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception:
            self._send(400, {"error": "Invalid request."})
            return
        self._send(200, preview)


def main() -> int:
    host = os.environ.get("MOYO_PREVIEW_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT") or os.environ.get("MOYO_PREVIEW_PORT") or "8080")
    server = ThreadingHTTPServer((host, port), PreviewHandler)
    print(f"moyo exposure preview listening on {host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
