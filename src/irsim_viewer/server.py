"""A local web server for the frame viewer: the Python standard library and NumPy, nothing else.

``python -m irsim_viewer outputs/`` (or ``make viewer RUN=outputs``) serves one page on
``127.0.0.1`` and opens it in a browser. The page asks this server four questions, all JSON
except the image:

==============================================  ==============================================
``GET /api/runs``                               every run found under the root
``GET /api/frames?run=R``                       its frames, and each frame's planes and metadata
``GET /api/image?run=R&frame=F&plane=P``        a plane as a PNG (8-bit planes as they are;
                                                 anything else stretched to gray for viewing)
``GET /api/probe?run=R&frame=F&plane=P&pts=..``  every plane at each ``x,y`` of ``pts``
                                                 (``x1,y1;x2,y2``), clicked on plane ``P``
==============================================  ==============================================

``pts`` takes several points because every marker is re-read when the frame changes: stepping a
clip keeps the markers where they are and refreshes what is under them.

Nothing here reads a file: runs come from :func:`irsim_viewer.source.discover_sources`, so a new
data layout is a new :class:`~irsim_viewer.source.FrameSource`, not a server change. The server
binds to loopback only and resolves every name through the discovered runs, so a request cannot
reach a path the root does not contain.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from numpy.typing import NDArray

from irsim.io.png import png_bytes
from irsim_viewer.probe import probe
from irsim_viewer.source import FrameSource, discover_sources

__all__ = ["ViewerApp", "make_server", "main", "display_png"]

STATIC = pathlib.Path(__file__).resolve().parent / "static"


def display_png(plane: NDArray[Any]) -> bytes:
    """Encode a plane for looking at: uint8 as-is, anything else stretched 0.5-99.5 % to gray.

    The stretch is for finding things on screen, never for reading a value off -- the probe reads
    the plane itself. Non-finite pixels are drawn black.
    """
    arr = np.asarray(plane)
    if arr.dtype == np.uint8 and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))):
        return png_bytes(np.ascontiguousarray(arr))
    if arr.ndim == 3:
        arr = arr[..., 0]
    values = arr.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return png_bytes(np.zeros(values.shape, dtype=np.uint8))
    lo, hi = np.percentile(values[finite], [0.5, 99.5])
    span = hi - lo if hi > lo else 1.0
    gray = np.clip((values - lo) / span, 0.0, 1.0) * 255.0
    return png_bytes(np.where(finite, np.round(gray), 0).astype(np.uint8))


class ViewerApp:
    """The runs under one root, and the answers to the page's questions about them."""

    def __init__(self, root: pathlib.Path, sources: list[FrameSource] | None = None) -> None:
        self.root = pathlib.Path(root)
        self._sources = sources
        self._lock = threading.Lock()

    def sources(self) -> dict[str, FrameSource]:
        """Discovered once, on first use; restart the server to pick up a new run."""
        with self._lock:
            if self._sources is None:
                self._sources = discover_sources(self.root)
            return {s.name: s for s in self._sources}

    def source(self, run: str) -> FrameSource:
        try:
            return self.sources()[run]
        except KeyError:
            raise KeyError(f"no run {run!r} under {self.root}") from None

    def runs(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "runs": [{"name": n, "frames": len(s.frames())} for n, s in self.sources().items()],
        }

    def frames(self, run: str) -> dict[str, Any]:
        source = self.source(run)
        out = []
        for frame in source.frames():
            planes = source.planes(frame)
            out.append(
                {
                    "name": frame,
                    "metadata": source.metadata(frame),
                    "planes": [
                        {
                            "key": p.key,
                            "unit": p.unit,
                            "dtype": p.dtype,
                            "shape": list(p.shape),
                            "image": p.is_image,
                        }
                        for p in planes.values()
                    ],
                }
            )
        return {"run": run, "frames": out}

    def image(self, run: str, frame: str, plane: str) -> bytes:
        return display_png(self.source(run).read_plane(frame, plane))

    def probe(self, run: str, frame: str, plane: str, pts: str) -> dict[str, Any]:
        points = []
        for chunk in filter(None, pts.split(";")):
            x, y = (int(float(v)) for v in chunk.split(","))
            points.append(probe(self.source(run), frame, x, y, plane))
        return {"points": points}


def _handler(app: ViewerApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    page = (STATIC / "index.html").read_bytes()
                    self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self._json(app.runs())
                elif url.path == "/api/frames":
                    self._json(app.frames(q["run"]))
                elif url.path == "/api/image":
                    body = app.image(q["run"], q["frame"], q["plane"])
                    self._send(HTTPStatus.OK, body, "image/png")
                elif url.path == "/api/probe":
                    self._json(app.probe(q["run"], q["frame"], q["plane"], q.get("pts", "")))
                else:
                    self._json({"error": f"no such path {url.path}"}, HTTPStatus.NOT_FOUND)
            except KeyError as err:
                self._json({"error": str(err).strip("'\"")}, HTTPStatus.NOT_FOUND)
            except (ValueError, OSError) as err:
                self._json({"error": str(err)}, HTTPStatus.BAD_REQUEST)

    return Handler


def make_server(app: ViewerApp, port: int = 8765) -> ThreadingHTTPServer:
    """Bind the viewer to loopback. ``port=0`` picks a free port (``server.server_address``)."""
    return ThreadingHTTPServer(("127.0.0.1", port), _handler(app))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m irsim_viewer",
        description="Click on rendered IR frames and read every saved per-pixel plane.",
    )
    parser.add_argument("root", nargs="?", default="outputs", help="a run, or a folder of runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    app = ViewerApp(root)
    runs = app.sources()
    if not runs:
        parser.error(f"no frames with sidecars under {root} (a render writes them via FrameWriter)")
    server = make_server(app, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"irsim viewer: {len(runs)} run(s) under {root}  ->  {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
