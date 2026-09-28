"""The frame viewer (irsim_viewer): a click reads the right pixel of every plane, and says so.

A debugging tool that shows a neighbouring pixel's temperature is worse than none, because the
number looks right. So the tests pin the pixel mapping (same grid, supersampled, non-nesting),
that each plane's value comes back exact and with its unit and legend name, that the server
answers over real HTTP, and that the data layer is swappable -- a source that is not files at all
drives the same server unchanged.
"""

from __future__ import annotations

import ast
import json
import pathlib
import threading
import urllib.request
from typing import Any

import numpy as np
import pytest

from irsim.io.dataset import write_frame
from irsim.io.png import read_png
from irsim_viewer import (
    PlaneInfo,
    SidecarDirectorySource,
    discover_sources,
    map_pixel,
    probe,
    register_plane_reader,
)
from irsim_viewer.server import ViewerApp, display_png, make_server

H, W = 6, 8


class _Outputs:
    def __init__(self, seed: int) -> None:
        rows, cols = np.mgrid[0:H, 0:W]
        self.apparent_t = (280.0 + rows + cols / 10.0 + seed).astype(np.float32)
        self.radiance = (10.0 + rows * W + cols).astype(np.float32)
        self.dn16 = (1000 + rows * W + cols).astype(np.uint16)
        self.display8 = np.zeros((H, W, 4), np.uint8)


def _run(tmp_path: pathlib.Path, frames: int = 2) -> pathlib.Path:
    run = tmp_path / "outputs" / "demo"
    part = np.zeros((H, W), np.uint16)
    part[:, 4:] = 1
    for i in range(frames):
        write_frame(
            run,
            _Outputs(seed=i),
            frame_index=i,
            extra_planes={"part_id": part, "distance_m": np.full((H, W), 4.5, np.float32)},
            legends={"part_id": {0: "(sky)", 1: "motor_front_left"}},
            extra_metadata={"node_temperatures_k": {"motor_front_left": 301.25}, "range_m": 4.5},
        )
    return run


# -- mapping ------------------------------------------------------------------------------------


def test_pixel_mapping_is_exact_for_nesting_grids_and_flagged_otherwise() -> None:
    assert map_pixel(3, 2, (6, 8), (6, 8)) == (3, 2, None, True)
    x, y, note, exact = map_pixel(3, 2, (6, 8), (18, 24))  # 3x supersampled: centre sample
    assert (x, y, exact) == (10, 7, True) and "supersampled" in (note or "")
    assert map_pixel(3, 2, (6, 8), (3, 4))[:2] == (1, 1)  # 2x coarser
    assert map_pixel(3, 2, (6, 8), (5, 8))[3] is False  # does not nest: a warning, not silence


# -- probing ------------------------------------------------------------------------------------


def test_a_click_reads_every_plane_at_that_pixel_with_units_and_names(
    tmp_path: pathlib.Path,
) -> None:
    source = SidecarDirectorySource(_run(tmp_path))
    out = probe(source, "frame_000001", 5, 3, "display8")
    by = {r["key"]: r for r in out["readings"]}
    assert by["apparent_t"]["value"] == pytest.approx(280.0 + 3 + 0.5 + 1)
    assert by["apparent_t"]["celsius"] == pytest.approx(by["apparent_t"]["value"] - 273.15)
    assert by["dn16"]["value"] == 1000 + 3 * W + 5  # read back out of the 16-bit PNG
    assert by["radiance"]["value"] == pytest.approx(10.0 + 3 * W + 5)
    assert by["part_id"]["label"] == "motor_front_left"
    assert by["distance_m"]["unit"].startswith("m")
    assert {"key": "node_temperatures_k[motor_front_left]", "value": 301.25} in out["joins"]
    assert out["warnings"] == []
    left = probe(source, "frame_000001", 1, 3, "display8")
    assert {r["key"]: r for r in left["readings"]}["part_id"]["label"] == "(sky)"


def test_a_click_outside_the_image_is_refused(tmp_path: pathlib.Path) -> None:
    source = SidecarDirectorySource(_run(tmp_path))
    with pytest.raises(ValueError, match="outside"):
        probe(source, "frame_000000", W, 0, "display8")


def test_a_plane_off_the_image_grid_is_read_but_warned_about(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "odd"
    write_frame(run, _Outputs(0), extra_planes={"coarse": np.zeros((5, 8), np.float32)})
    out = probe(SidecarDirectorySource(run), "frame_000000", 7, 5, "display8")
    assert any("coarse" in w and "does not nest" in w for w in out["warnings"])


# -- discovery and the swap point ---------------------------------------------------------------


def test_runs_are_discovered_by_their_sidecars(tmp_path: pathlib.Path) -> None:
    _run(tmp_path, frames=3)
    (tmp_path / "outputs" / "empty").mkdir()
    (tmp_path / "outputs" / "demo" / "summary.json").write_text('{"not": "a sidecar"}')
    sources = discover_sources(tmp_path / "outputs")
    assert [s.name for s in sources] == ["demo"]
    assert sources[0].frames() == ["frame_000000", "frame_000001", "frame_000002"]


def test_a_new_container_is_one_registered_reader(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "custom"
    write_frame(run, _Outputs(0))
    (run / "frame_000000_depth.raw").write_bytes(np.full((H, W), 7.0, np.float32).tobytes())
    doc = json.loads((run / "frame_000000.json").read_text())
    doc["planes"]["depth"] = {"file": "frame_000000_depth.raw", "dtype": "float32",
                              "shape": [H, W], "unit": "m"}  # fmt: skip
    (run / "frame_000000.json").write_text(json.dumps(doc))
    register_plane_reader("raw", lambda p: np.fromfile(p, np.float32).reshape(H, W))
    out = probe(SidecarDirectorySource(run), "frame_000000", 2, 2, "display8")
    assert {r["key"]: r for r in out["readings"]}["depth"]["value"] == 7.0


class _MemorySource:
    """A FrameSource with no files at all -- what a Warp/CUDA buffer source would look like."""

    name = "memory"

    def __init__(self) -> None:
        self.t = np.full((H, W), 300.0, np.float32)
        self.img = np.zeros((H, W), np.uint8)

    def frames(self) -> list[str]:
        return ["f0"]

    def metadata(self, frame: str) -> dict[str, Any]:
        return {"source": "gpu buffer"}

    def planes(self, frame: str) -> dict[str, PlaneInfo]:
        return {
            "view": PlaneInfo("view", "gray", "uint8", (H, W)),
            "temperature_k": PlaneInfo("temperature_k", "K", "float32", (H, W)),
        }

    def read_plane(self, frame: str, key: str) -> np.ndarray:
        return self.img if key == "view" else self.t


def _get(port: int, path: str) -> Any:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        body = r.read()
        return json.loads(body) if r.headers["Content-Type"] == "application/json" else body


def test_the_server_answers_the_page_over_http(tmp_path: pathlib.Path) -> None:
    app = ViewerApp(tmp_path, sources=[_MemorySource(), SidecarDirectorySource(_run(tmp_path))])
    server = make_server(app, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        runs = _get(port, "/api/runs")["runs"]
        # newest first; a source that cannot say when it was written sorts last
        assert [r["name"] for r in runs] == ["demo", "memory"]
        assert all(r["truth"] for r in runs)  # both carry a truth plane
        assert b"<canvas" in _get(port, "/")
        frames = _get(port, "/api/frames?run=demo")["frames"]
        assert frames[0]["metadata"]["range_m"] == 4.5
        points = _get(port, "/api/probe?run=memory&frame=f0&plane=view&pts=1,1;2,3")["points"]
        assert [p["readings"][1]["celsius"] for p in points] == [pytest.approx(26.85)] * 2
        png = _get(port, "/api/image?run=demo&frame=frame_000000&plane=apparent_t")
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(port, "/api/frames?run=nope")
        assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_run_without_truth_planes_is_flagged_camera_only(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "old"
    write_frame(run, _Outputs(0))  # the camera planes only, as every run before ADR 0154
    (row,) = ViewerApp(tmp_path).runs()["runs"]
    assert row["name"] == "old" and row["truth"] is False and row["modified"] > 0


def test_a_float_plane_is_stretched_for_display_only(tmp_path: pathlib.Path) -> None:
    ramp = np.linspace(250.0, 350.0, H * W, dtype=np.float32).reshape(H, W)
    ramp[0, 0] = np.nan
    path = tmp_path / "v.png"
    path.write_bytes(display_png(ramp))
    gray = read_png(path)
    assert gray.dtype == np.uint8 and gray[0, 0] == 0 and gray[-1, -1] == 255


def test_the_viewer_imports_no_engine_or_gui_toolkit() -> None:
    forbidden = {"omni", "pxr", "isaacsim", "warp", "carb", "irsim_isaac", "PyQt5", "PySide6",
                 "tkinter", "torch", "cv2", "PIL"}  # fmt: skip
    package = pathlib.Path(__file__).resolve().parents[2] / "src" / "irsim_viewer"
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not roots & forbidden, f"{path.name} imports {roots & forbidden}"
