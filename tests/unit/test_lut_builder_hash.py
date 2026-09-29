"""GT.11 -- a LUT bundle knows which builder made it.

`make luts` for AT.24 moved a gitignored MWIR bundle by a uniform 20 % although its grid had not
changed: an older builder had produced it, the config and spectral hashes still matched, and a
golden was recorded against it. The sidecar now carries a hash of the builder's own code inputs
-- the quadrature grid rule, the Planck forms, the constants -- and a bundle whose builder
differs is refused naming `make luts`. Roadmap GT.11; ADR 0012.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import (
    StaleLUTError,
    build_band_lut_for_config,
    builder_sha256,
    load_band_lut_for_config,
    lut_paths,
)


def test_the_builder_hash_is_the_builders_source_and_is_stable() -> None:
    a, b = builder_sha256(), builder_sha256()
    assert a == b and len(a) == 64
    import inspect

    from irsim.radiometry import band_integration

    assert "FINE_GRID_FRACTION" in inspect.getsource(band_integration), "the grid rule is hashed"


def test_a_bundle_records_its_builder_and_one_built_by_another_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    config = load_sensor_config("boson640")
    _, paths = build_band_lut_for_config(config, tmp_path)
    meta = json.loads(paths.sidecar.read_text())
    assert meta["builder_sha256"] == builder_sha256()
    load_band_lut_for_config(config, tmp_path)  # the bundle it just built loads
    meta["builder_sha256"] = "0" * 64
    paths.sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with pytest.raises(StaleLUTError, match="different builder.*make luts"):
        load_band_lut_for_config(config, tmp_path)
    del meta["builder_sha256"]
    paths.sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with pytest.raises(StaleLUTError, match="unrecorded"):
        load_band_lut_for_config(config, tmp_path)
    assert lut_paths(meta["band_hash"], tmp_path).sidecar == paths.sidecar
