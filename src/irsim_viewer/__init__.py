"""Frame viewer: open a rendered frame in a browser, click a pixel, read everything saved under it.

A debugging tool, not part of the physics. Each click drops a coloured marker on the image and
lists, for that pixel, every plane the render saved -- true surface temperature, part, distance,
radiance, apparent temperature, ADC code -- with its unit, plus the frame's own metadata. Markers
stay put while you step through a clip, so one surface can be followed frame to frame.

Layout: :mod:`irsim_viewer.source` is the only module that knows about files (the swap point for
a future data layout), :mod:`irsim_viewer.probe` reads a pixel, :mod:`irsim_viewer.server` answers
the page, and ``static/index.html`` is the page. Standard library and NumPy only, so it runs in
any environment the core runs in, with no GUI toolkit installed.

Separate from ``irsim`` because the core is the physics and this is a tool that reads its output;
it imports the core (for the PNG codec), never the reverse, and never the engine.
"""

from irsim_viewer.probe import map_pixel, probe
from irsim_viewer.source import (
    FrameSource,
    PlaneInfo,
    SidecarDirectorySource,
    discover_sources,
    register_plane_reader,
    register_source_factory,
)

__all__ = [
    "FrameSource",
    "PlaneInfo",
    "SidecarDirectorySource",
    "discover_sources",
    "map_pixel",
    "probe",
    "register_plane_reader",
    "register_source_factory",
]
