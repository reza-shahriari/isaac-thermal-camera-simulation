"""A vessel whose deck and deckhouse are solved **per cell**, static in the stage frame (PT.10).

The maritime lane's vessels (`maritime_demo`) carry three temperatures: a hull pinned near the
sea, a deckhouse near air temperature and one very hot funnel. That is the right *coarse*
signature and it is the defect ADR 0087 exists to remove -- a real deck at 16:00 is not one
number, because the deckhouse stands three metres above it and lays a shadow down half its length.

This module is the maritime counterpart of :mod:`irsim_isaac.quad_outbound`, and it makes the same
two choices for the same reasons (ADR 0123):

* **The vessel is static in the stage and the camera moves.** `OccluderSpec` has no pose for a
  moving frame, so a vessel that steams loses its self-shadowing -- which is the whole point-wise
  signature here. Range comes from moving the camera instead.
* **The parts are authored at the scene config's own world coordinates**, with no root transform,
  so every patch stays in the world frame and keeps its occluders.

The boxes come from :func:`irsim_isaac.maritime_demo.vessel_boxes`, not from numbers retyped here:
a patch authored against a different description of the vessel sits off its prim, and the frame
does not fail -- it renders part of a deck as a field and part as a flat value.

One part is added that the demo stage has no use for: a thin **weather deck** plate at the
deckhouse's own base level. A box prim's top face cannot carry a patch on its own, because the
hull's *sides* are the same prim and their pixels would project into the same rectangle and take
the deck's temperature. The quadrotor has the same arrangement for the same reason -- its deck and
belly are separate plates rather than faces of the body.

docs/physics-model.md §6, §13.1; ADR 0087 (the field), ADR 0095 (occluders), ADR 0123 (static
target, moving camera), ADR 0078 (the analytic sea behind it).
"""

from __future__ import annotations

from typing import Any

from irsim_isaac.airframe import Part, author_parts
from irsim_isaac.maritime_demo import vessel_boxes

__all__ = [
    "VESSEL_LENGTH_M",
    "VESSEL_ROOT",
    "DECK_THICKNESS_M",
    "vessel_parts",
    "POINTWISE_VESSEL",
    "deck_plane_y_m",
    "build_pointwise_vessel",
]

#: A 26 m coastal patrol craft -- the `patrol` entry of
#: :data:`~irsim_isaac.maritime_demo.DEMO_VESSELS`, so the point-wise scene and the demo stage
#: describe the same boat.
VESSEL_LENGTH_M = 26.0
VESSEL_ROOT = "/World/Targets/vessel"

#: Plate, not a face: thin enough to be a deck and thick enough to survive a float32 position AOV.
DECK_THICKNESS_M = 0.008

#: The deck plate is inset from the hull's own plan by this much, so the hull is never the prim a
#: deck pixel lands on at the sheer.
_DECK_INSET_M = 0.05


def deck_plane_y_m(length_m: float = VESSEL_LENGTH_M) -> float:
    """Height of the weather deck's upper face: the deckhouse's own base, so the two are flush."""
    (_, house_centre_y, _), (_, house_height, _) = (
        vessel_boxes(length_m)["superstructure"][0],
        vessel_boxes(length_m)["superstructure"][1],
    )
    return float(house_centre_y - 0.5 * house_height)


def vessel_parts(length_m: float = VESSEL_LENGTH_M) -> tuple[Part, ...]:
    """The four prims, in the stage frame the scene config is authored in.

    Materials follow the demo stage's own choices, including the one it had to measure: the funnel
    is **painted**, not bare metal. Bare aluminium is eps = 0.09 in LWIR, so a bare-metal funnel
    reflects the sky instead of radiating and reads cold however hot it is.
    """
    boxes = vessel_boxes(length_m)
    hull_centre, hull_size = boxes["hull"]
    house_centre, house_size = boxes["superstructure"]
    stack_centre, stack_size = boxes["stack"]

    deck_top = deck_plane_y_m(length_m)
    deck_centre = (0.0, deck_top - 0.5 * DECK_THICKNESS_M, 0.0)
    deck_size = (
        hull_size[0] - 2.0 * _DECK_INSET_M,
        DECK_THICKNESS_M,
        hull_size[2] - 2.0 * _DECK_INSET_M,
    )
    return (
        Part("hull", "box", hull_centre, hull_size, "car_paint_white", "hull"),
        Part("weather_deck", "box", deck_centre, deck_size, "car_paint_white", "superstructure"),
        Part(
            "superstructure", "box", house_centre, house_size, "car_paint_white", "superstructure"
        ),
        Part("stack", "box", stack_centre, stack_size, "painted_composite", "stack"),
    )


POINTWISE_VESSEL: tuple[Part, ...] = vessel_parts()


def build_pointwise_vessel(stage: Any, *, parts: tuple[Part, ...] | None = None) -> dict[str, str]:
    """Author the vessel on ``stage`` at its scene-config coordinates; returns prim path -> node.

    No root transform, deliberately: see this module's docstring. The caller supplies the stage so
    this function stays the one piece a render driver needs and the environment, the camera track
    and the water surface stay where they already are (`maritime_demo`).
    """
    from irsim_isaac.stage import bind_visible_look

    return author_parts(
        stage,
        VESSEL_ROOT,
        POINTWISE_VESSEL if parts is None else parts,
        look_binder=bind_visible_look,
    )
