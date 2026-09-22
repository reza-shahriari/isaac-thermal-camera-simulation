# ADR 0123 — The camera moves so the aircraft can keep its shadows

**Status:** Accepted. First rendered point-wise aerial scene: a scene config plus one command
produces float32 frames whose target carries a gradient across a single prim, against sky, at a
range that changes.
**Date:** 2026-09-22
Roadmap: PT.9 (the in-engine half), IG.2 (§6.6, §15 T3)

## Context

`PT.9` shipped the point-wise aerial airframe engine-free: `quad_flight_pointwise.yaml` solves a
deck, a belly and two arms per cell, and `scripts/quad_flight_pointwise.py` draws them over a
synthetic G-buffer. Nothing rendered it. Two things stood between that scene and a frame, and
neither was in the physics:

1. **The prims did not exist.** The patches bind to `/World/Quad/deck`, `/belly`, `/arm_n`,
   `/arm_e`; the Isaac quadrotor (`irsim_isaac.quadrotor`) authors `body`, `battery`, `arm_0..3`,
   `motor_0..3`, `esc_0..3`, on a different airframe, in a different frame. A patch that misses
   its prim does not fail — `PointwiseTemperature` leaves those pixels at the per-prim fallback,
   and part of an airframe renders as a field and part as a flat value.
2. **The scene is ENU and the stage is Y-up.** The pad-frame coordinates in that file are not
   coordinates of anything on a Kit stage.

And one thing was in the physics: **`quad_flight`'s camera is fixed and its aircraft flies**, which
is the one arrangement a point-wise target cannot use.

## Decision

**A second scene, `quad_outbound_pointwise.yaml`, authored in the stage frame**, with
`world_frame: {up: [0,1,0], north: [0,0,-1]}` so the sun is still placed from the site and the
clock. That is `car_ignition_clear_night.yaml`'s own choice, and it is what lets every patch
coordinate be a coordinate of a prim. The ENU scene beside it is left exactly as it is: it is the
engine-free oracle, and moving it would move the numbers `PT.9` was measured against. The port is
checked by measurement, not by inspection — deck − belly is **29.3 K** in both.

**A stage of its own, `irsim_isaac.quad_outbound`**, authoring the aircraft the scene describes:
a 0.84 m frame with deck and belly as separate thin plates, four arms, four bells, four speed
controllers and a pack. `test_quad_outbound.py` walks every patch cell and every occluder face
against the authored prims, so the two files cannot drift apart silently.

**The camera moves along an outbound track and the aircraft does not move at all.**

## Consequences

### This is what keeps the shadows, and the shadows are the whole signature

`OccluderSpec` refuses any frame but `world`, because a shadow cast in a moving frame needs that
frame's pose and the thermal core does not carry one. An aircraft that flies must therefore
express its patches in its own prim's frame (ADR 0087/PT.5 support exactly that) — and loses every
occluder in doing so. What it loses here is **27 K across one arm**: the deck's own shadow on the
inner half, and each pod's shadow on the outer. That gradient is the point-wise signature of this
target, so it is the thing that must not be given up.

Relative motion is all a camera can see. Flying the camera instead gives the same picture — the
target shrinks as 1/R, the slant path in front of it grows — and costs nothing, because the
thermal solve was always going to treat the aircraft's attitude as fixed. In `quad_flight` the
render tilts an airframe the solver believes is level; here the render and the solver agree.

The camera's stage `y` goes well below zero at long range. There is no ground plane, so only
relative geometry exists, and nothing in the pipeline reads a stage coordinate as an altitude —
the sky, the atmosphere and every view cosine are built from each ray's own direction (ADR 0060).

### The boresight is checked against the sensor, and the render is refused if it fails

The aim sits 20° above the horizontal, against a 24.8° vertical field: 7.6° of margin. A longer
lens narrows that field and a shorter one widens it, so the driver computes the field from the
sensor config and **refuses to render** rather than putting ADR 0060's analytic `T_ground` across
the bottom of a sky-target frame. A horizon that is not a horizon is worse than no picture.

### Every patched prim is 2 mm smaller than its patch, and that was measured

The first render of this scene raised on **150 deck pixels**: a prim exactly the size of its patch
loses its edge, because the sampled surface position lands a float's width outside the rectangle.
The inset also stops the plates z-fighting against the body, which in the companion visible frame
is indistinguishable from a fault in the infrared one. 2 mm is a fifth of a pixel at the near end
of the track. `test_every_patched_prim_is_strictly_inside_its_patch` pins it.

### Two fixed spans, because the interesting surfaces are not on one scale

The motors reach tens of kelvin over air while the skin sits within a few; no single linear span
shows both, which is why `render_car_ignition` already carries two. The tight span is anchored on
the **air temperature**: seen from below — which is the only way a ground camera can see a drone
against sky — a quadrotor's largest surface is its belly, and the belly is within a kelvin or two
of the air. The camera's own AGC output is filmed beside both.

### What a ground camera cannot see, and the frame says so

The deck's 29 K excess over air is on the **far side of the aircraft** from a camera looking up at
it. That is not a limitation of the model; it is the answer to the question. An anti-UAV sensor
reads a belly near ambient with four hot bells on it, not a hot deck, and a scene that showed the
deck to a ground camera would be flattering rather than correct. The deck is in the float planes
and in the readout at every frame, so the number is available even where the picture is not.

### The motion plane is not requested

`IG.6`'s `motion_px` reads prim transforms. The aircraft is static and the camera slews, so the
tracker would report zero motion for a target that is crossing the frame. A zero smear asserted
from the wrong premise is worse than no smear declared, so the channel is left unattached (`IG.5`
makes that refusal explicit rather than a default).

### Not done here

The aircraft's attitude is fixed, so there is no pitch into the acceleration as in `quad_flight`.
The powered parts stay per-prim nodes: conducting their heat into the skin from a config still has
no route, which is `PT.9`'s remainder and `TC`'s subject. The arms are flat strips, not tubes —
`WM.7`'s mesh field is the curved answer and `quad_flight_mesh.yaml` already carries it.
