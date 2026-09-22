# ADR 0116 — The gain state is a radiance ceiling, and it is also the transfer

**Status:** Accepted. Completes the fire lane on the camera side; the scene side is
[ADR 0115](0115-fire-authored-by-what-it-emits.md) and the flame's own radiance is
[ADR 0098](0098-participating-media-and-the-phenomena-tier.md).
**Date:** 2026-09-22
Roadmap: PH.8 (§9.2, §11.3)

## Context

A thermal camera does not have one dynamic range; it has a **gain state**. FLIR's Boson (Rev 340)
publishes two — high gain holds a scene to about 140 °C, low gain to about 500 °C, both behind the
same 16-bit converter — and switching between them is a decision about what the scene contains.
Every camera in this repository has run, in effect, with no ceiling at all. That was fine until
`PH.7` put fire in a scene, at which point what a microbolometer does with a 1000 °C object stops
being radiometry and becomes a camera question with three parts: where the ceiling is, what units
it is in, and what the rail does to everything else in frame.

## Options considered

**No ceiling: let the converter rail where it rails.** The status quo. It gets the *existence* of a
rail right and its *position* wrong by whatever the radiometric range happens to be, which is a
constructor default nobody chose for this reason. Rejected: the position is the whole point — it
is what distinguishes the two states of a real camera.

**A ceiling on the scene temperature.** The obvious implementation and the one this ADR exists to
refuse. See below.

**A ceiling on the signal in DN.** Physically where a well saturates, and it would come free from
`quantise`. Rejected as the *only* mechanism because the datasheet quotes the ceiling as a scene
temperature; expressing it in DN would mean re-deriving it through the transfer at the point of
use, and the two could then disagree. Taken *as well*, indirectly — see the second decision.

## Decision

**The ceiling is a radiance, applied to the at-aperture plane, and never to a temperature.**

For one blackbody pixel the two are the same operation: `L_B` is monotone, so clipping `T` at
500 °C and clipping `L` at `L_B(500 °C)` pick the same pixels. That is exactly why the mistake
survives review — every test written against a blackbody passes either way. They part company the
moment a pixel is a *scene*, because what arrives is `ε L_B(T) + (1 − ε) L_env` plus what the path
adds, and that is not `L_B` of anything. Measured on the shipped Boson, a 1273 K surface under the
low-gain ceiling:

| ε | radiance ceiling | temperature ceiling |
|---|---|---|
| 1.0 | rails | rails |
| 0.6 | **rails** | **does not rail** — the frame reads merely hot |
| 0.4 | 663 K, unsaturated | **544 K** — 119 K of error on a pixel that never reached the ceiling |

Both failures are silent. An image with a saturated flame in it looks like an image with a
saturated flame in it, and 544 K where 663 K belongs is a plausible number on an object nobody has
a second measurement of. The ε = 0.4 row is the worse of the two: a clip that should not have
applied at all changed a pixel by more than a hundred kelvin.

**The gain state is also the transfer.** A Boson's two states share one converter and differ in how
much scene they map onto it, so `gain_ceiling_k` becomes the top of the camera's radiometric range
as well as the clip level. One number, so the clip and the rail cannot disagree: a clipped pixel
lands exactly on the top code rather than somewhere arbitrary below it, and the 16-bit signal rails
for the physical reason rather than by a second mechanism. Measured: a 400 °C object — chosen
because it sits *between* the Boson's two ceilings — rails high gain at DN 65535 reading 413 K, and
reads 668 K at DN 45 780 in low gain. Same scene, same camera, and the state decides whether the
frame is usable.

**The rail does not reach the measurement.** The AGC is a display stretch and the radiometric branch
inverts the ADC plane upstream of it (ADR 0031), so `apparent_t` is bit-identical across AGC modes.
The rail is the only cap on a measurement, which is what gives FLAME 3's radiometric histogram its
shape: an ambient mode and a tail piled *against* 500 °C rather than spread below it.

**What the rail does reach is the picture, and the number is large.** The same person against the
same room, on the same camera:

| | no fire in frame | 1273 K fire in frame |
|---|---|---|
| `agc_linear` | 252 DN of separation | **1.0 DN** |
| `agc_plateau` | — | **90 DN** |

FLIR quote about 0.7 % of range, which is 1.8 DN; this reproduces it. The linear stretch is not a
bad one — it is a perfectly good one being asked to span a thousand kelvin. Plateau equalisation
spends its output range on occupied bins instead of on the empty gap between the room and the fire,
which is the entire argument for the mode, and now it is a measurement rather than an assertion.

## Consequences

**Easy now:** a fire scene renders on a camera that behaves like a camera, and the AGC choice has a
measured cost. `XD.5` can compare against FLAME 3's actual histogram when the dataset is in hand;
until then `tests/unit/test_gain_state.py` pins the *shape* a low-gain radiometric camera produces
from FLAME 3's published description, and says so rather than implying an agreement.

**Error introduced:** the clip is applied to the scene-radiance plane before the PSF, where a real
core blurs and then saturates. The difference is confined to the optical skirt of a railed object —
a railed flame does not bloom quite as far as it would on a real core. It is bounded by the PSF's
own width, which for these optics is under a native pixel.

**A schema field costs a golden regeneration, even at its default.** `config_hash` covers the
serialised sensor config, so adding `gain_ceiling_k` moved it on 23 stored sidecars although every
one of them has the field as `null` and takes the same code path. The guard caught it and said so
in the right words — "the inputs that produced the reference changed, so this is not a physics
failure" — and the arrays were verified **bit-identical** before `make golden-update` ran, which is
the only thing that makes a regeneration honest. Worth knowing before the next optional field.

**Not done:** no shipped sensor config declares a gain state. `gain_ceiling_k` defaults to `None`,
which is the behaviour every camera in this repository had before, and adding one to
`flir_boson_640_lwir.yaml` would move the radiometric range's top from 473 K to 413 K and rewrite
every golden that camera has — to model a state no scene yet needs. It belongs with the first fire
scene, alongside the `fire:` schema block ADR 0115 left for the same reason.

## Revisit when

A camera in a scene switches gain state *during* a run, which is what a real core does when a fire
enters frame — the transfer changes between frames and the NUC, the FFC and the temporal filter all
have to be told. Or a photon FPA needs the same treatment, where the ceiling is the well in
electrons rather than a scene temperature and the two do not map onto each other as cleanly.
