# ADR 0194 — A person's skin is solved segment by segment by JOS-3, installed and not rewritten, on the scene's weather

**Status:** Accepted
**Date:** 2026-10-06
Roadmap: HU.4 (§6.1, §6.2, §16.2). Extends [ADR 0122](0122-a-person-is-two-surfaces-and-iso-7730-is-used-outside-its-range.md)
(a person is two surfaces; the skin was one authored set point) and builds on
[ADR 0192](0192-a-human-is-a-fixed-taxonomy-labelled-by-its-skeleton.md) (the seventeen segments).

## Context

ADR 0122 authored one skin temperature for the whole body from ISO 7730's set point,
`35.7 − 0.028 (M − W)` °C, and solved one clothing surface over it. That is right for the
question ISO 7730 asks — a comfort index for the whole person — and wrong for the question a
camera asks. Measured people are not one temperature: forehead near 34.7 °C against a nose at
33.5 °C indoors, palms near 28 °C and fingers near 25 °C in a cool room. Face minus fingers is
about 10 K, two hundred NETDs, and the body produces it on purpose: thermoregulation defends the
core by cutting blood flow to the extremities. Skin temperature is a *physiological* field, not a
surface energy balance, so no amount of per-cell solving of the skin as a slab gets it.

HU.2 (ADR 0192) made the body a fixed taxonomy of seventeen segments — JOS-3's — and HU.3 put a man
into the renderer with one prim per segment. What was missing was a temperature per segment.

## Options considered

1. **Keep ISO 7730's set point per segment with authored offsets** (forehead −0, nose −1.2,
   hands −6, …). Cheap, and wrong everywhere but the one condition the offsets were read at: the
   nose moves 3.5 K over 18–30 °C of room air while the forehead moves under 1 K. An offset
   table is a snapshot of one weather.
2. **A two-node model (Gagge)** with an authored distribution. One skin node; the same problem.
3. **JOS-3** (Takahashi, Tanabe et al. 2021, Energy & Buildings 231:110575) — seventeen segments,
   each with core, muscle, fat and skin nodes, arterial and venous blood, skin blood flow,
   sweating and shivering, driven by air, radiant temperature, humidity, air speed, clothing per
   segment, activity and posture. Validated against subject data by its authors. MIT. Pure
   NumPy as its authors ship it on PyPI (`jos3`). Chosen.
4. **Fiala / UTCI-Fiala, the Berkeley model, the 65MN.** Fiala is proprietary; Berkeley is
   CC BY-NC; the 65MN's open implementation has no stated licence. All closed to this project.

### Which JOS-3

JOS-3 exists twice: the authors' `jos3` on PyPI, and a port inside `pythermalcomfort`. The port
was tried first because PH.12 already names `pythermalcomfort` as its oracle. **It pins
`numpy < 2.3` and pulls in numba and scipy.** Installing it into Isaac Sim's interpreter
downgraded Isaac's numpy from 2.3 to 2.2.6 — silently, with a `-q` install — which showed up as
thirty new mypy errors from numpy's changed shape typing before anything else did. The install was
reverted (numpy 2.3.5; numba, llvmlite and the port removed). **The authors' `jos3` depends on
numpy alone** and is now a project dependency. `pythermalcomfort` stays what PH.12 made it: a
dev-only oracle in the `comfort` extra, never imported under `src/irsim`.

### What forces the body

**One `WeatherSeries`** (CLAUDE.md #6). Air temperature, humidity and wind are read at every tick.
The **mean radiant temperature** is built from the same sky and sun every surface in the scene is
forced by, as VDI 3787 Part 2 defines it outdoors: half the body sees the sky through the project's
own clear-sky emissivity with the weather's cloud (`irsim.thermal.longwave`), half sees the ground
(taken at air temperature, ESTIMATED), and the sun's beam lands on Fanger's projected area of a
standing person, `f_p = 0.308 cos(γ(0.998 − γ²/50000))`, with half the diffuse and half the ground's
reflection (0.2, ESTIMATED), all weighted by skin's α_k/ε_p = 0.65/0.97. Measured on the
`clear_midlat_summer` noon (26.2 °C air, DNI 823, DHI 106 W/m², sun at 61°): **T_mrt = 47.5 °C,
+21 K over the air**; under a clear 10 °C night sky: **−12 K** below the air; overcast and sunless:
the air exactly.

ASHRAE 55's SolarCal (`solar_gain` in the comfort package) was tried first and **rejected**: it is
a model of a person behind a window over a reflective floor, and with the obvious outdoor settings
(whole sky visible, whole body exposed, floor reflectance 0.6) it gave **+55 K** at this noon,
three times the outdoor balance; with the floor at 0.2 still +34 K.

### Acclimatisation

JOS-3 starts from its neutral state (34 °C skin at 28.8 °C) and creeps for hours: at its own neutral
condition, seated and bare, the mean skin reads 34.28 °C after 1 h, 33.94 after 2 h, 33.75 after
3 h, 33.51 after 6 h. The body runs **two hours** at the scene's start conditions before the first
frame: the drift is then under 0.2 K/h, the mean sits within 0.3 K of Fanger's 33.7 °C, and two
hours is about as long as a pedestrian has been outdoors. A person who has just stepped out of a
warm building is a different scene and authors a shorter run.

## Decision

`irsim.thermal.human_body.HumanBodySolver` owns one JOS-3 state per person, sized from the asset's
phenotype (sex, age, height, mass), clothed per segment from the asset's garments through the body
schema's slots, forced from the scene's weather and sun as above. It is a `TemperatureSolver` for
the mean skin, and it hands out one `SegmentView` per segment — also a `TemperatureSolver` — which
the scene registers as `<name>.skin_<Segment>` targets, plus `<name>.eyes`, `.eyebrows`,
`.eyelashes` and `.hair` at the head's skin. **The views step the body once per tick** however many
are asked: the body remembers the step it last took. A scene authors `solver: human` with the
asset's name, a posture and optionally an activity in met, and nothing else: no schedule, no set
point, no offset — a person's skin is solved, not typed.

Scene schema **v22**. Every committed scene moved to 22; nothing in a v21 scene changed meaning.

## Consequences

- **A person is now seventeen temperatures that move with the weather.** At 0, 10 and 20 °C in a
  1 m/s wind, bare: head 27.9 / 29.6 / 31.5 °C, hands 9.4 / 16.3 / 23.4 °C, feet 8.5 / 15.5 /
  22.9 °C — head > hand > foot at every one, and the extremities fall three times as far as the
  head does. That gradient is the picture.
- **Outdoors is JOS-3's extrapolation**, as outdoors was ISO 7730's (ADR 0122). A bare body at 0 °C
  in wind for two hours is hypothermic in the model as in life; the numbers past the model's
  indoor validation are the model's. HU.12 measures them against radiometric faces.
- **The ground is at air temperature in the radiant balance.** A sunlit road under a person is
  warmer than the air by tens of kelvin (PT.17 solves it), so the long-wave half of T_mrt is low
  on a summer afternoon and high on a clear night. The upgrade is to read the scene's own ground
  field; it is bounded by half the ground's excess over air in T_mrt, a few kelvin.
- **Clothing is insulation per segment only.** The clothing *surface* temperature a camera sees is
  PH.12's ISO 7730 solve, run per garment in HU.5; here a garment only insulates the skin beneath.
- **Prim naming.** The segment objects are `skin_<Segment>`, not `skin.<Segment>` as HU.3 first
  wrote them: a USD prim name cannot carry a dot, and the renderer finds a segment by its prim's
  leaf. Two more faults in the asset prep surfaced on the way and are fixed: the components cache
  was keyed on weld, turn and cuts but not on the source file, so a re-exported model was split
  from the old measurement; and a mesh datablock left behind by the import made the exporter
  number the part's Mesh prim (`skin_Head_001`). The render driver also resolves a numbered leaf
  back to the part the scene names.

## Revisit when

- HU.12 shows the single head node is the limiting error against Charlotte-ThermalFace: then face
  sub-regions are authored offsets on the head node, or a finer model.
- A scene wants the person on a solved ground: then the radiant balance reads the ground field.
- JOS-3 moves on PyPI or the port drops its numpy pin: then the one `import jos3` is the only
  place to change.
