# 0143 — A capability scene gives every part its own thermal node, and the scene decides the granularity

Date: 2026-09-26
**Status:** Accepted
Roadmap: AI.7 (§6.6, §12.3)

## Context

ADR 0138 split the imported Phantom 4 into nineteen functional parts so that a temperature could
name hardware. The scene that flies it, `phantom4_parts.yaml`, then gave those nineteen parts
**four** thermal nodes — airframe, motor, ESC, battery — and the render driver carried the
part → node map as a hard-coded table. Physically that was defensible: the four motors share one
§6.6 model and one throttle history, so they *are* the same temperature in that scene.

Shown as a clip, it reads as the opposite. Four motors at one temperature, four ESCs at one
temperature and a single "airframe" value on every shell, arm and leg look exactly like a
simulator that assigns one temperature per category, which is the defect that drove the project's
owner off the previous simulator (the per-point requirement, ADR 0110) and which a capability
demonstration exists to refute. The owner's words on seeing the four-node proposal: *"i want to
have per-part temperature, as its a test i want to see the simulation capabilities."*

Two things stood in the way. The driver's map hard-wired the granularity, so a finer scene could
not be expressed without editing Python. And the parts that make up most of the silhouette —
shells, arms, gimbal — carry 150–220 k faces each in the prepared archive, and the mesh solver
puts a cell on every face (ADR 0132, ADR 0137), so mesh-solving them is a cell budget the
current archive does not have.

## Decision

1. **The scene is the authority on node granularity.** `render_phantom4.target_for_part` tries
   the part's *own name* first, then an ordered list of kind-level candidates
   (`esc_front_left` before `esc`, `arms` before `airframe`), and takes the first one the scene
   defines. The same driver and the same asset therefore give four nodes on
   `phantom4_parts.yaml` and fifteen on `phantom4_perpart.yaml`. Only targets are candidates,
   never §12.3 surfaces: a surface already reaches the pixels through its cells, and the readout
   legend draws every mapped name.

2. **`phantom4_perpart.yaml` gives the nineteen parts nineteen temperatures**, by three
   mechanisms of different fidelity, each stated in the file:
   * four motor nodes and four ESC nodes on four **different throttle histories** — rear pair
     loaded in nose-down cruise, outer pair in a banked orbit, upwind pair hovering in a wind.
     The asymmetries are 6–12 % of throttle; at 45 K × u² that is 4–7 K between motors, eighty
     to a hundred and forty NETDs on the Boson;
   * nine **mesh-solved surfaces** as before (battery skin, four mounts, four propellers), with
     each propeller's boundary-layer speed now derived from *its own* motor's throttle;
   * six **lumped airframe-family nodes** (shell_upper, shell_lower, arms, landing_gear, gimbal,
     camera_lens) as `T_air + offset`, with ESTIMATED offsets that say what sits behind each
     surface — gimbal motors, an image sensor, sun on a white shell.

3. **The airframe family is lumped on purpose, and that is the fidelity skipped.** A mesh solve
   of the shells is the physically right answer (a sun-lit upper shell carries a gradient, ADR
   0110) and is a scene edit away once the thermal archive is decimated to what conduction can
   resolve (ADR 0137 measured 77 % of the Phantom 4's faces as finer than that floor). Until then
   an offset per part is honest about being a per-part *value*, not a field, and the readout
   says which parts are which.

4. **A third display clip, plateau-equalised over the whole run.** The aircraft-stretched and
   sky-stretched clips of AI.2 each hide half the scene. `--agc-clip` builds one §11.3 plateau
   LUT over every frame of the run and applies it to each, so the cumulus and the motors share
   one frame the way a real camera shows them, without the per-frame rescaling that would cancel
   the warm-up being filmed. Display only; the float32 planes are untouched (ADR 0068).

## Consequences

* A capability demonstration of this asset now shows same-kind parts diverging and converging
  with the mission, and `tests/unit/test_phantom4_perpart.py` pins that on the scene's own
  solvers, engine-free: rear pair over front pair in cruise, outer pair in each orbit, all four
  within 0.1 K once landed.
* The offsets in the airframe family are numbers someone chose. They are labelled ESTIMATED in
  the scene and must not be read as a thermal model of a gimbal; a Tier 4 fit against public
  aerial IR would replace them, or a decimated archive would make them unnecessary.
* The readout legend carries fifteen rows on a 512-row frame. That is its capacity; a scene with
  more nodes than that needs a legend that pages or groups, not a taller frame.
* `TARGET_BY_PART` is now a map of candidate tuples and its docstring says so; any driver that
  copies the old `dict[str, str]` form is copying the wrong thing.
