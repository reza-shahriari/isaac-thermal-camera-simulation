# 0170 — The AGC transfer is damped across frames at the core's own rate

Date: 2026-09-29

**Status:** Accepted (2026-09-29, SC.10).

## Context

Every AGC operator in `irsim.isp` was a pure function of one frame (ADR 0031, ADR 0014's
no-cross-frame-state rule for an SPG port). Real cores are not: the Boson filters its AGC with a
Damping Factor, "an Infinite Impulse Response (IIR) filter used to adjust how quickly the AGC
algorithm reacts to a change in scene" (0 = immediate, 99 = slowest, 100 = frozen; [R51]), and the
Lepton applies "(N/256) · previous + ((256 − N)/256) · current" to its HEQ transformation function
(default N = 64; Lepton IDD). So a hot target entering the frame re-maps the rest of the picture
over a few frames in a real clip and in a single frame in a render. The SC.10 ablation ranked this
switch first between the rendered display stream and the real set (AUC 1.000).

## Options

1. **Damp the histogram.** Rejected: the information-based mode builds its transfer from two
   histograms and a gain limit, and damping the inputs is not what either vendor documents; the
   Lepton names the *transformation function*.
2. **Damp the displayed image.** Rejected: an IIR on pixels is a ghost of every moving object,
   which is the membrane's job (§9.2) and not the AGC's.
3. **Damp the transfer table** (chosen). Every global mode already builds a table over the
   2^bit_depth DN bins before indexing the frame; each now hands it to an optional `lut_hook`
   first. `irsim.isp.damping.TransferDamper` is that hook: it keeps the damped table in
   `PipelineState.buffers` and returns `a · previous + (1 − a) · current`. In the information mode
   the damped table is also what the detail layer's slope is taken from. The first frame adopts
   its own table, so a hook-on first frame is the per-frame operator bit for bit.

**The coefficient is per native frame, scaled by the time that passed**:
`a_eff = a ** (Δt · frame_rate_hz)`, with Δt the scene time since the last display update. A core
filters at its own rate whether or not every frame is recorded, so a six-second time-lapse has let
a 60 Hz core run 360 steps (0.85^360 ≈ 1e-26: undamped, as it should be) and continuous video is
damped at the camera's rate. The same policy the membrane IIR adopted for a time-lapse (ADR 0074).

`isp.agc_damping` is the per-frame coefficient in [0, 1] (Boson df / 100, Lepton N / 256). It
defaults to 0 and is left out of the hashes there, so every existing config is unchanged. The
Boson preset carries **0.85**, FLIR's factory Damping 85 ([R51] p. 5). `plateau_local` damps its
whole `(ty, tx, bins)` stack of tile tables, each on its own history; its SC.25 linear blend,
applied after the tiles, is not damped (a remainder: that blend is off in every shipped config
that uses the tiled mode).

## Consequences

* A background pixel's displayed value after `k` frames of a new scene is exactly
  `a^k · old + (1 − a^k) · new` in the linear, plateau, equalise and tiled modes (1e-6), and the information mode's
  first frame shows 15 % of the step (`tests/unit/test_agc_damping.py`).
* Through `run_frame`: 60 Hz video shows the old map in the first frame after a step and settles
  to within one display code after 60 frames; a six-second time-lapse is bit-identical to the
  undamped operator.
* Goldens: no array moved (each is a single frame, which adopts its own transfer); the Boson's
  config hash did, so 14 sidecars were regenerated.
* The Warp display twin (`display_stage_warp`) has no damping; it is not on the production path,
  and its equivalence tests are single-frame, where the two agree.
