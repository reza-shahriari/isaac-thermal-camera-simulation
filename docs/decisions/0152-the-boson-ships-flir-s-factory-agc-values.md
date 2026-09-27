# ADR 0152 — The Boson config ships FLIR's factory AGC values

**Status:** Accepted
**Date:** 2026-09-27
Roadmap: SC.22. Spec: `docs/physics-model.md` §11.3, spec issue S52. Applies
[ADR 0147](0147-information-based-equalisation-approximates-the-boson-default-agc.md) (the operator)
and [ADR 0149](0149-agc-is-a-set-of-vendor-neutral-families-and-a-band-follows-its-detector.md)
(a camera is a parameter set of the families).

## Context

`flir_boson_640_lwir.yaml` selected global plateau equalisation at plateau 0.012, with DDE 0.35.
Neither is a value FLIR ships. On a cluttered sky that operator is full histogram equalisation, and
the drone in `phantom4_perpart` got 2–3 grey codes (S52). ADR 0147 added FLIR's default operator as
an opt-in and left the switch to this step.

ADR 0147 assumed FLIR publishes no defaults beyond the plateau and the Smoothing Factor. That is
wrong. [R51] p. 5 prints the factory values in a screenshot of the camera's "Current Camera Values"
panel: **Information-Based Mode on, Tail Rejection 0, Max Gain 1.38, Damping Factor 85, ACE 0.97,
Plateau Value 7, Linear Percent 20, Detail Headroom 12, DDE 0.95, Smoothing Factor 1250.** The page's
text layer does not carry them, which is how they were missed.

## Options considered

1. **Tune the values to make the demo look best.** Linear Percent 0.3 was the demo's choice. This
   is the thing the project exists not to do: the display has to match what a camera shows.
2. **Map every published value onto irsim's controls, and write down each one that does not
   map** (chosen).
3. **Move `halmstad_boson_320.yaml` as well**, as the roadmap row said. Its `linear` is not a Boson
   AGC. It stands for the dataset recorder's Y16 → 8-bit conversion, so moving it would misdescribe
   the public frames it is compared against.

## Decision

Option 2, for the Boson 640 only:

| [R51] | irsim | mapping |
|---|---|---|
| Information-Based on | `agc: information_based` | direct |
| Plateau Value 7 | `plateau: 0.07` | direct: both are a fraction of pixels per bin |
| Linear Percent 20 | `linear_percent: 0.20` | direct |
| Max Gain 1.38 | `max_gain: 1.38` | display codes per count, as both define it; **ESTIMATED** because an irsim DN is not a Boson count |
| Detail Headroom 12 | `detail_headroom: 0.047` | 12 of the 8-bit range (0–127 of 255) reserved at each end, i.e. 12/255 |
| Smoothing Factor 1250 | `smoothing_sigma_dn: 1250` | already the default. It shares the count-scale caveat, which ADR 0147 already records |
| DDE 0.95 | `dde_gain: 0.0` | FLIR's value is the total high-pass gain. irsim adds the high-pass back at 1 + `dde_gain`, with `dde_gain` ≥ 0, so 0 is the nearest legal value: 5 % more detail than the Boson |
| Tail Rejection 0 | — | the min–max map that `linear_percent` blends in is already tail-free |
| ACE 0.97 | `gamma: 1.0` | not mapped. FLIR itself says "ACE is implemented differently in Boson", so it is not a gamma |
| Damping Factor 85 | — | not modelled: the AGC has no temporal state (`SC.10`) |

The Warp display path refuses `information_based` (ADR 0147). The GPU integration tests for it now
pin the pre-SC.22 plateau ISP, because that is the operator they cover. No render script uses that
path: every one takes `display8` from the CPU branch.

## Consequences

- **Display only.** Across every golden, only the two `display8` arrays changed (ramp by up to 20
  codes, hot patch by up to 227). `dn16`, `radiance` and `apparent_t` are bit-identical, and only their
  hash sidecars moved, because the golden key hashes the whole config. `test_boson_default_agc.py`
  checks the same thing through `run_frame`.
- **What it buys.** On the phantom4-shaped synthetic frame the target goes from 2 to **37** codes,
  with its four parts in temperature order. On the saved `phantom4_perpart` planes, frame 48 goes from
  3 to 26 codes, frame 96 from 2 to 20, and frame 136 from 33 to 105.
- **A fire does not collapse it.** On PH.8's fire scene the factory ISP gives the person 51 codes
  against the linear stretch's 1 (`test_gain_state.py`). Max Gain is what costs shades there:
  plateau with Linear Percent and Max Gain but no information histogram gives 3, because on a
  noiseless two-temperature frame the per-DN slope cap has almost no occupied bins to spread over.
  Tests that compare the AGC *families* now zero the Boson's controls and compare bare operators.
- `scripts/redisplay_planes.py` takes every AGC control from the camera unless a flag overrides it,
  so its `FIXED_agc` clip is the camera's own display. Under the factory values the mid-clip drone
  gets 16–28 codes, less than the 25–39 the demo got with Linear Percent 0.3. That is the camera, not
  a regression.
- The two mappings that are not exact (Max Gain and Smoothing on irsim's DN scale; DDE clamped at
  unity) stay estimated until a Tier 4 fit against public Boson footage exists, as ADR 0147
  requires for `info_weight`.

## Revisit when

A published Boson counts-per-kelvin figure makes Max Gain and Smoothing Factor convertible to
irsim's DN scale; `dde_gain` is allowed below zero (a blurring DDE); or `SC.10` gives the AGC the
temporal state that Damping Factor needs.
