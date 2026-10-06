# STEP2PAPER — the steps from this repo to a submitted paper

When the work is "focus on the paper", work this file top to bottom. Each step has a **done-when**
that a reader can check without having been in the conversation. Tick a step only when its
done-when holds, and put the date and the file that proves it next to the tick.

Rules that apply to every step:

- **No number goes into a paper unless a committed report file produced it.** Anything quoted from
  a log, a terminal, or memory gets re-run and written to `docs/validation/` first.
- **Negative results stay in.** The Tier 4 failure and its attribution to the signal path are
  findings, not things to hide.
- **Licences decide the figures.** Halmstad: statistics only, no frames republished. Anti-UAV:
  no stated licence, so no frames or trained weights published (ADR 0068, ADR 0188).
- The engine-free core, the physics citations and `make check` still apply — paper work is not an
  exemption from CLAUDE.md.

---

## The plan: two papers, not many

| | Paper A — the simulator | Paper B — sim-to-real |
|---|---|---|
| One-line claim | An open, engine-free, multi-band IR camera simulator whose radiometry closes in physical units, with per-point temperature and a published, tiered validation method | Physics-based synthetic LWIR improves drone detection on real footage, and the physics/signal-path ablation shows which mechanisms close the gap (the 09-15 run pointed at the signal path, the 10-04 run at spatial structure — P3/P6 decide) |
| Core evidence | T1–T3 test stack, Kirchhoff closure, fidelity ablation, Tier 4 report | T5 detector experiments + physics ablation of the synthetic set + Tier 4 attribution |
| Candidate venues | SPIE Defense + Commercial Sensing (Infrared Imaging Systems), *Optical Engineering* | CVPR PBVS workshop, WACV |
| Status | most evidence exists; needs packaging | pilot exists; main experiment not run |

Optional later: a dataset paper (Thermal Model Zoo, paired RAW-16 / AGC-8 set — roadmap EV.13,
open question 7).

---

## Evidence snapshot — 2026-10-04

### Tier 4 (synthetic vs Halmstad, display domain)

| run | histogram EMD (≤ 8) | PSD shape ratio (≤ 1.5) | discriminator AUC (≤ 0.7) | file |
|---|---|---|---|---|
| 2026-09-15 | 24.69 FAIL | 5.698 FAIL | 0.907 FAIL | `docs/validation/tier4-2026-09-15.md` |
| 2026-10-04 (re-run, current code) | 27.89 FAIL | 7.081 FAIL | 0.970 FAIL | `docs/validation/tier4-2026-10-04.md` |
| 2026-10-04 self-test control | 0.079 PASS | 1.046 PASS | 0.505 PASS | `validation_report.py --self-test` |

The control passes, so the method works. The comparison against real data still fails, and the
two runs are **not like for like**: since EV.1 and EV.2, metrics are measured per clip, on gated
real clips (6 of 39 admitted). What the new run changes for the paper:

1. **The "signal path dominates" story no longer holds on this run.** On 09-15 the heaviest
   discriminator feature was `noise_scale` (recorder/codec). Now it is `psd_slope`,
   `gradient_median`, `skew` — by the report's own attribution rule that is a **spatial-structure
   (physics/scene) gap**: sky texture, clouds, target edges. Paper B's second claim must be
   re-tested, not assumed.
2. **The synthetic set fails the admission gate it is compared through:** 0 of 6 synthetic clips
   would be admitted (4 `no_flat_window`, 1 moving, 1 `noise_floor_collapsed`). The "matched"
   scenario is not yet matched to what the gate keeps on the real side.
3. **The EMD and PSD targets are below the real set's own spread.** Real clip vs real clip reads
   30.57 (EMD) and 11.51 (PSD) against targets of 8 and 1.5, so no simulator could pass at clip
   level. The ADR 0068 targets need revisiting (e.g. "between-set ≤ within-set spread"), and the
   paper should report them that way.
4. AUC is the metric that discriminates. It is the one to drive down and the headline number.

### T5 pilot (YOLO11n, 30 epochs, scored on the real Anti-UAV sky test split)

| arm | trained on | test mAP50 | test mAP50-95 |
|---|---|---|---|
| real→real | Anti-UAV sky (14,784 frames) | 0.991 | 0.568 |
| synthetic→real | `irsim_sky_pilot` (300 frames) | 0.379 | 0.113 |
| mixed→real | both | 0.991 | 0.565 |

Source: `outputs/validation_phase/*/report.json` (not committed yet — see P5).
Reading: synthetic-only transfers but weakly; mixed shows no gain **because real→real is already
at the ceiling**. The pilot cannot show a benefit — the low-real-data regime (P5) is where one
would appear.

---

## Steps

### P1 — Decide venues, claims and authors  *(owner decision)*
- Pick one venue per paper and write its deadline here. Write each paper's one-sentence claim.
- Agree the author list and order now — several people work on this repo.
- **Done when:** the table above has venues + deadlines and an author line per paper.

### P2 — Fix the known measurement bugs before any final number
- `scripts/generate_matched_scenario.py` builds the `LayeredAtmosphere` without the camera's
  spectral response (warning: a 6.8× error in the `h2o_wing` class weight). Pass `responses=`.
- Recorder model: `agc: linear` and CRF 23 are ESTIMATED stand-ins for Halmstad's Y16 → 8-bit →
  h.264 path (M12.1). Tier 4 attributes its failures here first.
- Halmstad boxes are MATLAB MCOS objects with no Python reader, so contrast-ratio and ESF checks
  are untestable. Either write the reader or state the limitation in the paper.
- Matched clips must pass the same admission gate as the real ones (0 of 6 do on 2026-10-04):
  give the scenario priors a flat-sky window, a gated target motion and a noise floor.
- Re-state the EMD/PSD targets relative to the real set's own clip-to-clip spread (ADR 0068
  amendment); the absolute 8-code / 1.5 targets are tighter than real-vs-real.
- Optional, time-boxed: XD.12 — ask the Halmstad authors for the Y16 originals.
- **Done when:** each item is fixed with a test, or written into Paper A's limitations.

### P3 — Tier 4 against the Anti-UAV sky set (the T5 reference), clear sky first
Why this set and not only Halmstad: per-clip EMD/PSD on Halmstad measure *scene* differences
(real-vs-real already reads 30.6 / 11.5). Anti-UAV sky (ADR 0188, ~28.8k frames) is the same
real set T5 scores on, is already sky-only, keeps the clip id in each filename, and has YOLO boxes —
so the contrast check stops being untestable.

1. **Reader:** `validation_report.py` accepts the YOLO layout, grouping frames by clip id.
2. **Clear/cloudy split of the real clips:** hand-label at clip level (tens of clips). Do not use a
   texture filter — it would select on the statistic being tested.
3. **Real-side reference stats now** (no synthetic needed): per-clip histogram, radial PSD on the
   flat-sky window, drone-vs-sky contrast inside the boxes; commit as `docs/validation/`.
4. **Population test, not pairs:** report where each synthetic clip falls in the real clips'
   distribution (and the between-set vs within-set spread), plus the discriminator AUC.
5. **Run clear-vs-clear** with `irsim_sky_v1` (clear-sky by design) once it finishes; add
   cloudy-vs-cloudy when the cloud model settles.
6. Match resolution/FOV to the Anti-UAV camera (sensor config), or compare at a common scale.
- Caveats for the paper: Anti-UAV's camera/ISP is unknown (signal path `display`), so noise-
  and fine-texture statistics are weaker evidence than on Halmstad; licence unstated — stats only.
- Keep Halmstad as the secondary set (known Boson signal path) and run `--self-test` alongside.
- **Done when:** a dated clear-sky `docs/validation/tier4-antiuav-*.md` exists from post-P2 code.

### P4 — Freeze the evaluation protocol
- Detector, epochs, image size, augmentation, real splits (Anti-UAV sky, ADR 0188), and
  **≥ 3 seeds per arm** — single-seed mAP differences are not publishable.
- Write it as an ADR so the numbers in Paper B have one definition.
- **Done when:** the ADR is committed and `train_detector.py` takes the seed list.

### P5 — The main sim-to-real experiment (Paper B, Table 1)
- Use the full `irsim_sky_v1` set (in progress by another session), not the 300-frame pilot.
- Data-scarce curve: real fraction ∈ {0, 1 %, 5 %, 10 %, 25 %, 100 %}, each with and without
  synthetic data, plus synthetic pre-train → real fine-tune. This is where a benefit can appear.
- Report mean ± std over seeds; commit every `report.json` under `docs/validation/t5-*/`.
- **Done when:** the curve exists with error bars and is committed.

### P6 — Which physics matters (Paper B, Table 2; also Paper A)
- Re-render the synthetic set with one mechanism removed at a time, re-train, score on real:
  per-object constant temperature instead of per-point; no noise chain; no AGC/codec round trip;
  no clouds; fixed vs real-matched viewpoint (EV.10 — the literature's biggest single lever).
- Add a naive baseline: RGB render converted to grayscale.
- Cross-check against `scripts/fidelity_ablation.py` (which mechanisms move the frames) — the
  paper's story is the agreement or disagreement between the two rankings.
- **Done when:** a committed table ranks the mechanisms by Δ mAP with seeds.

### P7 — Paper A's evidence package
- One table of the validation tiers: what T1/T2/T3 assert, how many tests, what passes
  (`TECHNICAL_REPORT.md` status table is the source).
- Kirchhoff closure across the material library; aperture factor; float32 round trip — the
  non-negotiables, each as a measured number.
- Related-work comparison: DIRSIG, MuSES/CoTherm, OKTAL-SE (SE-Workbench), CARLA/AirSim thermal
  modes, recent synthetic-IR drone sets. Columns: open source, bands, per-point temperature,
  radiometric units, noise/ISP model, validated against real data.
- **Done when:** both tables are in `docs/paper/` with every cell sourced.

### P8 — Figures
- Pull from `site/gallery.yaml` renders; grayscale white-hot.
- Pipeline diagram (scene → thermal solve → radiance → atmosphere → optics → detector → ISP).
- Tier 4 attribution figure; P5 curve; P6 ablation bars.
- No Halmstad or Anti-UAV frames in any figure.
- **Done when:** every figure has a script under `scripts/paper/` that regenerates it.

### P9 — Reproducibility
- Tag a release (`paper-a-v1`, `paper-b-v1`). One command reproduces every number from the tag.
- Publish the synthetic set and configs if licence allows (open question 7 for the paired set).
- **Done when:** a clean checkout of the tag reproduces the tables.

### P10 — Write
Most of the text exists; adapt it rather than write from zero:

| section | source in the repo |
|---|---|
| Method — radiometry, materials, thermal, atmosphere, optics, detector, ISP | `docs/physics-model.md` §3–§12 |
| Design choices and approximations | `docs/decisions/` (ADRs) |
| Validation method | `docs/physics-model.md` §15, `ir-sim-testing` skill, `docs/validation/` |
| Limitations | `TECHNICAL_REPORT.md` → Current limitations |
| Data and licences | `data/validation/README.md` |

- **Done when:** a full draft exists in `docs/paper/<paper>/` (LaTeX, venue template).

### P11 — Review and submit
- Internal read by someone not involved; check every number against its report file.
- arXiv preprint if the venue allows it.
- **Done when:** submitted, with the submission ID written here.
