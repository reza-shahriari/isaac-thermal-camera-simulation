# Humans in the simulation: what exists, what to build (2026-10-06)

The owner's requirement, 2026-10-06: a pedestrian is a necessary part of an autonomous-car
simulation, and a human is a different kind of asset from a drone or a ship. The clothes change,
the person may be a child, a woman, a man — but the *human is human*: the same body, the same
regions, the same physiology, every time. Two deliverables were named:

1. **One human that ships with the project** and looks like a real person in real life.
2. **A fast, forward path for any downloaded human** — a scan, a game character, a parametric
   export — to enter the simulation with the same thermal materials and the same regions, so that
   one scene can hold a girl in a blue shirt and another in a red one, for RGB as much as for IR.

This file records the audit of the repository, the external evidence (every claim with its
source; "not verified" where the primary page could not be opened), and the plan the evidence
supports. The research was done by four parallel surveys on the day; their verdict keys are kept.
Nothing in `docs/roadmap.md` is changed by this file — where the work is tracked is one of the
open decisions at the end.

---

## 1. What the repository already has

| exists | where | what it gives a human |
|---|---|---|
| Skin + clothing thermal model (PH.12, ADR 0122) | `src/irsim/thermal/human.py` | Skin **authored** from the ISO 7730 set point `35.7 − 0.028 (M − W)` °C; clothing surface **solved** by bisection from ISO 7730's own balance. One skin temperature for the whole body. Bound to no asset and no scene. |
| Skin and cotton materials | `configs/materials/human_skin.yaml`, `cotton_clothing.yaml` | ε 0.98 / 0.95 LWIR. No hair, denim, polyester, wool, leather, shoe rubber. |
| Name-glob mapping | `configs/materials/mapping.yaml` | `*skin*`, `*cloth*`, `*shirt*`, `person`, `pedestrian` → the two materials above. |
| Part decomposition (ADR 0138) | `scripts/prep_asset.py`, `irsim.io.asset_parts` | Parts recovered **by connected component** and claimed by `PartSelector` predicates (material, object, position, size). Right for a drone; wrong for a human, who is *one* connected shell whose parts are body regions. |
| Contacts and hidden parts (AI.11, AI.16, AI.18) | `ContactSpec`, `HiddenPartSpec`, `PartSelector.objects` | A part may be the object a person separated and named — the hook a region-split body can use. |
| Asset ingestion (AI lane, `ingest-asset` skill) | licence gate, provenance, Blender prep, audit, FBX side-export | Reusable end to end *except* the decomposition step. |
| Weather injected once (non-negotiable 6) | `WeatherSeries` | Any physiology model must take its forcing from here. |
| Skinned / deforming meshes | — | **Nothing.** No `UsdSkel` handling anywhere in `src/irsim_isaac`; every asset so far is rigid. The local Isaac build ships no people extension. |

The gap is therefore not the physics of a person, which exists in a whole-body form, but:
(a) per-region physiology, (b) a human-shaped decomposition path, (c) six materials, (d) a
deforming body through the Isaac glue, and (e) a redistributable human to ship.

---

## 2. External evidence

### 2.1 Where a redistributable human can come from

Verdict key: **FIT** = publish images *and* redistribute models; **IMAGES-ONLY** = render and
publish, never redistribute the model; **NO** = non-commercial and no redistribution.

| source | facts | licence (quoted) | verdict |
|---|---|---|---|
| **MakeHuman / MPFB2** (Blender add-on) | MPFB 2.0.17 (22 Jul 2026), Blender ≥ 4.2 — the owner runs **5.2.2 LTS**. Eleven macro sliders; **age 0.0 = baby, 0.1875 = child, 0.5 = young adult, 1.0 = old** — children are native. Rigs: Default, **Rigify**, **GameEngine**, CMU MB, a dedicated **Mixamo** rig. 47 CC0 asset packs (meshes, materials, poses), 13 CC-BY (some hair and clothing). The body is one vertex group, so regions must come from the rig's bone weights. | Add-on GPL-3.0; **core assets CC0**: "All core assets (the base mesh, targets, skins…) are shared under CC0" — [FAQ](https://static.makehumancommunity.org/mpfb/faq/can_i_sell_models.html); [extension page](https://extensions.blender.org/add-ons/mpfb/); [repo](https://github.com/makehumancommunity/mpfb2); [rigs](https://static.makehumancommunity.org/mpfb/docs/characters/rig.html); [asset packs](https://static.makehumancommunity.org/assets/assetpacks.html) | **FIT** — the only surveyed tool where body, textures, children *and* clothing can all be redistributed |
| **Anny** (NAVER LABS Europe, Nov 2025) | A parametric body model **built on MakeHuman**, phenotype parameters (gender, age, height, weight) over blendshapes, infants to elders, calibrated on WHO statistics; a Python model, differentiable. | **Apache-2.0** — [blog](https://europe.naverlabs.com/blog/anny-a-free-to-use-3d-human-parametric-model-for-all-ages/), [paper](https://arxiv.org/abs/2511.03589), [code](https://github.com/naver/anny) | **FIT** — the scripted route to a *population* (random phenotypes) without opening Blender |
| **SMPL / SMPL-X** (MPG) | 27-part per-vertex segmentation JSON exists; SMIL for infants; AGORA children by interpolation. The Meshcapade commercial channel **shut down 18 Apr 2026** after Epic's acquisition (see §2.2). | "sole purpose of performing non-commercial scientific research … any use for commercial, pornographic, military, or **surveillance**, purposes is prohibited"; "shall not be copied, shared, distributed" — [model licence](https://smpl-x.is.tue.mpg.de/modellicense.html) | **NO** for redistribution; the surveillance clause is a live risk for a pedestrian-detection AV use |
| **Isaac Sim people** | `omni.anim.people` is replaced by `isaacsim.replicator.agent` (IRA) 1.x; 4+ adult characters (`People/Characters/…police_04`, `…medical_01`, …) with retargeted skeletons; whole-character semantic label only, **no per-body-part labels documented**; adults only. | "may not be modified or redistributed except as expressly permitted" — [licence FAQ](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-faq.html), [additional licence](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license-isaac-sim-additional.html); [discussion #477](https://github.com/isaac-sim/IsaacSim/discussions/477); [IRA tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/action_and_event_data_generation/tutorial_replicator_agent.html) | **IMAGES-ONLY**; useful as a renderer-side test of skinned meshes, not as library content |
| **Renderpeople free people** (~9 scans) | photoreal, rigged, clothed | may render "for commercial or private purposes", "not allowed to make the 3D data available to third parties" — [T&C](https://renderpeople.com/general-terms-and-conditions/) (domain blocked; from search excerpts) | IMAGES-ONLY |
| **Sketchfab CC-BY scans** | e.g. "Scanned posing man Dennis" (Renderpeople upload, 100 k tris, clothed) — [link](https://sketchfab.com/3d-models/scanned-posing-man-dennis-f61c61c98be743c0b37008a44ef113c4); rig status not verified | CC-BY 4.0 | **FIT** — static pose, cannot be re-dressed; the right *exemplar for the arbitrary-mesh path* |
| **Blender Studio Human Base Meshes v1.4.1** | CC0 male/female bases, heads, hands, feet; nude, likely unrigged (not verified) | CC0 — [demo files](https://www.blender.org/download/demo-files/) | FIT as a sculpt base only |
| HumGen3D, Character Creator 5, Daz Genesis 9, MetaHuman | commercial; CC5 and Daz forbid AI/ML dataset use; MetaHuman's EULA update 21 bans use "to build or enhance any database or train or test … machine learning" (wording from [CG Channel](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/); EULA pages returned 403, **not verified**) | — | **UNFIT** for a dataset-producing simulator |
| BEDLAM, AGORA, CAPE, THuman, 2K2K, SynBody, SURREAL | research scans / renders | all non-commercial, no redistribution, MPG template with the surveillance clause | NO |
| Quaternius / Kenney characters | CC0, rigged | — | FIT but stylised — fail "looks like a real human" |

### 2.2 Rigging and segmenting an arbitrary human mesh

| tool | what it does | licence | verdict |
|---|---|---|---|
| **Argmax over skinning weights** | A rig *is* a segmentation: Mixamo's 65-bone skeleton (`Head, Neck, Spine*, Left/RightArm, …ForeArm, …Hand, …UpLeg, …Leg, …Foot, …ToeBase`), Rigify's human metarig, SMPL-X's 27 parts and MPFB's GameEngine rig all map by name onto a head / neck / torso / upper arm / forearm / hand / thigh / shin / foot list. [Mixamo skeleton](https://community.adobe.com/t5/mixamo/mixamo-standard-65-bone-skeleton/m-p/11442179), [SMPL-X segmentation JSON](https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json) (wiki repo has **no LICENSE file**) | — | **FIT** whenever a rig exists |
| **Make-It-Animatable** (CVPR 2025) | Any humanoid mesh, any pose → **Mixamo-standard 52-bone** skeleton, weights and rest pose in under a second; `app_blender.py` included | **MIT** (verified) — [repo](https://github.com/jasongzy/Make-It-Animatable) | **FIT** — the default auto-rigger: its output already carries Mixamo names |
| **UniRig** (SIGGRAPH 2025) / **SkinTokens** (2026) | Skeleton + skinning for arbitrary shapes (humans, animals, objects); bone naming not a fixed template | **MIT** — [UniRig](https://github.com/VAST-AI-Research/UniRig), [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) | FIT, needs a naming pass; the fallback when a mesh is not humanoid enough |
| **Auto-Rig Pro** | one-click biped marker detection, scriptable in Blender 2.93–5.2 | GPL code, $50 | FIT (paid) |
| **Mixamo** auto-rigger | online, free, unmaintained, outages in 2025; raw character/animation files may not be redistributed — [FAQ via Adobe Community](https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400) | — | NO for automation; animation retargeting onto our own meshes is permitted but the clips cannot ship in the repo |
| RigAnything, PartField | Adobe / NVIDIA research licences, non-commercial | — | NO |
| **Scan → SMPL registration** (NICP, Mesh2SMPL, SMPLify-X) | the technically best label-transfer route | all inherit the MPG model licence; Meshcapade's "mesh to SMPL" service **shut down** ([notice](https://www.mpg.de/26082348/)) | NO as a default; opt-in local-only at most |
| **Sapiens** (Meta) | 28-class body-part segmentation of an *image*, including `Upper_Clothing, Lower_Clothing, L/R_Shoe, Hair, Face_Neck, Torso, L/R_Upper_Arm, …` — maps almost 1:1 onto the thermal regions; multi-view render → project → per-face vote | **CC BY-NC 4.0** (verified) — [repo](https://github.com/facebookresearch/sapiens) | FIT-NC: best vocabulary of all, licence-gated |
| **CloSe** (3DV 2024) | 18 garment classes on scans; needs SMPL parameters | code MIT, SMPL dependency | FIT-NC |
| **Find3D** | open-vocabulary, text-queried 3D part segmentation on point clouds ("shoe", "hair", "hand") | **MIT** — [repo](https://github.com/ziqi-ma/Find3D) | FIT — the commercial-safe fallback for skin / hair / garment |
| SAMPart3D, HoloPart | class-agnostic 3D parts, need naming | MIT | FIT as a last resort |

### 2.3 The physiology of a person seen in the infrared

**Regional skin temperature is the image.** Neutral indoors by thermography: forehead 34.7 ± 0.4,
chin 34.3, cheeks ≈ 34.0, nose 33.5 ± 0.8 °C ([PMC8955831](https://pmc.ncbi.nlm.nih.gov/articles/PMC8955831);
room temperature not verified). The nose is the coldest facial region whenever air ≤ 26 °C
([S0306456522002364](https://www.sciencedirect.com/science/article/abs/pii/S0306456522002364)).
Over 18–30 °C indoor air the dynamic range is **nose 3.46 K, cheeks 2.53 K, forehead 0.89 K**
([S2666123325000844](https://www.sciencedirect.com/science/article/pii/S2666123325000844)). Hands
in a cool room: palm 28.0 ± 1.8, fingers 24.6 ± 2.9 °C ([RG 311929004](https://www.researchgate.net/publication/311929004));
in cold workplaces 99 % of workers had at least one finger ≤ 24 °C ([PMC6265722](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6265722/)).
So a person is **not** one skin temperature: face minus fingers is 10 K in a cool room, two hundred
NETDs, and the shipped whole-body set point renders it as zero.

**The open multi-node model is JOS-3.** Takahashi et al. 2021 (Energy & Buildings 231:110575,
[doi](https://doi.org/10.1016/j.enbuild.2020.110575)); **17 segments** — head, neck, chest, back,
pelvis, L/R shoulder, arm, hand, thigh, leg, foot — each with skin and core temperature,
wettedness, blood flow; takes **solar shortwave gain, age, sex, height, weight**; pure
Python/NumPy; **MIT** ([TanabeLab/JOS-3](https://github.com/TanabeLab/JOS-3), also
`pythermalcomfort.models.JOS3`, [MIT](https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort)).
`pythermalcomfort` is already this project's declared `comfort` extra. Fiala and the Berkeley model
are proprietary or CC BY-NC; Tanabe 65MN's licence is not verified. **No production IR simulator
documents a pedestrian physiology model** except ThermoAnalytics' closed Human Thermal Extension
([page](https://www.thermoanalytics.com/human-thermal-extension)); DIRSIG moves humans as rigid
geometry ([mov](https://dirsig.cis.rit.edu/docs/new/mov.html)). A thermoregulated pedestrian in an
open simulator is a differentiator, not a catch-up.

**Clothing.** 1 clo = 0.155 m² K W⁻¹; garment and ensemble tables in ISO 9920 (paywalled) and,
open, in [pythermalcomfort's clothing docs](https://pythermalcomfort.readthedocs.io/en/latest/documentation/clothing.html)
(garments 0.01–0.56 clo, ensembles 0.27–2.17). The clothing-surface balance the project already
solves is ISO 7730 Annex D's (ADR 0122).

**Surface optics, 8–14 µm.** Skin **0.98 ± 0.01, flat 1–14 µm, independent of pigmentation**
(Steketee 1973, [doi](https://doi.org/10.1088/0031-9155/18/5/307); re-check
[PLoS ONE 2020](https://doi.org/10.1371/journal.pone.0241843)). **Cotton, nylon, polyester ≈ 0.88**,
independent of moisture; acrylic 0.81 → 0.88 wet (Belliveau et al. 2020, Textile Res. J. 90:1431,
[doi](https://doi.org/10.1177/0040517519888825)). "**Color has no effect on surface emissivity for
the same fabric**"; order PA > PET > wool > cotton (Zhang, Hu, Zhang 2009, J. Textile Inst. 100:90,
[doi](https://doi.org/10.1080/00405000701692486); wool ≈ 0.73 from a snippet, not verified in text).
Leather 0.95–1.00 and denim / hair only from vendor tables — **no primary source found**.
⚠️ The library's `cotton_clothing` carries ε 0.95 LWIR from `docs/physics-model.md` §16.2;
Belliveau's measurement is 0.88. That is a spec issue to log, not to fix silently.

**Colour and band.** Dye dominates Vis–NIR (an olive dye 23–32 % in the visible, 50–58 % in the
NIR); "all dyed samples (irrespective of colour …) gave similar signatures in the SWIR" (Kaur et
al. 2024, Appl. Spectrosc., [doi](https://doi.org/10.1177/00037028241258111)). Skin reflects
strongly 800–1100 nm regardless of tone and darkens past 1400 nm (Troy & Thennadil 2001,
[doi](https://doi.org/10.1117/1.1344191)). So the blue shirt and the red shirt differ in RGB and
NIR by reflectance, in LWIR **only through temperature** — the darker garment absorbs more sun and
runs warmer. That is exactly the hook the material schema already has (`solar_absorptivity`).

### 2.4 Data to validate against

| dataset | radiometric? | licence | use |
|---|---|---|---|
| **Charlotte-ThermalFace** | **16-bit T-linear**, ~10 k images, 10 subjects, varying room temperature and distance — [repo](https://github.com/TeCSAR-UNCC/UNCC-ThermalFace), [doi](https://doi.org/10.1016/j.infrared.2022.104209) | CC BY-NC-SA 4.0 | **face-region deltas vs ambient** — the JOS-3 head check |
| **Teledyne FLIR ADAS v2** | 14-bit T-linear TIFF + 8-bit, 26 442 frames, pedestrians at street level — [form](https://oem.flir.com/solutions/automotive/adas-dataset-form/) | terms page 404, **not verified** | the pedestrian-in-a-street check, once the licence is read |
| KAIST, LLVIP, CVC-14, OSU, M3FD | 8-bit, AGC'd | NC variants | contrast and shape only |

---

## 3. The idea that makes humans "plan differently"

A drone's parts are discovered — connected components, clustered by size and position, named by
research. A human's parts are **known in advance and identical for every human**: the same 17
segments JOS-3 solves, each either bare or under a garment, plus hair and eyes. So for a human the
decomposition is not a discovery problem but a **labelling** problem — map every face of whatever
mesh arrives onto a fixed taxonomy — and the labelling has a cheap, reliable oracle: **the
skeleton**. Every rigged humanoid already carries per-vertex bone weights, and the bones have
names (Mixamo, Rigify, SMPL-X, MPFB) that map onto the segments. A mesh with no rig gets one in
under a second from an MIT auto-rigger that emits Mixamo names.

That is what makes the "fast and forward" path possible: **the taxonomy is data, the rig is the
labeller, and the physiology, the materials and the clothing slots hang off the taxonomy, never
off the mesh.** A new human is a new mesh under the same labels; a new shirt is a material and a
colour in one slot.

```
canonical body schema (data, once)
  17 skin segments ──► JOS-3 per-segment skin T  (from WeatherSeries)
  garment slots     ──► material + clo + colour ──► ISO 7730 clothing T on the segment beneath
  hair, eyes        ──► optics only

any human mesh ──► rig (own, or Make-It-Animatable) ──► argmax weights ──► segment per face
               ──► skin / hair / garment per face (objects, slots, albedo; Find3D; a person)
               ──► split into one prim per (segment × layer) ──► .usdc + YAML, as every asset
```

---

## 4. The plan — lane `HU` in `docs/roadmap.md`

The steps live in the roadmap as lane **`HU`** (`HU.1`–`HU.13`, phase C), where their deps, sizes and
measurable criteria are maintained; this file keeps the reasoning. The order is the owner's:
**a person in RGB first, then the bare body in the infrared, then clothing**, repeated for a man, a
woman, a child, a police officer and a soldier, and only then the path for any downloaded human.

| step | what | why in this order |
|---|---|---|
| HU.1 | this survey and plan | the decisions below were taken on it |
| HU.2 | the body schema as data: 17 segments, garment slots, hair, eyes, bone-name maps; `kind: human` | everything after hangs off the taxonomy, never off a mesh |
| HU.3 | a man from MPFB2 in RGB, split by the rig into one prim per segment, standing in a real-sky scene | proves the asset path and the renderer before any physics is asked of it |
| HU.4 | the same bare body in LWIR: JOS-3 per segment, forced from the one `WeatherSeries` | the skin is the brightest thing about a person; it has to be right alone before clothing hides most of it |
| HU.5 | clothing on him: six materials, clo per garment, ISO 7730 per garment, colour → α and NIR only | the blue shirt / red shirt requirement, and ADR 0122's 20 K step now per region |
| HU.6 | a woman and a child as phenotype changes only | proves "the human is human": nothing new but numbers |
| HU.7 | a police officer: garments + equipment parts (retroreflective tape is a mirror in LWIR) | the first occupation, and the first low-ε hazard on a person |
| HU.8 | a soldier: NIR-compliant camouflage, helmet, plate carrier | the first material whose NIR curve is the point |
| HU.9 | any downloaded human: auto-rig (MIT) → argmax → skin / hair / garment by priority, Sapiens allowed | the second deliverable, after the schema has been exercised five times |
| HU.10 | a body that moves through the Isaac glue (`UsdSkel` probe; clips local only) | motion carries no emissivity, so it waits for the still body to be right |
| HU.11 | a firefighter and the occupations after | data only, no code, if HU.7–HU.8 held |
| HU.12 | validation on Charlotte-ThermalFace and FLIR ADAS v2 | the head segment against radiometric faces |
| HU.13 | portable instructions (`ingest-human`); the add-on's human mode is B14 in `blender_addon/PLAN.md` | the path must outlive this conversation |

### What is deliberately *not* in the plan

- **Fitting SMPL-X to scans** as the default labelling route. Technically the best, but the whole
  family is non-commercial with a surveillance exclusion and its commercial channel closed in April
  2026. The skeleton route reaches the same labels with MIT tools; SMPL tools may still be used
  case by case under decision 3.
- **Finger-level or face-sub-region physiology.** JOS-3 has one hand node and one head node. The
  face offsets in HU.12 are authored from measurements; a 65-node model is a later step if the
  validation shows the head node is the limiting error.
- **Sweat, shivering, wet clothing, breath plumes.** Later phenomena rows; the plume machinery
  (PH.6) exists for breath when wanted.
- **A soft blend across region seams.** Splitting at argmax seams gives a hard thermal step
  between, say, forearm and hand, where the body has a gradient. The upgrade path is the bone
  weights themselves as a per-vertex primvar read as a blend; it is not needed to get the ordering
  and the magnitudes right.
- **Shipping animations.** Walk cycles are retargeted locally from whatever source is convenient
  and never committed or published: they change where a surface is, not what it emits.

---

## 5. Decisions taken by the owner, 2026-10-06

| # | question | answer |
|---|---|---|
| 1 | where the work is tracked | **a new roadmap lane, `HU`** — it changes the physics core and the asset schema, unlike the add-on and the Unreal port |
| 2 | the shipped human's source | MPFB2 as recommended; the owner's starting set is **a man, a woman, a child, a police officer and a soldier**, each built RGB → IR body → clothing; firefighters and other occupations follow |
| 3 | non-commercial tools in the pipeline | **allowed** — "I am ok with different tools as I'll add the licences". Sapiens and SMPL-family tools may be used; the provenance YAML records which tools touched an asset so the owner can carry the licences |
| 4 | animation source | **not shipped at all** — "the animation does not act with emissivity". Clips are used locally; the question of their licence does not arise for the repository |
| 5 | cotton's emissivity | logged as spec issue **S66**; open for the spec owner, resolved by `HU.5` |

---

## 6. Not verified in this survey

MetaHuman and Reallusion EULA text (403s; wording from secondary reports); Renderpeople T&C
(domain blocked); FLIR ADAS v2 licence (404); CMU mocap terms; NICP, SAMesh and the
Meshcapade-wiki segmentation JSON licences (no LICENSE files); wool, leather, denim and hair LWIR
emissivities from a primary paper; whether Isaac's people USDs are split into skin and clothing
prims; whether `UsdSkel` deformation reaches the float32 position AOV in Isaac 6.0 (HU.7 is the
probe); the room temperature behind the PMC8955831 face figures.
