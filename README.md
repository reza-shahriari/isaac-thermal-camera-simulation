# irsim — a physically-based infrared camera simulator

<p align="center">
  <img src="docs/media/hero-phantom4-lwir-rgb.gif" alt="A DJI Phantom 4 Pro in flight, simulated in LWIR (left) beside its visible-light companion (right)" width="800">
</p>
<p align="center"><em>A DJI Phantom 4 Pro in flight: the simulated LWIR camera (left) and the visible
camera (right), the same moments through both lenses. Motors, battery and airframe each have their own
temperature.</em></p>

<p align="center">
  <a href="https://reza-shahriari.github.io/isaac-thermal-camera-simulation/"><b>Project site &amp; gallery</b></a> ·
  <a href="TECHNICAL_REPORT.md">Technical report</a> ·
  <a href="docs/physics-model.md">Physics model</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

**irsim** renders what a real infrared camera would see — thermal (LWIR), mid-wave (MWIR),
short-wave (SWIR) and near-infrared (NIR) — inside **NVIDIA Isaac Sim**. It doesn't just make images
that look thermal: every pixel starts as a surface temperature, turns into radiance in physical
units, passes through the atmosphere, the lens and the detector, and comes out as the counts a real
camera would record.

It is built for:

- **Synthetic training data** for detecting drones, aircraft, vessels and vehicles in infrared.
- **Sensor trade studies** — compare cameras, bands and lenses before buying hardware.
- **Perception development** for autonomous systems that carry thermal cameras.

## Highlights

### One scene, four bands

<img src="docs/media/four-bands-drone.webp" alt="The same quadrotor seen through visible, LWIR, MWIR, SWIR and NIR cameras" width="100%">

The same drone, weather and sun through five cameras. In LWIR it is hot motors against a cold sky;
in NIR it is a dark silhouette against a bright one. Adding a band is a config file, not new code.

### Every point has its own temperature

<img src="docs/media/pointwise-vs-objectwise.gif" alt="The same flight modelled with one temperature per object (left) and one per surface point (right)" width="100%">

Most simulators give an object a single temperature. irsim solves heat balance per surface point —
the sunlit deck warms, the shaded belly stays at air temperature, and heat flows along the arms from
the motors.

### Sky, sea and ground

<img src="docs/media/four-bands-ship.webp" alt="A vessel on open water in four infrared bands" width="100%">

Sky radiance that changes with elevation, clouds, a sea surface that reflects the sky, sun glint,
and slant-path atmosphere between the target and the camera.

## What it models

| | |
|---|---|
| **Thermal** | per-point energy balance with sun, sky, wind and weather; heat moving between connected parts; engines, exhausts, batteries, water and hot gas |
| **Materials** | spectral emissivity and reflectance that always add up correctly; angle-dependent emissivity for glass and water; a material library |
| **Atmosphere** | transmission and path radiance along the line of sight, sky radiance, clouds, haze |
| **Optics** | aperture, blur (MTF), vignetting, distortion, defocus, the warm lens housing |
| **Detector & noise** | microbolometers and cooled photon detectors, NETD, 3-D noise, fixed-pattern noise, dead pixels, flat-field correction |
| **Camera processing** | automatic gain control (including a FLIR Boson preset), detail enhancement, white-hot grayscale output |
| **Assets** | import real 3D models from a link or a file, split them into real parts and assign each part its real material |
| **Tooling** | a frame viewer where you click a pixel to read its true temperature, material and range; a validation suite against public thermal datasets |

Validated so far against physics identities and public thermal imagery — see the
[technical report](TECHNICAL_REPORT.md) for exactly what is checked and what is not.

## What's new

- **A dressed man, and a shirt of any colour** — the man now wears a T-shirt, trousers and shoes, each with its own infrared temperature solved on the skin beneath it; change the shirt's colour in one line of the asset file and the RGB camera sees a new shirt while the thermal camera sees the same cloth, warmer only by what the darker dye absorbs from the sun.
- **Every kind of cloud, in both cameras** — cumulus, congestus, stratocumulus, stratus, storm and cirrus are drawn per pixel for the visible camera and marched for the infrared one from the same cloud, and the two are checked against each other frame by frame.
- **The first human** — an adult man generated from MakeHuman's free assets, labelled into head, neck, chest, back, pelvis, arms, hands, legs and feet by his own skeleton, and standing in a real sky in RGB and LWIR; any rigged human can enter the same way.
- **DJI Matrice 100** — the developer quadcopter joins the asset library, split into its propellers, motors, arms, landing legs, battery, GPS mast and frame, each with an infrared material.
- **A Liberty ship** — a WWII cargo ship joins the asset library, split into its hull, hatches, masts, deckhouses, funnel, lifeboats, rudder and propeller, each with an infrared material.
- **Two FPV drones** — the DJI Avata 2 and DJI FPV join the asset library, split into their propellers, motors, arms, ducts, battery and camera, each with an infrared material.
- **Spectral materials from Blender** — the Blender add-on can now create a material from a measured spectrum or a single value, and plots any material across the infrared bands.

Full history in the [changelog](CHANGELOG.md).

## Quick start

```bash
# Use Isaac Sim's bundled Python (any Python >= 3.10 works for the physics core alone)
export PYTHON=/path/to/IsaacSim/_build/linux-x86_64/release/python.sh
make install
make check          # lint, type-check and the test suite — no GPU needed

# Render a scene in Isaac Sim (writes frames and an .mp4 into outputs/)
IRSIM_GPU=0 $PYTHON scripts/render_phantom4.py --frames 96

# Inspect the frames: click a pixel to see what is behind it
make viewer
```

Scenes live in [`configs/scenes/`](configs/scenes/), cameras in [`configs/sensors/`](configs/sensors/).
The physics core in `src/irsim/` is plain Python + NumPy and runs anywhere, without Isaac Sim.

## Project status

irsim is under active development. The aerial lane (drones and aircraft against the sky) comes
first, then maritime, then ground scenes. The [roadmap](docs/roadmap.md) holds the plan; the
[technical report](TECHNICAL_REPORT.md) holds the per-component status, validation evidence and
known limitations.

## Documentation

| | |
|---|---|
| [Project site](https://reza-shahriari.github.io/isaac-thermal-camera-simulation/) | gallery of renders, every document, every test |
| [Technical report](TECHNICAL_REPORT.md) | component status, validation tiers, limitations, commands, contributing |
| [Physics model](docs/physics-model.md) | the specification every equation comes from |
| [Decisions](docs/decisions/) | why each significant choice was made |
| [Roadmap](docs/roadmap.md) | what is next |
| [Changelog](CHANGELOG.md) | every shipped step |

## Licence

irsim is **source-available, not open source**. You may download it and use it unmodified for
personal, research and educational work. Modifying it, redistributing it and commercial use all need
written permission. See [LICENSE](LICENSE).
