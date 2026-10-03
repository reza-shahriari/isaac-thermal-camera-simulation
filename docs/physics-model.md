# A Physically-Based Infrared Camera Model for Isaac Sim

**Scope:** the mathematics of a multi-band IR/thermal camera simulator (LWIR, MWIR, SWIR, NIR) built on a single shared core, targeted at NVIDIA Isaac Sim 6.0 and portable to Unreal Engine. This is a specification, not an implementation.

**Research date:** September 2026. Isaac Sim 6.0 GA and the `omni.rtx.spg` Sensor Processing Graph are the current implementation surface. Sources are listed in §17 and referenced inline as `[Rn]`.

---

## Table of contents

1. [What "real" means here, and the fidelity ladder](#1-what-real-means)
2. [The master equation](#2-the-master-equation)
3. [Radiometry core: Planck, band integration, inversion](#3-radiometry-core)
4. [Surface model: emissivity, Kirchhoff, Fresnel](#4-surface-model)
5. [Scene radiance: the thermal rendering equation](#5-scene-radiance)
6. [Temperature field: where the signal actually comes from](#6-temperature-field)
7. [Atmosphere](#7-atmosphere)
8. [Optics: irradiance, MTF, stray radiance](#8-optics)
9. [Detector: photon devices and bolometers](#9-detector)
10. [Noise: the 3D noise model and NETD](#10-noise)
11. [Signal chain: NUC, AGC, palette](#11-signal-chain)
12. [Band specialisation: one core, four cameras](#12-band-specialisation)
13. [Isaac Sim 6.0 implementation mapping](#13-isaac-sim)
14. [Unreal Engine mapping](#14-unreal)
15. [Validation protocol](#15-validation)
16. [Reference parameter tables](#16-parameters)
17. [Sources](#17-sources)

---

<a name="1-what-real-means"></a>
## 1. What "real" means here, and the fidelity ladder

There is no single "IR camera equation." What exists is a **chain of physical models**, each of which can be swapped for a cheaper approximation. "Real" means every link is a recognised physical model whose units close, and every approximation is a deliberate, documented choice rather than an accident.

The chain, in signal order:

```
temperature field  →  surface radiance  →  atmospheric path  →  optics
      |                     |                    |                |
 heat transfer       Planck + Kirchhoff     Beer-Lambert      f/# + MTF
                                                                  |
                                          detector  →  noise  →  ISP  →  pixels
                                              |          |        |
                                        QE / bolometer  3D noise  NUC + AGC
```

The honest fidelity ladder, so you can choose where to stop:

| Level | Temperature | Reflection | Atmosphere | Detector | Good for |
|---|---|---|---|---|---|
| **L0** | Painted textures | none | none | Gaussian noise | Looks thermal. Worthless for engineering. |
| **L1** | Static per-material T | ambient constant | constant τ | NETD + FPN | Smoke tests, UI development |
| **L2** | Lumped-capacitance ODE | Kirchhoff + sky-view factor | band-averaged Beer-Lambert | full FPA chain + 3D noise | **Target here.** Detection/tracking dev, sensor trades |
| **L3** | 1D slab energy balance (THERM-class) | path-traced in T-domain | band model (MODTRAN LUT) | + MTF cascade, bolometer dynamics | Range prediction, sim2real training data |
| **L4** | 3D FEM (MuSES-class), internal heat | full spectral BRDF, polarimetric | line-by-line | measured SITF | Signature prediction, countermeasures |

DIRSIG (RIT) and MuSES (ThermoAnalytics) are the reference L3/L4 systems. MuSES calculates internal and external temperatures in full 3D, accounting for internally generated heat and internal circulation, using a finite-volume method to solve the energy balance for conduction, convection and radiation [R4][R5]. You are not going to reproduce that inside Isaac Sim, and you don't need to. **L2 with a clean upgrade path to L3 is the right target**, and it is already above what real-time engine plugins typically deliver.

One structural decision drives everything: **separate the temperature problem from the radiance problem.** They have different timescales (thermal: seconds to hours; radiometric: per frame) and different solvers. Every serious system does this — in DIRSIG the temperature solver is an explicitly pluggable subsystem that the radiometric core consumes [R3].

---

<a name="2-the-master-equation"></a>
## 2. The master equation

Everything in this document serves one equation. The digital number at pixel $(i,j)$:

$$
\mathrm{DN}_{ij} = \mathcal{Q}\Big[\, g_{ij}\,\mathcal{S}\big(\Phi_{ij}\big) + o_{ij} + n_{ij} \Big]
$$

where $\Phi_{ij}$ is the in-band power (or photon rate) collected by that pixel:

$$
\Phi_{ij} = \frac{\pi A_d \tau_{\text{opt}}}{4F^2+1}\cos^4\theta_{ij}
\int_{\lambda_1}^{\lambda_2} R(\lambda)\,\big[\mathrm{MTF} \ast L_{\text{sensor}}\big](\lambda,i,j)\, d\lambda
\;+\; \Phi_{\text{self}}
$$

and the sensor-reaching spectral radiance decomposes into three terms:

$$
L_{\text{sensor}}(\lambda) =
\underbrace{\tau_{\text{atm}}(\lambda,d)\,\varepsilon(\lambda,\theta)\,B(\lambda,T_s)}_{\text{self-emission}}
+\underbrace{\tau_{\text{atm}}(\lambda,d)\big(1-\varepsilon(\lambda,\theta)\big)L_{\downarrow}(\lambda)}_{\text{reflected}}
+\underbrace{\big(1-\tau_{\text{atm}}(\lambda,d)\big)B(\lambda,T_{\text{air}})}_{\text{path radiance}}
$$

This three-term form is the standard thermal-IR radiative transfer model, with distance-dependent transmittance $\tau(\lambda;d) = [\tau_{1\mathrm{m}}(\lambda)]^{d}$ [R11]. The same structure appears in MODTRAN-compatible formulations, where $\tau$ is the surface-to-sensor transmission, $B(T)$ the Planck function of surface temperature, $\rho$ the directional-hemispherical reflectance, and $U$ the atmospheric path radiance [R12].

**Symbols**

| Symbol | Meaning | Units |
|---|---|---|
| $B(\lambda,T)$ | Planck spectral radiance | W·m⁻²·sr⁻¹·µm⁻¹ |
| $\varepsilon(\lambda,\theta)$ | directional spectral emissivity | – |
| $L_\downarrow(\lambda)$ | downwelling radiance onto the surface | W·m⁻²·sr⁻¹·µm⁻¹ |
| $\tau_{\text{atm}}(\lambda,d)$ | atmospheric transmittance over path $d$ | – |
| $\tau_{\text{opt}}$ | optics transmittance | – |
| $R(\lambda)$ | normalised spectral response (QE × filter × window) | – |
| $F$ | working f-number | – |
| $A_d$ | detector active area | m² |
| $\theta_{ij}$ | field angle of pixel $(i,j)$ | rad |
| $\Phi_{\text{self}}$ | self-emission of optics/housing reaching the pixel | W |
| $\mathrm{RI}_{ij}$ | relative illumination of pixel $(i,j)$: $\cos^4\theta_{ij}$ × measured mechanical vignetting | – |
| $T_{\text{shutter}}$ | temperature of the flat-field shutter at the last FFC | K |
| $g_{ij}, o_{ij}$ | per-pixel gain and offset (post-NUC residual) | – |
| $\mathcal{S}$ | detector transfer (photoelectrons or bolometer response) | – |
| $\mathcal{Q}$ | quantisation | – |

The $\cos^4\theta$ term is natural vignetting; real IR optics add mechanical vignetting on top, and that is normally measured rather than derived.

> **Convention warning.** Some references write the aperture factor as $1/(4F^2)$, others as $1/(4F^2{+}1)$. The `+1` form is exact for a Lambertian extended source through a circular aperture ($E = \pi L \sin^2\theta_{\max}$); the other is the paraxial limit. At $F/1.0$ — routine for uncooled LWIR — the difference is 20%. **Use `+1`.** This is the most common radiometric bug in home-grown IR simulators.

---

<a name="3-radiometry-core"></a>
## 3. Radiometry core

### 3.1 Planck's law

Spectral radiance of a blackbody per unit wavelength:

$$
B(\lambda,T) = \frac{c_{1L}}{\lambda^5\left(e^{c_2/\lambda T}-1\right)}
$$

with, in practical units ($\lambda$ in µm, $B$ in W·m⁻²·sr⁻¹·µm⁻¹):

$$
c_{1L} = 2hc^2 = 1.19104297\times10^{8}\ \text{W·µm}^4\text{m}^{-2}\text{sr}^{-1},
\qquad
c_2 = \frac{hc}{k_B} = 1.4387769\times10^{4}\ \text{µm·K}
$$

**Photon form** — mandatory for photon detectors (MWIR/SWIR/NIR), because quantum efficiency is a per-photon quantity:

$$
B_q(\lambda,T) = \frac{c_{1q}}{\lambda^4\left(e^{c_2/\lambda T}-1\right)},
\qquad c_{1q} = 2c = 5.995849\times10^{26}\ \text{ph·s}^{-1}\text{µm}^{3}\text{m}^{-2}\text{sr}^{-1}
$$

Unit test worth writing: at $\lambda=10$ µm, $T=300$ K, $B = 9.925$ W·m⁻²·sr⁻¹·µm⁻¹ and $B_q = 4.996\times10^{20}$ ph·s⁻¹·m⁻²·sr⁻¹·µm⁻¹, and $B_q \cdot hc/\lambda = B$ to machine precision.

Total exitance: $M=\sigma T^4$, $\sigma = 5.670374\times10^{-8}$ W·m⁻²·K⁻⁴; Lambertian radiance $L=\sigma T^4/\pi$. Peak: $\lambda_{\max}T = 2897.77$ µm·K — a 300 K scene peaks at 9.66 µm (LWIR), a 500 K exhaust at 5.8 µm, a 1000 K flare at 2.9 µm (MWIR). Band choice is a targeting decision, not a preference.

### 3.2 Band integration — precompute, don't evaluate per pixel

You need $L_B(T) = \int_{\lambda_1}^{\lambda_2} R(\lambda) B(\lambda,T)\,d\lambda$.

**(a) Closed-form fractional exitance.** With $x = c_2/(\lambda T)$, the fraction of total blackbody exitance below $\lambda$:

$$
F_{0\to\lambda T} = \frac{15}{\pi^4}\sum_{n=1}^{\infty}\frac{e^{-nx}}{n}\left(x^3+\frac{3x^2}{n}+\frac{6x}{n^2}+\frac{6}{n^3}\right)
$$

Converges to double precision in 3–8 terms for $x>2$ (all realistic scene temperatures in LWIR/MWIR). Then

$$
L_B(T) = \frac{\sigma T^4}{\pi}\Big[F_{0\to\lambda_2 T}-F_{0\to\lambda_1 T}\Big]
$$

Use when $R(\lambda)$ is approximately a top-hat. Good to ~1% for most uncooled LWIR cores.

**(b) Precomputed LUT — recommended.** Tabulate $L_B(T)$ on a uniform grid $T\in[200,1000]$ K at 0.05–0.1 K spacing, per band, with the true $R(\lambda)$ folded in by numerical quadrature (Simpson on a 0.01 µm grid is ample). float32, ~16k entries, 64 kB per band. Linear interpolation in $T$ is smooth well below NETD.

*(Your existing Unreal LUT approach is the right call and carries over unchanged.)*

**(c) Runtime quadrature.** Only needed if $\varepsilon(\lambda)$ has real spectral structure that varies per pixel — i.e. hyperspectral work. Not needed for an imaging camera.

> **Precision trap.** If temperature or band radiance passes through a **float16** render target, you lose. At $T\approx300$ K, fp16 spacing is $2^{8}\cdot2^{-10} = 0.25$ K — five times coarser than a 50 mK NETD. Encode as `(T − T_ref)/T_span` in float32, or use a float32 AOV. This is the most common failure mode in engine-based thermal cameras.

### 3.3 Inversion: apparent temperature

A radiometric camera does not output temperature; it outputs *apparent* (radiometric, brightness) temperature:

$$
T_{\text{app}} = L_B^{-1}\big(L_{\text{measured}}\big)
$$

$L_B$ is monotonic, so invert by binary search on the same LUT, or store the inverse. The gap between $T_{\text{app}}$ and true kinetic $T_s$ *is* the emissivity + reflection + atmosphere error, and **reproducing that gap is the entire point of a physical simulator.** A simulator that hands back ground-truth temperature has not simulated a camera.

### 3.4 Thermal derivative — needed for NETD

$$
\frac{\partial B}{\partial T}(\lambda,T)=B(\lambda,T)\cdot\frac{c_2}{\lambda T^2}\cdot\frac{e^{x}}{e^{x}-1},\qquad x=\frac{c_2}{\lambda T}
$$

In-band: $\left(\partial L/\partial T\right)_B=\int R(\lambda)\,\partial B/\partial T\,d\lambda$ — tabulate alongside $L_B(T)$.

This derivative is why **NETD is not a constant**. The blackbody thermal derivative rises with temperature, so the same detector noise in volts corresponds to a smaller temperature error on a hot target; one measured uncooled core went from ≈39 mK NETD at ambient to ≈23 mK against a 100 °C blackbody [R9]. If your simulator applies a flat NETD in Kelvin across the scene it is wrong in a way that matters for hot-object detection.

> **Rule:** add noise in radiance or electron space, then convert. Never add Kelvin-space Gaussian noise to a temperature buffer.

---

<a name="4-surface-model"></a>
## 4. Surface model

### 4.1 Kirchhoff's law

For an opaque surface in local thermodynamic equilibrium:

$$
\varepsilon(\lambda,\theta,\phi)=\alpha(\lambda,\theta,\phi)=1-\rho(\lambda,\theta,\phi)
$$

where $\rho$ is the **directional-hemispherical reflectance** (DHR). This is the bridge between the renderer's material system and radiometry, and it is what lets a thermal camera see reflections at all.

DIRSIG states the operational form directly: the model computes the complementary reflectance or emissivity from the other, so users never supply both; when a BRDF model is used, directional-hemispherical emissivity is derived as `DHE = 1 − DHR`, with DHR obtained by hemispherically integrating the BRDF at the given view geometry [R3].

**Implication for material authoring:** author **one** spectral optical property per band and derive the other. Authoring both is how simulators end up violating energy conservation.

The rule is per wavelength, not per file. A material may give its one quantity as a measured curve where one exists, a per-band value where it does not, and a grey value beyond both (§12.3). It may even tabulate one curve segment as the opaque complement, since short-wave libraries measure reflectance and long-wave libraries measure emission. What is forbidden is two values at the same wavelength, whether they are ε and ρ or two copies of ε.

### 4.2 Angular dependence — the thing that makes it look right

Emissivity falls off toward grazing incidence for dielectrics, which is why a smooth curved object shows a cooler-looking rim in LWIR at uniform temperature. A January 2026 validation study makes the point sharply: the imaginary part of the complex refractive index governs absorption and, through Kirchhoff's law, the angular emissivity responsible for radiometric falloff toward a sphere's edges — a purely real refractive index does not reproduce it [R1].

**Level A — Fresnel from complex refractive index (physical).**
With $\tilde n(\lambda)=n(\lambda)+ik(\lambda)$ and $Z=\tilde n^2-\sin^2\theta_i$:

$$
\sqrt{Z}=A+jB,\qquad
A=\sqrt{\frac{|Z|+\Re(Z)}{2}},\qquad
B=\operatorname{sgn}\big(\Im(Z)\big)\sqrt{\frac{|Z|-\Re(Z)}{2}}
$$

$$
r_\perp=\frac{\cos\theta_i-(A+jB)}{\cos\theta_i+(A+jB)},\qquad
r_\parallel=\frac{(A+jB)-\tilde n^2\cos\theta_i}{(A+jB)+\tilde n^2\cos\theta_i}
$$

$$
R_\perp=r_\perp r_\perp^{*},\quad R_\parallel=r_\parallel r_\parallel^{*},\quad
R=\tfrac12\big(R_\perp+R_\parallel\big),\quad \varepsilon(\lambda,\theta)=1-R(\lambda,\theta)
$$

The $A/B$ reparameterisation is not cosmetic. Naïve complex square roots carry a branch ambiguity: both branches satisfy the Helmholtz equation, but passivity requires the transmitted field to decay with depth, i.e. $\Im(k_z)>0$. Choosing wrong produces reflectance discontinuities across wavelength, instabilities near grazing incidence, and reflectances exceeding unity. The $A/B$ form selects the admissible branch automatically [R1]. In the IR this matters because $k$ is large for many materials. Complex indices are available from curated databases (refractiveindex.info); the paper validates against water at 10 µm ($n=1.218$, $k=0.0508$) and against bismuth.

Note also a counter-intuitive consequence the same work confirms: in strongly absorbing media there is **no sharp total internal reflection** — reflectance stays below 1 at every incidence angle [R1].

**Level B — Empirical angular falloff (cheap, adequate for L2).**

$$
\varepsilon(\theta)=\varepsilon_0\Big[1-a\big(1-\cos\theta\big)^{p}\Big]
$$

$a\approx0.15$–$0.35$, $p\approx4$–$6$ for painted metals and plastics; nearly flat ($a\to0$) for rough dielectrics like asphalt and vegetation. Fit $(a,p)$ once per material class against Level A and bake it. Two instructions in a shader.

**Level C — Constant $\varepsilon$.** Acceptable only for rough, high-emissivity surfaces ($\varepsilon>0.93$) within ±50° of normal. Vehicle bodies, glass and water all violate this.

### 4.3 Specular vs diffuse in the thermal IR

An IR reflection lobe is generally **narrower and stronger** than the same material's visible lobe: roughness that scatters 0.5 µm light diffusely can be optically smooth at 10 µm (when $\sigma \ll \lambda/8$). Consequences:

- Painted bodywork and glass show near-mirror sky reflections in LWIR. Roofs and hoods read very cold on a clear night because they reflect a very cold sky.
- Asphalt and concrete stay near-Lambertian even in LWIR.
- **Do not reuse visible-band roughness values.** Roughness must be a per-band material parameter. This strongly affects what an automotive perception stack sees.

### 4.4 Semi-transparent materials

Glass is opaque in LWIR ($\varepsilon\approx0.85$–0.92) but **transparent in SWIR and NIR**. A windshield in LWIR shows the windshield's own temperature; in SWIR it shows the driver. Encode as a per-band triple $(\varepsilon,\rho,\tau)$ with $\varepsilon+\rho+\tau=1$, handling $\tau>0$ as a second ray in the SWIR/NIR path. Make sure the material schema can express this from day one — retrofitting it is painful, and it is one of the headline capabilities of multi-band simulation.

---

### 4.5 Surface state — the property a material file must name

$\varepsilon$ is a property of the *surface*, not of the substance underneath it. One measurement campaign on aluminium window profiles reports **0.834–0.856** normal total emissivity for anodised exterior surfaces and **0.055–0.82** across untreated all-aluminium cavities in the same frames [R39] — one metal, one paper, a fifteen-fold spread. Polished aluminium is 0.04 at 8 µm [R39]; rough wrought iron is 0.94 against 0.28 polished [R47]. Oxidation dominates, roughness comes second, and both raise $\varepsilon$.

**So "aluminium" is not a material specification.** A material file must name the substance *and* its surface state — polished, machined, oxidised, anodised, painted, sandblasted, weathered — and cite the state its $\varepsilon$ was measured on. The §16.2 values are surface states that were never written down.

Two things look like they should predict $\varepsilon$, and do not:

**(a) Visible colour does not.** Paint emissivity is essentially independent of pigment colour from roughly 2–12 µm; a flat white coating routinely exceeds a gloss black one, because gloss and binder chemistry matter where colour does not. Black and white automotive paint therefore carry the **same** $\varepsilon$ per band and differ only in $\alpha_{\text{sol}}$ (§6.2), which is where colour genuinely acts. The folk rule "the duller and blacker a material is, the higher its emissivity" is half right: the *duller* half is roughness and is real; the *blacker* half is a visible-band intuition that does not survive into the thermal infrared. An author who "corrects" the library toward it makes the model worse, so the library holds this as a test rather than as a comment.

**(b) Thickness does not, above a knee.** Emitted radiation escapes from a layer of depth $d_{1/e}=\lambda/(4\pi k)$, so once a body is a few of those thick, adding more changes nothing at all. Computed from this project's own $n/k$ tables, 99 % of the emission leaving a surface comes from the top:

| material | LWIR (10 µm) | MWIR (4 µm) | SWIR (1.5 µm) |
|---|---|---|---|
| aluminium | 0.04 µm | 0.04 µm | 0.04 µm |
| paint (PMMA proxy) | 86 µm | 333 µm | 158 mm |
| glass (fused silica) | 44 µm | 24 mm | transparent |
| water | 72 µm | 318 µm | 2.4 mm |

A 55 cm slab and a 57 cm slab are optically identical for every one of them. The exception is a genuinely thin film: an anodic oxide on aluminium gives $\varepsilon\approx0.15$ at 1 µm of film, 0.45 at 2 µm and 0.91 above about 15 µm [R40]. That dependence is real and steep — and it is unobtainable, because no mesh carries a film thickness. **Author it as a named surface state; never solve it from a thickness.** `thermal.thickness_m` is the heat-capacity parameter of §6.1, and nothing optical may read it.

**Temperature.** $\varepsilon$ varies with temperature too — metals rising roughly in proportion to $T$, non-metals falling. That is a lookup $\varepsilon(\lambda,T)$ and not a circularity, since §6 produces $T$ without needing it. Below ~600 K the variation is smaller than the uncertainty the authored values already carry and is ignored here; at plume and fire temperatures it is not (Appendix A).

### 4.6 Weather changes the surface state

§4.5 makes the surface state the property a material names. Weather changes that state during a scene,
and for many surfaces the change is the largest optical event they ever see.

**Wet.** Water's own numbers in the §4.5 table decide it. Water's 1/e absorption depth is 3–19 µm across
LWIR [R100], so a film of a few tens of micrometres is, in LWIR, *water*. Its emissivity follows water's
Fresnel curve (§4.2, the same optics `PH.3` uses for a puddle) rather than the dry material's: 0.987 at
nadir, 0.66 at 80°. Its reflection is **specular**, because the film levels the roughness that made the
dry surface diffuse. Water's LWIR reflectance is 0.013 at normal incidence, 0.05 at 60°, 0.20 at 75°,
0.34 at 80° and 0.57 at 85°. A wet road or a puddle ahead of a vehicle camera is therefore a mirror for
the cold sky at the angles that camera sees it — the dark sheen of wet-road LWIR imagery, which tends to
the sky's radiance at long range [R113] — even though the road's emissivity at nadir barely changes.
Measured on urban surfaces, wetting raises asphalt, concrete and granite emissivity by more than 0.02 in
8–10 µm, 0.9–2.3 K of apparent temperature [R112]. In MWIR the same film is up to 89 µm deep and in SWIR
millimetres deep, so a film that is optically water in LWIR is a partly transparent layer in the short
bands.

For a film of depth $d$ covering a fraction $f$ of the surface, per band, with
$\delta_B = \lambda/(4\pi k_{\text{water}})$ the band's 1/e depth and $\mu$ the cosine of the refracted
angle:

$$
\varepsilon_{\text{film}} = \big(1-\rho_{w}(\theta)\big)\Big[\big(1-e^{-d/(\delta_B\mu)}\big)
+ e^{-d/(\delta_B\mu)}\,\varepsilon_{\text{dry}}\Big],\qquad
\varepsilon_{\text{wet}} = (1-f)\,\varepsilon_{\text{dry}} + f\,\varepsilon_{\text{film}},\qquad
f = \min(1, d/h)
$$

The first factor is the film's own emission plus the substrate's through it, both behind water's top
interface. $h$ is the surface's macrotexture depth (asphalt about 0.5–1 mm), and $f$ is also the
specular fraction. The form neglects interference and the film–substrate interface, both small on a
rough substrate. It is **approximate**: no published model was found that predicts a specific wet
road, so it is held to its limits ($d \to 0$ recovers the dry surface, $d \gg \delta_B$ recovers water)
and to the measured rise. In the reflective bands a wet surface is darker, because light trapped by
total internal reflection in the film is absorbed by the substrate [R111], and the same film model
gives that.

The film already exists: `PH.1` keeps it per cell in kg m⁻², rain fills it and evaporation empties it.
What is missing is that nothing optical reads it. The rule: **the film is the state.** A surface is not
authored as "wet asphalt"; it is asphalt with a film the solver carries.

**Dew and frost.** A surface below the dew point gains film by condensation, and below 0 °C it gains
frost, whose emissivity is ice's. A clear night's dew is 0.14 ± 0.12 mm [R114], already optically water
in LWIR. That is why a car parked overnight reads, at dawn, as a uniform water-emissivity skin over
every panel, metal included. The latent model computes the negative evaporation; dew onto a *dry*
surface is not yet admitted (`irsim.thermal.latent`).

**Snow cover** is the snow material capped at 0 °C (§6, `PH.10`). Fine-grained snow is 0.98–0.99 emissive
over 8–13 µm and nearly Lambertian; coarse grains fall to about 0.93 at 75° viewing [R106].

---

<a name="5-scene-radiance"></a>
## 5. Scene radiance: the thermal rendering equation

### 5.1 The general form

The radiance leaving a surface point $\mathbf{x}$ toward direction $\omega_o$:

$$
L_o(\mathbf{x},\omega_o,\lambda)=
\underbrace{\varepsilon(\lambda,\omega_o)B(\lambda,T_s(\mathbf{x}))}_{\text{emitted}}
+\underbrace{\int_{\Omega} f_r(\mathbf{x},\omega_i,\omega_o,\lambda)\,L_i(\mathbf{x},\omega_i,\lambda)\cos\theta_i\,d\omega_i}_{\text{reflected}}
$$

This is the ordinary rendering equation with an emission term that is **physically derived from temperature** rather than authored. That is the whole trick. Everything a path tracer already knows how to do — visibility, BRDF sampling, multiple bounces — remains valid; you have only changed what the light sources are.

Comprehensive IR signature simulators build exactly this. Their radiometric models account for spectral emissivity, spatial radiance distribution, specular reflection, reflected direct sunlight, reflected ambient light and atmospheric degradation, with all components combined in a spectral representation so that band-integrated radiance images can be produced for an arbitrary number of user-defined bandpasses [R2]. That last clause is the architecture you want: **one spectral scene, many band outputs.**

### 5.2 The two-regime split

The single most important architectural fact about multi-band IR:

| Band | λ (µm) | Dominant source at 300 K | Illumination needed |
|---|---|---|---|
| NIR | 0.75–1.0 | reflected | sun / moon / airglow / headlights |
| SWIR | 1.0–1.7 (ext. 2.5) | reflected | sun / **airglow** / lasers |
| MWIR | 3.0–5.0 | **both** | sun (glint) **and** self-emission |
| LWIR | 8.0–12.0 (7.5–13.5) | self-emitted | none (scene *is* the source) |

The crossover — where solar-reflected and self-emitted radiance from a 300 K surface are comparable — sits near **3.5–4.5 µm** in daylight, moving with albedo and sun angle. MWIR straddles it, which is why MWIR is the hardest band to model and the one where day/night behaviour flips most dramatically.

Implementation consequence: **one shading core, two illumination paths.** Both are always present in the code; per band, one is typically negligible and can be culled by a compile-time flag.

### 5.3 Downwelling radiance $L_\downarrow$ — do not skip this

In LWIR, the reflected term is only $(1-\varepsilon)\approx0.05$–$0.2$ of the signal, but the sky is *extremely* cold, so it produces large apparent-temperature swings on low-emissivity surfaces. Modelling a constant ambient sky is a well-documented error: a clear dry sky reaches about **−40 °C at 15° elevation in LWIR, while the same sky reads above +10 °C in MWIR** [R13]. That difference between bands is real, not a modelling artefact, and it means a constant sky background badly mismodels a system's ability to detect small aerial targets.

Three implementations, cheapest first:

**(a) Sky-view-factor blend (L2).** Precompute per-pixel sky visibility $V_s\in[0,1]$ from the surface normal and local occlusion (an AO-style term, or a cosine-weighted hemisphere factor):

$$
L_\downarrow \approx V_s\,L_{\text{sky}}(\theta_{\text{zen}}) + (1-V_s)\,L_{\text{ground}}
$$

with $L_{\text{sky}}$ from an elevation-parameterised effective sky temperature:

$$
T_{\text{sky}}(\theta_{\text{zen}}) = T_{\text{air}} - \Delta T_{\text{clear}}\cdot\big(1-\text{cloud}\big)\cdot\cos^{q}(\theta_{\text{zen}})
$$

Practical LWIR values: $\Delta T_{\text{clear}}\approx 55$–$70$ K at zenith for a dry clear sky, dropping toward 5–10 K under thick overcast or high humidity; $q\approx0.5$–$1.0$. Under overcast, $T_{\text{sky}}\to T_{\text{air}}$ and the reflected term nearly vanishes — which is exactly why thermal images look "flat" on cloudy days. That behaviour falling out of your model for free is a good sanity check.

**In a reflective band the ground and the cloud are lit, not only warm** (spec issue S55). The
ground's own emission $L_B(T_{\text{ground}})$ is all of $L_{\text{ground}}$ in LWIR and next to nothing
in NIR and SWIR, where a sunlit ground returns
$L_{\text{ground}} = L_B(T_{\text{ground}}) + \rho_{B,\text{ground}}\,E_B/\pi$ with
$E_B = f_{\text{dir}}\,\text{DNI}\sin h + f_{\text{diff}}\,\text{DHI}$, the weather's own irradiance
through the band's shares of the beam and of the (Rayleigh-blue) sky. Leave it out and every
downward-facing surface, $V_s \to 0$, reflects nothing: a white underside renders black against a
bright sky. A cloud likewise scatters sunlight. Seen from below, its base is drawn as a Lambertian
reflector $R(\tau,\mu_0)\,E_B/\pi$ with the two-stream $R$ the visible companion uses, so the two
bands agree about which clouds are bright. Otherwise an opaque cloud in SWIR swaps the column's
thermal emission for its own, both near zero, and the clouds vanish (ADR 0153).

**(b) Low-resolution irradiance cubemap (L2.5).** Render a 32×32×6 cubemap of scene band-radiance (including sky) from a few probe positions, prefilter into an irradiance map. This gets you building/vehicle self-heating reflections without full path tracing. Cheap, and it is what makes urban thermal scenes stop looking synthetic.

**(c) Path-trace in the radiance domain (L3).** Let the renderer do it, with materials whose emission is $\varepsilon B(T)$ and whose BRDF is the IR BRDF. Correct, and expensive.

### 5.4 The solar term (MWIR/SWIR/NIR)

$$
L_{\text{sun}} = \frac{\rho(\omega_s,\omega_o)}{\pi}\,E_{\text{sun}}(\lambda)\,\tau_{\text{atm}}^{\text{sun}}(\lambda)\cos\theta_s\,S(\mathbf{x})
$$

$S$ is the shadow term, $E_{\text{sun}}$ is top-of-atmosphere spectral irradiance attenuated along the solar path. In MWIR, **solar glint off water, glass and painted metal saturates detectors** and is a first-order phenomenon, not a detail. Model it with the specular lobe, not a Lambertian albedo.

### 5.5 Night illumination for SWIR/NIR

This is the band-specific piece people forget. At night in SWIR the scene is lit by **airglow (nightglow)** — chemiluminescence in the upper atmosphere, largely OH emission. Its spectrum peaks between roughly 1 and 1.8 µm, right in the InGaAs response band; at full moon, moonlight and airglow radiation densities are comparable, and on moonless nights SWIR illumination exceeds visible illumination by about an order of magnitude [R7][R8]. Reported airglow irradiance values span roughly 3.5–39 nW/cm², with substantial spatial, temporal and seasonal variability [R10].

For simulation, model night SWIR illumination as:

$$
E_{\text{night}}(\lambda) = E_{\text{airglow}}(\lambda)\cdot k_{\text{cloud}} + E_{\text{moon}}(\lambda,\phi_{\text{moon}}) + E_{\text{star}}(\lambda) + E_{\text{artificial}}
$$

Treat $E_{\text{airglow}}$ as a spectrally shaped hemisphere source with a configurable level (default ~10 nW/cm², range 3.5–39) and a slow temporal variation. Note that airglow is **not** blocked in the same way as moonlight — cloud attenuates it, but it does not have a directional shadow.

Getting this right is what makes a SWIR channel behave like a real SWIR camera at night rather than like a dark visible camera.

---

<a name="6-temperature-field"></a>
## 6. Temperature field: where the signal actually comes from

In LWIR, **your image is your temperature field.** No amount of radiometric sophistication rescues a bad $T_s$. This section deserves more of your effort than any other.

### 6.1 The energy balance

For a surface element with area-normalised heat capacity $C = \rho c_p \delta$ (J·m⁻²·K⁻¹) over an effective thickness $\delta$:

$$
C\frac{dT_s}{dt} =
\underbrace{\alpha_{\text{sol}}\,Q_{\text{sol}}}_{\text{absorbed solar}}
+\underbrace{\varepsilon Q_{\text{LW}\downarrow}}_{\text{absorbed sky/env}}
-\underbrace{\varepsilon\sigma T_s^4}_{\text{emitted}}
-\underbrace{h\,(T_s-T_{\text{air}})}_{\text{convection}}
-\underbrace{k\frac{\partial T}{\partial z}\Big|_{z=0}}_{\text{conduction}}
+\underbrace{q_{\text{int}}}_{\text{internal}}
$$

with $\alpha_{\text{sol}}$ the solar *absorptivity* (absorbed solar is $\alpha_{\text{sol}}Q_{\text{sol}}$; an earlier draft wrote it as a complement with a stray exponent — spec issue S2). This is the same balance DIRSIG's THERM solver implements: a 1-D slab model taking conduction, convection and radiation into account to estimate surface temperature, driven by material thermodynamic properties plus weather data [R3][R6].

The identical structure appears in urban-scale surface energy balance work as a transient equation coupling radiative fluxes with thermal storage and convection [R14], and in planetary thermophysical models where the boundary condition is
$(1-A_B)\big((1-S)\psi F_{\text{SUN}} + F_{\text{SCAT}}\big) + (1-A_{TH})F_{\text{RAD}} + k(dT/dx)_{x=0} - \varepsilon\sigma T^4_{x=0}=0$ [R15]. It's the same physics with different labels; the shadow flag $S$ and illumination-cosine term $\psi$ are worth copying directly.

### 6.2 The parameters you must author

DIRSIG's parameter set is the minimal complete one [R3]:

| Symbol | Property | Units | Notes |
|---|---|---|---|
| $h$ | convection heat transfer coefficient | W·m⁻²·K⁻¹ | **Largest single source of uncertainty** |
| $k$ | thermal conductivity | W·m⁻¹·K⁻¹ | |
| $\varepsilon$ | thermal emissivity | – | spectral; drives both heat balance and radiance |
| $\alpha_{\text{sol}}$ | solar absorptivity | – | |
| $c_p$ | specific heat | J·kg⁻¹·K⁻¹ | |
| $\rho$ | mass density | kg·m⁻³ | |

On $h$, DIRSIG is explicit that it is difficult to parameterise because it depends on surface roughness, fluid viscosity, vertical temperature stability and humidity, and it tends to be the largest source of uncertainty in temperature prediction; MuSES lets users set it explicitly while THERM computes it at runtime from conditions such as wind speed [R3]. Follow THERM: derive it.

A serviceable forced/free-convection parameterisation for exterior surfaces:

$$
h = \max\Big(h_{\text{free}},\; a + b\,v_{\text{wind}}^{\,n}\Big),
\qquad h_{\text{free}} = c\,\big|T_s-T_{\text{air}}\big|^{1/3}
$$

with $a\approx 5$, $b\approx 4$, $n\approx 0.8$ (SI, wind in m/s), $c\approx1.5$. For a moving vehicle, $v_{\text{wind}}$ must include vehicle speed — a car at 100 km/h has a completely different hood temperature from the same car parked. **This is a first-order effect for automotive IR and it is trivial to add.**

### 6.3 Thermal inertia — the diurnal fingerprint

Define thermal inertia $P=\sqrt{k\rho c_p}$ (J·m⁻²·K⁻¹·s⁻¹ᐟ²). It controls how strongly a surface swings over a day:

- **Low $P$** (dry sand, foliage, thin painted metal): fast, large swings. Hot by day, cold by night.
- **High $P$** (water, concrete, wet soil, engine blocks): sluggish, damped. Cooler by day, warmer at night.

**Thermal crossover** — the times near dawn and dusk when different materials pass through equal apparent temperature and contrast collapses — is a direct consequence, and is one of the most operationally important phenomena in thermal imaging. If your simulator does not reproduce a contrast collapse near sunrise/sunset, your temperature model is not running. Make "does the scene go flat at 0600 and 1900?" an acceptance test.

### 6.4 Practical solver: lumped capacitance with a substrate node

Full 1-D diffusion is overkill for L2. Use two nodes — surface and substrate — which captures both the fast surface response and the slow diurnal storage:

$$
C_1\frac{dT_1}{dt}=\alpha_{\text{sol}}Q_{\text{sol}}+\varepsilon Q_{\text{LW}\downarrow}-\varepsilon\sigma T_1^4-h(T_1-T_{\text{air}})-\frac{T_1-T_2}{R_{12}}+q_{\text{int}}
$$
$$
C_2\frac{dT_2}{dt}=\frac{T_1-T_2}{R_{12}}-\frac{T_2-T_{\text{deep}}}{R_{2d}}
$$

with $R_{12}=\delta_1/(2k)+\delta_2/(2k)$. Integrate with explicit RK2 at a 1–60 s step; the stability limit is $\Delta t < 2C_1/(h+4\varepsilon\sigma T^3+1/R_{12})$, which for thin painted metal ($C_1\sim 5$ kJ·m⁻²·K⁻¹) lands around 60–200 s. **Decouple this from the render loop entirely** — run it on a fixed thermal tick (say 1 Hz) and interpolate.

**Spin-up matters.** THERM initialises the slab to air temperature some hours before the desired prediction time and integrates forward [R6]. Do the same: run 24–48 h of weather history before $t=0$, or your night scenes will be wrong. This is cheap (a few thousand steps per material class) and it is the difference between a plausible scene and a correct one.

### 6.5 Where Newton's law of cooling fits

$\dot T = -\kappa(T-T_\infty)$ is the linearised, radiation-free, solar-free special case of §6.1. It is exactly right for:

- short-horizon relaxation of an object with no solar input (a body in shade, a machine after shutdown),
- objects whose temperature you are *scripting* rather than predicting.

It is wrong for anything under solar load or exchanging with a cold sky, because it has no $T^4$ term and no $Q_{\text{sol}}$. **Keep it as the "scripted actor" solver and add the full balance as the "environment" solver.** That two-solver split is exactly how DIRSIG organises things (data-driven vs predictive) [R3][R6], and it is what your Unreal Blueprint component should become: one of several pluggable solvers behind a common interface, rather than the only one.

### 6.6 Active heat sources for automotive

These are scripted, not predicted, and they are where most of the useful signal lives:

| Source | Typical ΔT over ambient | Time constant | Notes |
|---|---|---|---|
| Engine bay / grille | +40 … +90 K | 5–20 min warm-up | Strong function of load; visible through grille |
| Exhaust pipe / tip | +80 … +250 K | 2–10 min | Near MWIR peak; saturates LWIR AGC |
| Exhaust plume | +20 … +150 K | seconds | Gas emission, not surface — band-selective (CO₂ 4.3 µm) |
| Tyres (contact patch) | +10 … +35 K | 10–30 min | Rises with speed and cornering; **strong ID cue** |
| Brake discs | +50 … +400 K | 30 s heat, 5 min cool | Best transient cue for braking events |
| Catalytic converter | +100 … +300 K | 3–10 min | Underbody, visible from behind |
| Human body (clothed skin) | +8 … +15 K | slow | Effective $\varepsilon\approx0.98$; face is warmest |
| Recently parked vehicle | +5 … +25 K | 20–60 min decay | Newton cooling is exactly right here |
| Road under a departed vehicle | −3 … −8 K | 5–20 min | Shadow "ghost"; realistic scenes need it |

That last row is worth implementing. Thermal shadows and residual heat traces are a signature phenomenon of the band and they routinely confuse detectors trained only on synthetic data that lacks them.

### 6.7 What the weather does to a surface's energy

**Precipitation carries heat.** Rain reaches a surface at the drop temperature, close to the wet bulb
(§7.8), and the water that lands brings its sensible heat:

$$
Q_{\text{P}} = \dot m_{\text{P}}\,c_w\,\big(T_{\text{rain}} - T_s\big)
$$

At 10 mm h⁻¹ ($\dot m_{\text{P}}$ = 2.8 × 10⁻³ kg m⁻² s⁻¹) on a road 5 K above the drops this is
−58 W m⁻². That is comparable to the whole net radiation under an overcast sky, and convective rain has
been measured cooling soil 6.5 °C at 5 cm depth in twelve minutes [R103]. With the evaporation `PH.1`
already carries, it is why prolonged rain drives every wet surface toward one temperature and thermal
contrast collapses. Snow lands at the snow temperature. On a surface above freezing it also takes $L_f$
per kilogram while it melts, which is `PH.10`'s cap run from the other side.

**A cloud's shadow is local.** The weather series attenuates the direct beam by the cloud fraction,
uniformly (`AT.16`). A cumulus field instead puts each point either in the beam or in a shadow whose
depth is $e^{-\tau_{\text{vis}}}$ along the sun ray through the same field the camera sees. The direct
beam on a facet is therefore

$$
Q_{\text{beam}}(\mathbf x) = \text{DNI}_{\text{clear}}\,\cos\theta_i\,e^{-\tau_{\text{vis}}(\mathbf x\to\odot)}
$$

with the diffuse part unchanged. The one-weather constraint: the field's area mean of $e^{-\tau}$ equals
the series' beam attenuation. The field drifts with the wind, so shadows move, and surfaces answer with
their own time constants. In LWIR a cloud shadow is a cool footprint that **lags** the shadow and is
deepest on low-inertia surfaces — one of the most recognisable things in real aerial thermal footage,
and absent from a model that dims the whole scene at once.

---

<a name="7-atmosphere"></a>
## 7. Atmosphere

### 7.1 Model

$$
\tau_{\text{atm}}(\lambda,d)=\big[\tau_{1\mathrm{m}}(\lambda)\big]^{d}
=\exp\big(-\gamma(\lambda)\,d\big),\qquad \gamma = \gamma_{\text{mol}}+\gamma_{\text{aer}}
$$

and path radiance from the emitting atmosphere itself:

$$
L_{\text{path}}(\lambda,d)=\big(1-\tau_{\text{atm}}(\lambda,d)\big)B(\lambda,T_{\text{air}})
$$

This is the standard Beer-Lambert treatment with a unit-path transmittance describing distance-dependent attenuation, and it is the basis of the three-term observed-radiance model given in §2 [R11]. Path radiance arises from thermal emission of the atmosphere along the line of sight and varies with elevation angle for a ground-based sensor [R13].

### 7.2 Getting $\gamma(\lambda)$ without MODTRAN

MODTRAN is the standard band-model radiative transfer code and the reference for transmittance and background radiance from 0.2 µm outward, retaining LOWTRAN's model atmospheres, aerosol models, clouds and rain attenuation while improving spectral resolution to 2 cm⁻¹ [R16]. If you have access, precompute with it.

If you don't, use this practical procedure:

1. Choose 4–6 representative atmospheres (US Standard, mid-latitude summer/winter, tropical, plus a fog/rain case).
2. For each, obtain band-averaged transmittance vs. range from published curves. MODTRAN runs for EO/IR analysis of ground-level horizontal paths are typically parameterised exactly this way — e.g. a 2 km horizontal path, 1976 US Standard atmosphere, rural aerosols, 16 km visibility [R17].
3. Fit $\gamma_B$ per band per atmosphere. Store a small table indexed by (band, atmosphere, humidity, visibility).

**Which camera the table describes.** A per-band $\gamma_B$ or $\tau$ describes the **band's nominal range**, not any one camera. When the band is split into spectral classes (ADR 0113), the classes' common scale is fitted once, on the nominal top-hat. Every camera in the band then uses those per-class extinctions over its own response. A camera whose range holds less absorbing edge, such as 8–12 µm inside LWIR, therefore sees *further* than the nominal band, and one that reaches into the 6.3 µm water band sees less. Re-fitting the scale per camera would force every camera to the same 200 m transmittance and invert that order (ADR 0177).

**Automotive-relevant magnitudes** (0–300 m horizontal, ground level):

| Condition | LWIR $\tau$ @200 m | MWIR $\tau$ @200 m | SWIR $\tau$ @200 m | Visible $\tau$ @200 m |
|---|---|---|---|---|
| Clear, dry | 0.90–0.96 | 0.92–0.97 | 0.93–0.98 | 0.95–0.98 |
| Humid (30 °C, 80% RH) | 0.72–0.85 | 0.85–0.93 | 0.88–0.95 | 0.93–0.97 |
| Haze (5 km vis) | 0.85–0.92 | 0.82–0.90 | 0.70–0.85 | 0.45–0.65 |
| Light fog (200 m vis) | 0.35–0.60 | 0.25–0.50 | 0.10–0.30 | 0.02–0.10 |
| Dense fog (50 m vis) | 0.02–0.10 | 0.01–0.06 | ~0 | ~0 |

Two things to take from this table:

- Over automotive ranges in clear air, atmospheric attenuation is a **second-order** effect. Do not over-engineer it.
- In fog and haze it is the **dominant** effect, and it is exactly where LWIR's advantage over visible comes from — which is the entire commercial argument for thermal cameras on cars. Getting the *relative* band behaviour right matters far more than getting absolute $\tau$ right to 1%.

Note that LWIR is not universally best: the table shows SWIR beating LWIR in humid clear air (water vapour continuum absorption hits 8–12 µm hard), and LWIR beating everything in fog. A simulator that reproduces this crossover is doing real work.

**Two corrections to the fog rows, and one to how visibility is read** (spec issues S59, S60):

- **The MWIR and SWIR fog columns describe haze-sized droplets, not fog.** Fog droplets are 3–20 µm in
  effective radius; Mie theory on the standard fog size distributions [R84] puts NIR, SWIR and MWIR
  extinction at **1.0–1.3 times the visible**, not below it, and measured transmission through real fog
  agrees — 4 µm gives no meaningful advantage over 1.55 µm, while 10 µm gains more than 20 dB in
  continental fog [R88][R85]. Only LWIR beats the eye in fog, and by how much depends on droplet size
  (§7.7): the LWIR/visible ratio is about 0.35 for a young radiation fog ($r_{\text{eff}} \approx 3$ µm),
  0.72 at 6 µm, and above 1 for advection fog. The table's MWIR/SWIR fog entries are reproduced only by
  droplets near 0.5–1.5 µm — small-droplet artificial fog, which is what fog-chamber benchmarks
  generate. They are kept as a record and are not a target.
- **Visibility is the meteorological optical range, defined at 5 % transmittance** [R91]:
  $\sigma_{\text{vis}} = -\ln 0.05/V = 2.996/V$. Koschmieder's 3.912/V is the 2 % contrast threshold of a
  human observer, a different quantity, and applying it to a reported visibility overstates the
  extinction by 31 %. `isaac-weather-fx` already converts at 5 %, so a weather state read with 3.912
  is two weathers.

### 7.3 Humidity dependence

Water vapour dominates LWIR attenuation. A usable engineering form:

$$
\gamma_{\text{LWIR}} \approx \gamma_0 + \beta \cdot w,\qquad w = \text{precipitable water (g·m}^{-3})
$$

with $w$ from temperature and relative humidity via the Magnus formula:
$$
e_s(T) = 6.112\exp\!\left(\frac{17.67\,T_C}{T_C+243.5}\right)\ \text{hPa},\qquad
w = 216.7\,\frac{\mathrm{RH}\cdot e_s}{T_K}
$$

Fit $\gamma_0,\beta$ per band against your MODTRAN table or published curves.

### 7.4 What you can safely ignore

For ground-vehicle simulation under ~500 m: multiple scattering, adjacency effects, spherical refraction, and spectral fine structure within a band. These matter for airborne and satellite work — they are why MODTRAN's correlated-k treatment exists for LWIR flux calculations [R18] — and not for you. Document that you have dropped them.

### 7.5 Clouds as a participating medium

For aerial targets the cloud is the dominant clutter, and it is also something a target can be
behind or inside. It is therefore not a sky colour: it is a volume of absorbing, emitting air
that every camera ray crosses, to whatever the ray hits (spec issue S33; ADRs 0146, 0162, 0169;
sources and search notes in `docs/clouds-in-the-infrared.md`).

**One field for both bands.** The cloud is `isaac-weather-fx`'s density field $\rho(\mathbf x)\in[0,1]$
scaled by a **visible** extinction $\sigma_{\text{vis}}$ per metre. The visible companion and the
infrared band read the same array from the same origin (the camera, less the wind's drift), so
they cannot disagree about where a cloud is; each band derives its own optical properties from it.

**Visible extinction to band absorption.** Along a camera ray the band's absorption coefficient is

$$
\beta_B(\mathbf x) = r_B\,\sigma_{\text{vis}}\,\rho(\mathbf x)
$$

with $r_B$ a per-band, per-cloud-phase datum, never code:

- **LWIR, liquid water:** $r = 0.5$. The large-particle limit of $Q_{\text{abs,IR}}/Q_{\text{ext,vis}}$
  is exactly one half; measured cloud ratios are 2–3 in extinction terms [R54][R55], and Shaw &
  Nugent use "LWIR optical depth is half the visible" for a calibrated LWIR cloud imager [R56]. The
  same number follows from the mass coefficients: $\tau_{\text{vis}} = 3\,\text{LWP}/(2\rho_w r_e)$
  gives 0.15 m²/g at $r_e = 10$ µm [R57], and Stephens' LWIR flux coefficient 0.130–0.158 m²/g [R58]
  divided by the diffusivity factor 1.66 is 0.08–0.10 m²/g. Scattering (single-scattering albedo
  near 0.5 in the window) is folded into the ratio, the scaled-absorption approximation that keeps a
  non-scattering solver within about 2 % of the flux [R59].
- **A ray takes no diffusivity factor.** $\varepsilon = 1 - e^{-0.79\,\tau_{\text{vis}}}$ [R56] is a
  *flux* emissivity (0.5 × 1.58) and belongs in the hemispherical blends of §5.3, not on a pixel.
- **MWIR:** no citable liquid-water coefficient was found; water's absorption is weaker there, so
  the cloud scatters more than it emits and daytime sunlight scattered by it is first order.
  ESTIMATED until measured.
- **Ice (cirrus):** its own coefficients, polynomials in $1/D_e$ per band [R60]; not used until the
  field carries a phase.

A liquid cloud of 100 g/m² is black in LWIR [R56]: a cumulus is opaque within tens of metres of
its edge, which is why its emission comes from a thin skin facing the camera.

**Temperature.** Surface air lifts dry-adiabatically to the base $z_b$ and saturated above it:

$$
T(z) = T_{\text{air}} - \Gamma_d\,z_b - \Gamma_m\,(z - z_b),\qquad z \ge z_b,\qquad \Gamma_d = g/c_p
$$

$\Gamma_m$ is the moist adiabat, 5–6 K/km in the lower troposphere; the atmosphere preset's
environmental 6.5 K/km stands in for it today, which makes a kilometre-deep cloud's top about 1 K
too cold. An opaque cloud reads close to the temperature of the level it is seen at: its base from
below, its top from above [R56].

**Along the ray.** With the non-scattering emission–absorption (Schwarzschild) equation [R61], a
pixel whose ray hits a surface at range $R$ reads

$$
L = \tau_c(R)\,\big[\tau_{\text{air}}(R)\,L_{\text{hit}} + L_{\text{path,air}}(R)\big]
  + \tau_{\text{air}}(R_c)\int_0^R \beta_B(s)\,L_B\big(T(z(s))\big)\,e^{-\tau_c(s)}\,ds
$$

$$
\tau_c(s) = \int_0^s \beta_B\,ds',\qquad
R_c = \text{the range at which the integral's weight is centred}
$$

discretised per sample, from the camera outward, each sample emitting what it absorbs at its own
height's temperature. A sky pixel is the same integral with $R = \infty$, composed with the clear
column beyond the cloud as ADR 0126 does. A ray is dropped once $\tau_c > 12$ in the **band**.

**Two fidelity tiers, one rule.**

| | Path-traced render | Real-time render |
|---|---|---|
| Visible cloud | 3-D volumes: parallax, shadow, occludes targets | painted on the dome, at infinity |
| Infrared sky pixels | marched per pixel, the integral above with $R=\infty$ | **the same** |
| Infrared pixels on a target | marched to the hit: attenuated, cloud emission in front | clear air to the hit |
| Target behind or inside cloud | occluded in both bands | occluded in neither |

The rule is that **occlusion follows the render path, so the two bands never disagree about
it**. The real-time tier may drop target occlusion for speed. It may not drop the cloud itself:
in both tiers the infrared frame shows the cloud, from the same field, to the same criteria.

**What "realistic" means, in both tiers.**

1. An opaque cloud reads within 1 K of $T(z)$ at the level it is seen, through the air in front.
2. A cloud edge is a fringe of falling emissivity, not a step, and the sampling adds no structure
   of its own: residual lag-one correlation of the march error along a row below 0.6.
3. The two bands cover the same pixels with cloud (they sample one field from one origin).
4. Clear sky, broken cloud and overcast are compared with calibrated full-sky LWIR imagery (the
   ARM Infrared Cloud Imager, roadmap `XD.6`): the cloud-minus-clear radiance and the width of
   cloud edges fall inside the spread of the real sky at matching air temperature and humidity.

**The shape of a cloud is a requirement on the field.** Both bands read one field, so a cloud that
looks wrong looks wrong in both, and no radiometry in either band can repair it. Real cumulus give
four measurable targets:

| property | real cumulus | sources |
|---|---|---|
| projected outline, area–perimeter dimension $D$ | 1.3–1.4 (1.35 for 1–10⁶ km²; 1.28 for trade cumulus at 15 m resolution) | [R62][R63] |
| size distribution $n(l)\propto l^{-b}$, $l=\sqrt{A}$ | $b$ = 1.7–2.0 from ~50 m to a break at 0.5–1 km | [R64][R65][R66] |
| edge | liquid water steps over ~0.3 m at the boundary; water, droplet number and size decline over the outer ~10 % of the cloud | [R67][R68] |
| depth to chord, shallow cumulus | 0.4–1 | [R69] |

The edge row decides what a camera sees. At a cumulus visible extinction of 0.05–0.12 m⁻¹ [R70] the
optical skin — the depth at which a ray reaches unit optical depth — is 8–20 m, and the LWIR
absorption skin of a ray is about twice that, 15–40 m ($r = 0.5$; the flux coefficient 100–160 m² kg⁻¹
of liquid water [R71] is the diffusivity factor times the ray's). A 60 m voxel holds
the whole visible surface of a cloud inside one cell: read trilinearly it is a 60 m ramp, "soft and
blobby"; read nearest it is a staircase. The real field is neither — it is nearly binary at metre
scale.

**Envelope plus detail.** The production answer, adopted here, keeps the stored field coarse and
makes the edge at sample time. A low-resolution *envelope* says where cloud may be and how far a
point is from its edge, and a tiling high-frequency noise *erodes* the envelope's edge when the field
is sampled — never multiplying the core, which would hollow it [R72][R73]. Detail then costs
arithmetic rather than memory: Guerrilla's voxel clouds reach 0.5 m effective precision from 8 m
voxels [R74]. Two consequences are physics, not rendering:

1. **The pixel integrates radiance (`WX.4`, ADR 0181).** A pixel at range $R$ with instantaneous
   field of view $\theta_p$ integrates the radiance over a footprint $R\theta_p$. Near the horizon that
   footprint is also stretched across the cloud layer by $1/\sin e$. Sampled once, an edge crossing
   the pixel aliases into noise: the static a broken-cumulus field shows toward the horizon.

   The integral is taken in **radiance**, not in density. Blurring the density to the footprint was
   measured to make the static worse: 26.6 % rms against a supersampled reference became 46.5 %, with
   a −17 % bias. Opacity is not linear in density. A half-covered pixel is half cloud and half sky,
   not a half-dense cloud over all of it, and along a kilometres-long grazing ray a half-dense cloud
   is still opaque.

   So each band marches one ray per pixel and marches again, through several rays, only the pixels
   an edge crosses:
   - **infrared:** every sample of the supersampled grid below 15° elevation, and one ray per 2 × 2
     block above it;
   - **visible dome:** 2 × 2 rays per texel, and 4 × 4 below 15°.

   Each band integrates over **its own** pixel, which keeps the one-field rule: the field is one
   function, and each camera integrates it as it physically does.
2. **The step is set by the skin, not the grid.** A ray must resolve the 8–20 m skin where it enters
   a cloud, or the emission and the in-scattered light both come out wrong at the edge, where they
   matter most. Inside the envelope the march steps at most half the finest surviving detail;
   outside it the march skips.

**Two sources, one contract.** A cloud field may be *procedural* — an envelope from fractal and
cellular noise, as `isaac-weather-fx` builds it — or a *volume asset*: a cloud from a large-eddy
simulation, which carries liquid water and temperature in physical units [R75][R76], or a sculpted
cloud, which carries shape only [R77]. Both answer the same questions: visible extinction
$\sigma_{\text{vis}}(\mathbf x)$ in m⁻¹, the phase (liquid, ice) and, where the source has them, the
liquid water content and temperature. Both are eroded by the same detail. The project keeps both until
one is accepted against real imagery, and neither may reach a band the other cannot.

**Band ratios from microphysics.** Where the field carries an effective radius $r_e$, the band ratio
is not a constant. With $\sigma_{\text{vis}} = 3\,\text{LWC}/(2\rho_w r_e)$ [R57] and a band's mass
absorption coefficient $\kappa_B$ for a pencil beam,

$$
r^{a}_{B} = \frac{\kappa_B\,\text{LWC}}{\sigma_{\text{vis}}} = \tfrac{2}{3}\,\kappa_B\,\rho_w\,r_e
$$

At $r_e$ = 10 µm and the ray coefficient 0.075 m² g⁻¹ this is 0.50, the constant used since ADR 0126.
It is the large-droplet limit: a droplet much thicker than water's 16 µm absorption depth at 10 µm
absorbs what it intercepts, $\kappa_B \propto 1/r_e$, and the ratio sits at one half. Smaller droplets
are not opaque at 10 µm, $\kappa_B$ tends to the volume absorption of water, and the ratio falls with
$r_e$. A continental cloud of small droplets is therefore less absorbing in LWIR, for the same visible
optical depth, than a maritime one. $\kappa_B(r_e)$ is band data computed once by Mie
(`irsim.atmosphere.mie`). The constant stays the default; a field that carries $r_e$ uses the
relation.

**The visible companion's cloud.** The visible frame is not radiometric output, but it is the
reference the infrared frame is read against, so its cloud is held to physical criteria:

1. **A cloud never darkens the sky behind it by scattering back less than it removes.** At a thin
   edge the observed radiance is
   $L_{\text{sky}}(1-\tau) + \tau\varpi\,[\,p(\Theta)E_\odot T_\odot + \langle L_{\text{sky}}\rangle_p\,]$.
   The last term is the sky light — including the sky directly behind the cloud — that a strongly
   forward phase function sends on to the camera. Dropping it, or replacing it with an ambient that
   vanishes at low density (a two-stream *reflectance* does), puts a rim darker than the sky around
   every cloud. The test is a **white furnace**: a non-absorbing cloud ($\varpi = 1$) under a uniform
   sky with no sun must vanish [R78].
2. **Energy-conserving steps.** Each step adds $\varpi L_{\text{in}}(1-T_{\text{step}})$ — the closed
   form of the in-scattering integral over a step of constant extinction — so no step size can make a
   step brighter than its opacity allows [R79]. The same closed form is the infrared emission per
   step, $(1-\varpi)B(T)(1-T_{\text{step}})$.
3. **Premultiplied composite.** The march yields an associated colour $C$ and an opacity $\alpha$, and
   the cloud is composed over the sky as $C + (1-\alpha)L_{\text{sky}}$. Any resampling — a coarse dome
   upsampled to the screen — is done on $C$ and $\alpha$ before that, never on a colour already divided
   by $\alpha$ [R80].
4. **Enough scattering orders.** Thick cumulus is white because light scatters in it of order a
   hundred times [R81]. A path tracer capped at a few volume bounces renders it grey: RTX's volume
   scattering cap `ptvol/maxBounces` defaults to 2 [R83]. A march approximates the missing orders —
   with Wrenninge's octaves, contribution ≤ attenuation for energy conservation [R82], or, as the
   dome does since `WX.3`, with the two-stream field along each sun chord (below).
5. **Aerial perspective.** A cloud at range $R$ is seen through the air:
   $T_{\text{air}}(R)\,C + \alpha\,L_{\text{air}}(0,R)$, its own light attenuated by the air in front of
   it plus the light that air scatters toward the camera. That is what makes a distant cloud hazy and a
   near one crisp. Koschmieder's law, contrast $= T_{\text{air}}(R)$, is the case of uniform air only.
   In a real atmosphere the aerosol sits low, so a ray that climbs out of it keeps more contrast than
   its transmittance says; the measurements below show by how much.

**How the dome meets them (`WX.3`, ADR 0180).** Each step of the visible march in-scatters, with
criterion 2's weight $\varpi\,\mathcal T\,(1-e^{-\Delta\tau})$, three lights:

- **The sky and the ground**, $h\,\bar L_\uparrow + (1-h)\,\bar L_\downarrow$: $h$ is the step's height
  fraction in the slab, and $\bar L_\uparrow$, $\bar L_\downarrow$ are the clear sky's own solid-angle
  means over the upper and the lower hemisphere. Summed over a ray the weights are $\varpi(1-\mathcal T)$,
  so under a uniform sky the cloud returns exactly what it removes (criterion 1's furnace).
- **The sun, scattered once**: $\dfrac{p(\Theta)}{4\mu_\odot}\,e^{-\tau_\odot}$ in units of $E_h/\pi$, the
  radiance of a white ground in the same sun, with $p$ Henyey–Greenstein at $g = 0.85$ normalised so
  that isotropic is 1. It is exact as $\tau \to 0$ and it is the whole of a silver lining.
- **The sun, scattered many times**: the δ-Eddington two-stream field of a conservative layer, taken
  along the sunlight's own chord through the point. With $\tau_\odot$ the optical depth to the cloud's
  edge toward the sun and $\tau_a$ away from it, $\tau_c = \tau_\odot + \tau_a$, $\tau^* = (1-g)\tau_c$ and
  $f = \tau_\odot/\tau_c$:
  $$
  I_B = R\,(1-f),\qquad I_F = T\,f,\qquad R = \frac{\tau^*}{2+\tau^*},\qquad T = \frac{2}{2+\tau^*} - e^{-\tau_c}.
  $$
  In chord form these are the plane-parallel $R = (1-g)\tau_v/(2\mu_\odot + (1-g)\tau_v)$ and its diffuse
  transmission exactly, since a plane-parallel chord is $\tau_v/\mu_\odot$. The ray reads the stream
  leaving the face it entered: the backward one through a lit face, where $f$ rises along the ray; the
  forward one through a dark face; both along a grazing ray. It in-scatters the source that stream
  implies, $S = I - dI/d\tau$ (from $dI/d\tau = I - S$, with $\tau$ increasing away from the camera).
  That source returns $R$ from above and $T$ from below for a uniform layer of any depth, and it
  returns only its own optical depth's share along a ray that grazes an edge.

The light maps store optical depth, not transmittance: interpolated transmittance put the depth above
a cloud base at 0.6 of a fine integration. On a 35 % cumulus field at a 58° sun the march meets the
criteria as follows:

- **White furnace:** within $10^{-4}$.
- **Thin edges** ($\tau < 0.3$, above 20°): at least 1.01 of the clear sky behind them.
- **Thick bases** ($\tau > 30$): 0.82–0.89 of a sunlit flank's luminance.
- **Uniform deck seen from above:** 1.005 of its two-stream albedo.

The path-traced volumes get `ptvol/maxBounces` = 32 (`WX.2`).

The thin-edge criterion is applied outside the 15° round the sun. Inside it, a wisp shadowed by its own
cloud is compared with the clear sky's aureole, and the same cloud shadows that aureole too. The dome
shadows only the air in front of a cloud (below), not the clear sky behind it.

**Aerial perspective in the dome (`WX.5`, ADR 0182).** The march reports an emission range $\bar R$:
the mean distance along the ray, weighted by what each step adds. The composite becomes
$T_{\text{air}}(\bar R)\,C + \alpha\,L_{\text{air}}(0,\bar R) + (1-\alpha)\,L_{\text{sky}}$. It is still
premultiplied, so criterion 3 holds.

$T_{\text{air}}$ and $L_{\text{air}}$ are the sky's own scattering integral, kept at 16 range slices out
to 64 km on the sky view's angles. This is Hillaire's aerial-perspective volume [R116], so at the far
end of a ray they are the sky view itself.

$L_{\text{air}}$ is kept in two parts: the share of the direct beam scattered once, and the share
scattered many times. The direct share is scaled by the sun's transmittance through the cloud field,
read from the field's light map at four points along the path below the cloud base. The air in a
cloud's shadow is therefore dark. Without that, a cloud near the sun was veiled by an aureole of air
that is in fact in its shadow, by up to 28 kcd/m², and read brighter than the clear sky.

Measured (upstream `tests/test_cloud_light.py`):

- **The far limit:** at 64 km the air light is the sky view to within 1.5 % from 30° up.
- **Contrast against Koschmieder:** a black object near the horizon keeps 1.02 times $T_{\text{air}}$ at
  5 km. As the ray climbs out of the aerosol it keeps 1.07 times at 15 km and 1.16 times at 30 km.
- **Distance:** a cloud 20 km off moves to within 0.6 of its old distance from the horizon's colour. A
  cloud overhead changes by under 15 %.

RTX volumes scatter but cannot emit [R83]. The infrared cloud is therefore never the renderer's: it is
always this section's march over the shared field.

**Not modelled, and flagged.** Multiple scattering in the thermal bands beyond the scaled ratio;
sunlight scattered by cloud toward the camera on a target pixel (sky pixels carry it, ADR 0153); ice
microphysics beyond a per-genus phase. In the visible dome, the light a finite cloud loses through its
sides (the march is one-dimensional along each sun chord, so the base of a small cumulus is as bright as
two-stream says a layer of its depth is); a cloud's shadow on the air beyond it and on the clear sky
around it, and so light shafts; the air above a cloud base, taken as sunlit. Cloud shadows on surfaces are §6.7; precipitation is §7.8.

### 7.6 One march for every medium

§7.1–7.3 treat the clear air as a horizontally uniform column with closed-form slant integrals
(`AT.1`). Everything else the weather puts between the camera and the hit has **structure**: a cloud,
a fog bank with a top, a haze layer capped by the boundary layer, a rain shaft, falling snow. The only
way to honour structure is to integrate along the ray, and §7.5 already does that for one medium. The
rule is that there is **one** such integral and every structured medium enters it:

$$
L(0) = \mathcal T(0,R)\,\big[\tau_{\text{air}}(R)L_{\text{hit}} + L_{\text{path,air}}(R)\big]
+ \tau_{\text{air}}(R_m)\int_0^R \sum_m \beta^{a}_{B,m}(s)\,L_B\big(T_m(s)\big)\,\mathcal T(0,s)\,ds
$$

$$
\mathcal T(0,s) = \exp\Big(-\int_0^s \sum_m \beta^{e}_{B,m}(s')\,ds'\Big),\qquad
\beta^{e}_{B,m} = r^{e}_{B,m}\,\sigma_{\text{vis},m},\qquad
\beta^{a}_{B,m} = r^{a}_{B,m}\,\sigma_{\text{vis},m}
$$

Each medium $m$ supplies three things from the one weather state:

- a **visible extinction field** $\sigma_{\text{vis},m}(\mathbf x)$ in m⁻¹ — the quantity the visible
  companion renders, so the two bands cannot disagree about where the medium is;
- a **class** (haze, fog droplet, cloud liquid, cloud ice, rain, snow);
- a **temperature** $T_m(\mathbf x)$.

The band supplies, per class, two ratios as data: $r^{e}$, the extinction the pixel loses, and
$r^{a} \le r^{e}$, the part of it that is absorption and therefore emits. Adding a band is a new row of
ratios; adding a medium is a new class row and a field. Neither touches the integrator (*bands are
data*).

The gap between $r^e$ and $r^a$ is scattering. Light scattered *out* of the pixel is lost; light
scattered *into* it from elsewhere is not carried, except where a class says how (rain's diffraction
halo, §7.8). This is the non-scattering (Schwarzschild) approximation with scaled absorption that §7.5
already makes for cloud [R59]. It is right where absorption dominates, or where the in-scattered field
is close to the emitted one (thermal bands inside a medium near air temperature). It is **wrong in the
reflective bands in daylight**, where sunlight scattered into the pixel by fog, rain or cloud is first
order. That term is carried separately, as for cloud bases (ADR 0153), and flagged where it is not.

**What stays closed-form.** The clear air keeps §7.1–7.3 and the layered column; integrating a uniform
medium numerically buys nothing and costs a march. The structured media are composed with it as §7.5
does for cloud: the hit and the air in front of it are attenuated by the media's transmittance, and the
media's emission is attenuated by the air between the camera and the range $R_m$ at which that emission
is centred.

**Sampling.** Inside a medium a step is at most half its finest surviving structure; outside every
medium's bounds a ray is not sampled. A pixel whose footprint a medium's edge crosses is integrated
by marching several rays through it, not by blurring the medium (§7.5). A ray is dropped once
$\mathcal T < e^{-12}$ in the **band**.

### 7.7 Fog, mist and haze have structure

§7.1–7.3 make haze and fog a property of the column: a visibility, a ratio per band, an exponential
profile. Four things that decide what a camera sees in fog are missing from that.

**Fog is a layer with a top.**

- Radiation fog is usually shallower than 200 m; one recent campaign measured tops of 83–115 m [R89].
- It becomes optically thick once its liquid water path passes about 30 g m⁻² [R90].
- Its liquid water and extinction rise with height, peaking near 80 % of its depth [R89][R85].
- Above the top the air is clear.

A fog is therefore a §7.6 medium, not a column property:

$$
\sigma_{\text{vis}}(\mathbf x) = \sigma_0\,f(z/z_{\text{top}})\,\big(1 + a\,n(x,y)\big)
$$

with a profile $f$ that rises toward the top and a horizontal modulation $n$. The difference is the whole
picture from a drone: above the layer the ground is seen through $z_{\text{top}}/\sin\theta$ of fog, not
through the camera's range of it, and the fog top is a surface with an edge. Horizontal patchiness
follows the ground — soil moisture and terrain set fog on 100 m–1 km scales [R93] — and is authored,
not inferred.

**Droplet size sets the band ratio, and fog type sets the droplet size.** Infrared extinction in fog is
nearly proportional to liquid water content: at 11 µm, about 133 km⁻¹ per g m⁻³, within a factor of two
over 341 measured droplet spectra [R86][R87]. Visible extinction is $1500\,W/r_{\text{eff}}$ (km⁻¹,
$W$ in g m⁻³, $r_{\text{eff}}$ in µm). The LWIR/visible ratio is therefore about
$0.089\,r_{\text{eff}}$[µm] until it saturates near 1.1 above ~10 µm. By Mie over a modified-gamma
distribution, with this project's own code and water constants:

| $r_{\text{eff}}$ (µm) | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|
| LWIR / visible | 0.085 | 0.21 | 0.35 | 0.48 | 0.72 | 0.89 | 1.00 | 1.10 |
| MWIR / visible | 0.28 | 0.87 | 1.24 | 1.36 | 1.28 | 1.18 | 1.13 | 1.09 |
| SWIR / visible | 1.11 | 1.23 | 1.10 | 1.07 | 1.05 | 1.04 | 1.04 | 1.03 |

Typical sizes by fog type:

- young radiation fog: 2–4 µm;
- mature radiation fog: 8–10 µm near the surface [R89];
- advection fog: 16–20 µm [R84].

The ratio is band data indexed by $r_{\text{eff}}$, and a fog medium carries $r_{\text{eff}}$. "LWIR sees
through fog" is true of young radiation fog and largely false of advection fog — which is what thermal
detection ranges show in practice (maritime aerosols, the largest, give the shortest [R97]). A path's transmittance is
the band integral of spectral $e^{-\beta L}$, not $e^{-\bar\beta L}$; the band-class machinery of §7.1
carries that.

**Fog temperature.** Fog air is saturated and its droplets are at air temperature to first order. Fog is
not isothermal, though:

- A thin fog sits inside a surface-based inversion, colder than the air above it — 12 °C at the ground
  and 14–16 °C above 40–60 m in one measured case [R89].
- A thick, mixed fog follows the saturated adiabat, 5.3–6.5 K km⁻¹, so its top is about 1 K colder than
  its base, under warmer air.

A drone looking down at a fog layer therefore sees the fog top — the coldest point in the column — and
not the surface air temperature. $T_{\text{fog}}(z)$ is part of the medium.

**Scattering in MWIR.** Fog's single-scattering albedo is 0.33–0.55 in LWIR but 0.68–0.86 in MWIR (Mie
on [R84]). In MWIR an optically thick fog is a partly reflecting layer, not a blackbody, and
$(1-\tau)B(T)$ overstates its emission when it is seen from above or looked up through at a cold sky.
§7.6 carries $r^a < r^e$ for this. The in-scattered sky and ground are not carried (flagged).

**Mist and haze.** The WMO thresholds [R91][R92]:

| | visibility | condition |
|---|---|---|
| fog | below 1 km | — |
| mist | 1–5 km | RH > 95 % |
| haze | ≤ 5 km | dry particles |

Haze keeps the layered model's exponential profile, with the standard models giving its scale. At 23 km
visibility LOWTRAN 7's boundary layer falls 0.158 → 0.099 → 0.062 km⁻¹ at 0, 1 and 2 km, a 2.1 km scale
height; at visibility ≤ 10 km it is uniform through the lowest kilometre [R94]. A haze layer capped at
the boundary-layer top is a §7.6 medium where a ray leaves it within the scene; otherwise the closed form
suffices.

### 7.8 Rain and snow along the line of sight

**Rain extinction is flat across the spectrum.** Raindrops, 0.1–6 mm across, are hundreds of wavelengths
wide even at 10 µm, so each intercepts twice its geometric cross-section ($Q_{\text{ext}}\to 2$) in every
band [R95]. The distribution is Marshall–Palmer [R96]:

$$
N(D) = N_0 e^{-\Lambda D},\qquad N_0 = 8000\ \text{m}^{-3}\,\text{mm}^{-1},\qquad
\Lambda = 4.1\,R^{-0.21}\ \text{mm}^{-1}
$$

The extinction over it is

$$
\sigma_{\text{rain}} = \frac{\pi}{2}\int N(D)\,D^2\,dD = \frac{\pi N_0}{\Lambda^3}
= 0.365\,R^{0.63}\ \text{km}^{-1}\qquad (R\ \text{in mm h}^{-1})
$$

Mie over the distribution holds this within 2 % from 0.55 to 12 µm. It is the law LOWTRAN 7 implements
[R94]. At 25 mm h⁻¹ it is 2.8 km⁻¹, so $\tau$(200 m) = 0.57 **in every band**: **LWIR's fog advantage
does not exist in rain** [R95][R97]. The empirical optical-link laws, fitted near 0.8–1.55 µm [R98][R99],
agree in magnitude; no 8–12 µm fit was found.

**What is absorbed differs, and the pixel decides what is lost.** Half the extinction is diffraction
into a narrow forward lobe. The other half is light the drop intercepts:

- In the thermal bands the drop absorbs it. Water's 1/e depth is 16 µm at 10 µm and 69 µm at 4 µm
  [R100], so a millimetre drop is opaque.
- In the visible the drop refracts it away.

Absorption is therefore 48 % of the extinction in LWIR, 46 % in MWIR, 22–36 % at 1.55 µm, about 1 % at
1 µm and nil in the visible [R95].

The diffracted half is not lost to the image. It lands in a lobe of half-width about $1.22\lambda/D$, and
whether it stays in the pixel depends on the pixel. The share of a drop's lobe inside a pixel of IFOV
$\theta_p$ is the Airy encircled energy

$$
f_B(D) = 1 - J_0^2(x) - J_1^2(x),\qquad x = \frac{\pi D\theta_p}{2\lambda}
$$

averaged over the distribution with weight $D^2$. For $\theta_p$ = 1 mrad at 10 mm h⁻¹ it is 70 % at
0.55 µm, 48 % at 1 µm, 30 % at 1.55 µm, 7 % at 4 µm and 1 % at 10 µm. So rain's ratios are

$$
r^{e}_{B} = 1 - \tfrac12\bar f_B,\qquad r^{a}_{B} = \text{the absorbed share}
$$

The light diffracted out of a pixel lands in its neighbours: it is a **blur, not a loss**. With $H_B$ the
diffraction kernel, scaled by where along the path the drops sit,

$$
L_{\text{obs}} = e^{-r^e\sigma s}L + \big(e^{-\sigma s/2} - e^{-r^e\sigma s}\big)\,(H_B\otimes L)
+ L_{\text{path}}
$$

This is the aerosol-MTF form [R101]: fine detail is attenuated by the full extinction, a large uniform
area by the intercepted half only. In LWIR the halo is 5–25 mrad wide, so rain softens a target's edges
more than it dims a large warm area. The form is single-scattering, adequate to optical depths of about
two.

**Rain emits at the wet bulb.** A falling drop reaches the psychrometric wet-bulb temperature within an
e-folding fall of 4 m (1 mm drop) to 57 m (4 mm) [R102]. Below the cloud base a rain shaft is therefore
at $T_{\text{wb}}(z)$, which `irsim.thermal.latent.wet_bulb_temperature_k` already solves. Convective rain
from cold tops arrives colder still: 3.8 °C below the wet bulb on average in one survey [R103]; that
offset is ESTIMATED and authored per event. The path emits through $r^a$ only. LOWTRAN 7 uses the full
extinction as path emissivity when multiple scattering is off [R94], which roughly doubles rain's LWIR
emission: at 25 mm h⁻¹ over 100 m, 0.246 against 0.127 from absorption.

**Snow** extinguishes nearly flat too, and far harder per millimetre of water. The optical-link law
[R98][R99] is

$$
\sigma_{\text{snow}} = a\,S^{b}\ \text{dB km}^{-1}
$$

$$
\text{dry: } a = 5.42\times10^{-5}\lambda_{\text{nm}} + 5.4959,\ b = 1.38;\qquad
\text{wet: } a = 1.023\times10^{-4}\lambda_{\text{nm}} + 3.7855,\ b = 0.72
$$

with $S$ the water-equivalent rate in mm h⁻¹. Its $\lambda$ term was fitted at 0.8–1.55 µm. Extrapolated
to 10 µm it gives LWIR/visible 1.09 (dry) to 1.25 (wet); measurements give IR/visible 1.05–1.43 [R104].
Dry snow at 2 mm h⁻¹ is 3.6 km⁻¹, against rain's 0.57 km⁻¹ at the same rate.

Visibility tracks snowfall with a scatter of 3–10× at a given rate [R105]. Where the weather gives a
visibility in snow, $\sigma_{\text{vis}}$ comes from it and the law is the fallback. Snowflakes are at
the ice bulb and never above 0 °C, and ice is above 0.98 emissive in LWIR [R106]. The absorbed share is
one half, as for rain (ESTIMATED: a flake is thick against ice's LWIR absorption depth).

**A weather states precipitation and visibility once.** A reported visibility in rain already includes
the rain. The aerosol or fog part of $\sigma_{\text{vis}}$ is the remainder after the precipitation's
own visible extinction, at the effective $Q \approx 1.15$–1.2 a visual observation implies [R98]. Folding
rain into a "fog" visibility gives LWIR the fog advantage it does not have in rain (spec issue S61).

**Single drops and flakes.** Close to the camera a drop is resolved: a 2 mm drop subtends 1 mrad at 2 m.

*In the visible* a drop is a bright refracting lens with a 165° field of view [R107], integrated over the
exposure into a streak.

*A thermal camera* sees an opaque emitter at the wet bulb, and a microbolometer integrates it through its
thermal time constant $\tau_{\text{th}}$ (about 10 ms). A drop of size $D$ falling at $v$ covers a pixel
for $D/v$, so its peak contrast is

$$
c = 1 - \exp\!\big(-D/(v\,\tau_{\text{th}})\big)
$$

This is independent of range, since angular size and angular speed both scale as $1/r$ [R108]:

- A 2 mm raindrop at 6.5 m s⁻¹ keeps **3 %**, smeared into a 40–90 mm streak. Falling rain is close to
  invisible to an uncooled LWIR camera as individual drops, and present only as the medium.
- A 5 mm snowflake at 1 m s⁻¹ keeps **39 %**: snow is seen as flakes.
- A cooled photon detector integrates for $t_{\text{int}}$ instead. Its drop is a streak of length
  $v\,t_{\text{int}}$ with contrast $D/(v\,t_{\text{int}})$.

Fall speeds for these figures:

- rain: $v = 9.65 - 10.3\,e^{-0.6D}$ m s⁻¹, $D$ in mm, above ~0.5 mm [R109];
- snow: per habit, e.g. $0.8\,D^{0.16}$ m s⁻¹ for unrimed aggregates of dendrites [R110].

Within the range where a particle subtends at least a tenth of a pixel, particles are rendered
individually, **at the positions the visible companion draws**, so the two bands show the same flakes.
The statistical medium begins beyond that range, so nothing is counted twice.

---

<a name="8-optics"></a>
## 8. Optics

### 8.1 Irradiance at the focal plane

For an extended Lambertian source of in-band radiance $L_B$:

$$
E_{\text{FPA}} = \frac{\pi L_B \tau_{\text{opt}}}{4F^2+1}\cos^4\theta
$$

and the power on one pixel is $\Phi = E_{\text{FPA}}A_d$. In photon terms, replace $L_B$ with $L_{q,B}$ and $\Phi$ becomes a photon rate.

The working f-number $F = f/D$ dominates sensitivity: NETD scales as $F^2$. This is why uncooled LWIR lenses are $F/1.0$–$F/1.4$ and expensive (germanium), and why a "faster lens" is the cheapest sensitivity upgrade available. Make $F$ a first-class config parameter so trade studies are one line of YAML.

### 8.2 Self-emission — the term that only exists in the IR

Your own optics glow. In LWIR at room temperature this is **not** negligible, and in a cooled MWIR system the cold shield exists precisely to suppress it. A worked example from a calibration patent decomposes in-band radiance at the detector into: lens emission, window emission times lens transmission, mirror emission times window and lens transmission, window emission times mirror reflectance times transmissions, and so on — each surface contributing $\varepsilon_i B(\lambda,T_i)$ attenuated by every element downstream [R19].

Generalised, for a stack of $N$ elements between scene and detector:

$$
L_{\text{self}} = \sum_{i=1}^{N}\varepsilon_i B(\lambda,T_i)\prod_{j>i}\tau_j
$$

Practical simplification for a single-lens uncooled core:

$$
\Phi_{\text{self}} \approx A_d\,\Omega_{\text{eff}}\,\big(1-\tau_{\text{opt}}\big)\,L_B\!\left(T_{\text{housing}}\right)
$$

with $\Omega_{\text{eff}} = \pi/(4F^2+1)$. Because $T_{\text{housing}}$ drifts, this term is the physical origin of **shutterless drift** and the reason cameras need periodic flat-field correction (§11.2). Model it and you get NUC behaviour for free; skip it and you have to fake NUC drift with an arbitrary random walk.

**Where the housing radiation lands on the array — the field dependence.** The single-lens form above
is one number for the whole array, and a featureless scene shows that to be wrong. A pixel at field
angle $\theta_{ij}$ sees the aperture as the projected solid angle $\Omega_{\text{eff}}\,\mathrm{RI}_{ij}$,
where $\mathrm{RI}_{ij}$ is the relative illumination of §8.1 ($\cos^4\theta_{ij}$ times the measured
mechanical vignetting). The rest of its Lambertian hemisphere, projected solid angle
$\pi - \Omega_{\text{eff}}\mathrm{RI}_{ij}$, is the inside of the camera. So for a uniform scene of
in-band radiance $L_{\text{scene}}$ the power on pixel $(i,j)$ is

$$
\Phi_{ij} = A_d\,\Omega_{\text{eff}}\,\mathrm{RI}_{ij}\Big[\tau_{\text{opt}}L_{\text{scene}} + (1-\tau_{\text{opt}})L_B(T_{\text{lens}})\Big]
          + A_d\big(\pi - \Omega_{\text{eff}}\mathrm{RI}_{ij}\big)L_B(T_{\text{housing}})
$$

and with $T_{\text{lens}} = T_{\text{housing}}$, which is the single-lens simplification,

$$
\Phi_{ij} = A_d\,\pi\,L_B(T_{\text{housing}})
          + A_d\,\Omega_{\text{eff}}\,\mathrm{RI}_{ij}\,\tau_{\text{opt}}\Big[L_{\text{scene}} - L_B(T_{\text{housing}})\Big]
$$

An uncooled detector measures the scene **relative to its own housing**, and the relative illumination
multiplies that *difference*, not the scene radiance [R48, R49]. Three consequences:

- A uniform scene warmer than the housing is brightest on axis; one colder — a clear sky sits tens of
  kelvin below any housing — is darkest on axis. The vignetting of §8.1 must therefore be applied to
  $L_{\text{scene}} - L_B(T_{\text{housing}})$, never to $L_{\text{scene}}$ alone: applying $\cos^4$ to the
  scene and adding a uniform self-emission term gets the sign of the shading wrong for every scene
  colder than the camera.
- The pedestal $A_d\pi L_B(T_{\text{housing}})$ is the term the flat-field correction removes, and its
  drift between shutter events is what the §11.2 residual is made of. It is not white noise: it is
  smooth and radial, because the out-of-cone weight $\pi - \Omega_{\text{eff}}\mathrm{RI}_{ij}$ grows
  toward the corners.
- In a cooled system the cold shield replaces the out-of-cone housing view with the cold-shield term of
  §9.1 (`cold_shield_efficiency`), and only the in-cone bracket survives; the field dependence then
  lives in the residual cold-shield leakage instead.

A real reference frame of nothing but clear sky, from an uncooled 640×512 core, shows exactly this: a
smooth, radially symmetric bowl centred on the optical axis and spanning the whole frame, with the
column striping of §10.2 running straight through it, stretched to full contrast by the AGC because
the scene itself has no contrast to offer. Any simulator whose uniform scene comes out uniform is
missing this term.

Related effect worth modelling if you care about realism: **narcissus** — the detector seeing its own cold reflection in the optics, producing a soft dark blob near image centre that shifts with focus and temperature. A radially symmetric multiplicative field with a slowly drifting amplitude reproduces it convincingly.

### 8.3 MTF cascade

System MTF is the product of independent contributions:

$$
\mathrm{MTF}_{\text{sys}}(\xi)=\mathrm{MTF}_{\text{diff}}\cdot\mathrm{MTF}_{\text{aberr}}\cdot\mathrm{MTF}_{\text{det}}\cdot\mathrm{MTF}_{\text{motion}}\cdot\mathrm{MTF}_{\text{defocus}}\cdot\mathrm{MTF}_{\text{elec}}
$$

**Diffraction** (circular aperture, incoherent):

$$
\mathrm{MTF}_{\text{diff}}(\xi)=\frac{2}{\pi}\left[\arccos\!\left(\frac{\xi}{\xi_c}\right)-\frac{\xi}{\xi_c}\sqrt{1-\left(\frac{\xi}{\xi_c}\right)^2}\right],
\qquad \xi_c=\frac{1}{\lambda F}
$$

**Detector footprint** (square pixel of width $w$):

$$
\mathrm{MTF}_{\text{det}}(\xi)=\left|\operatorname{sinc}(\pi w \xi)\right| = \left|\frac{\sin(\pi w\xi)}{\pi w\xi}\right|
$$

with first zero at $\xi = 1/w$. This term is always present in any imaging system with detectors, whether or not it is the limiting one [R20].

**Motion smear** over integration time $t_{\text{int}}$ with image-plane velocity $v$: $\mathrm{MTF}_{\text{motion}}(\xi)=|\operatorname{sinc}(\pi v t_{\text{int}}\xi)|$.

**Aberration / defocus:** in practice fit a Gaussian, $\exp(-2\pi^2\sigma^2\xi^2)$, from a measured slant-edge, rather than deriving it.

**Nyquist and aliasing.** The pixel pitch $p$ sets $\xi_N = 1/(2p)$; content above it aliases [R21]. Worked case for a 12 µm-pitch LWIR core at $F/1.0$, $\lambda=10$ µm:

- $\xi_c = 1/(\lambda F) = 100$ cyc/mm
- $\xi_N = 1/(2\times0.012) = 41.7$ cyc/mm
- $\mathrm{MTF}_{\text{diff}}(\xi_N) \approx 0.49$, $\mathrm{MTF}_{\text{det}}(\xi_N) \approx 0.64$

So the system is **detector-limited but not diffraction-free**, and it aliases. If you render at native resolution and apply a blur, you get the blur but not the aliasing, and aliasing is precisely what corrupts small-target detection at range — the case you care about for automotive. **Render at 3–4× and downsample with a box filter matching the pixel footprint.** That single choice buys you correct aliasing, correct detector MTF, and correct sub-pixel target behaviour with no extra model.

Cascade models for sampled IR systems exist precisely to handle this — they yield both the aliasing band and the averaged modulation response for a general sampling subsystem [R22].

### 8.4 Distortion

Standard camera-model territory: Brown-Conrady for moderate FOV, Kannala-Brandt or an f-theta polynomial for wide IR lenses. Isaac Sim's camera stack already supports OpenCV pinhole/fisheye, f-theta and Kannala-Brandt models [R23], so use the engine's distortion rather than reimplementing it.

---

<a name="9-detector"></a>
## 9. Detector

Two physically different device classes, two model paths, one interface.

### 9.1 Photon detectors — cooled MWIR/SWIR, and NIR

Signal in photoelectrons per frame:

$$
N_e = \eta\,A_d\,t_{\text{int}}\,\frac{\pi\,\tau_{\text{opt}}}{4F^2+1}\int R(\lambda)\,L_{q,\text{sensor}}(\lambda)\,d\lambda \;+\; i_{\text{dark}}t_{\text{int}}/q
$$

with $\eta$ the quantum efficiency (folded into $R(\lambda)$ if you prefer). Then:

$$
\mathrm{DN} = \min\!\left(\mathrm{DN}_{\max},\ \left\lfloor \frac{N_e + n_e}{N_{\text{well}}}\cdot 2^{N_{\text{bits}}} \right\rfloor\right)
$$

Key parameters: $\eta$, $N_{\text{well}}$ (well capacity), $i_{\text{dark}}(T_{\text{FPA}})$, read noise $\sigma_{\text{read}}$, $t_{\text{int}}$, $N_{\text{bits}}$.

Dark current follows an Arrhenius law, $i_{\text{dark}} \propto T^{3/2}e^{-E_g/2k_BT}$ — this is why MWIR sensors are cryocooled to 77–150 K. Model the cooldown transient if you want realistic startup behaviour.

**Cold shield efficiency.** A cooled system's cold stop limits the detector's view of warm surroundings. Effective f-number for background flux is set by the cold shield, not the lens: if they are mismatched ("cold shield inefficiency"), background flux rises and NETD degrades. Expose a `cold_shield_efficiency ∈ [0,1]` parameter; it is a real trade-study knob.

### 9.2 Microbolometers — uncooled LWIR

Different physics: absorbed radiation heats a thermally isolated membrane, changing its resistance. The FLIR Boson, a good reference core, uses a two-dimensional array of vanadium-oxide microbolometers at 12 µm pitch, where the temperature of each microbolometer varies in response to incident flux and the temperature change causes a proportional change in the detector's resistance [R24].

Membrane thermal balance:

$$
C_{\text{th}}\frac{d\Delta T}{dt} = \alpha_{\text{abs}}\Phi(t) - G_{\text{th}}\,\Delta T
$$

giving a first-order response with **thermal time constant** $\tau_{\text{th}} = C_{\text{th}}/G_{\text{th}}$ and frequency response

$$
\mathcal{R}(f) = \frac{\mathcal{R}_0}{\sqrt{1+(2\pi f \tau_{\text{th}})^2}},
\qquad \mathcal{R}_0 = \frac{\alpha_{\text{abs}}\,\beta\,I_{\text{bias}}\,R_0}{G_{\text{th}}}
$$

where $\beta = (1/R)(dR/dT)$ is the TCR, ≈ −2%/K for VOx. Typical $\tau_{\text{th}} = 8$–12 ms.

**This produces real motion smear.** At 60 Hz with $\tau_{\text{th}}=10$ ms, a moving object smears over roughly 0.6 frames. For a vehicle-mounted camera at speed, edges trail visibly. Implement it as a per-pixel exponential IIR across frames:

$$
S_n = S_{n-1} + \big(S^{\text{ideal}}_n - S_{n-1}\big)\big(1-e^{-\Delta t/\tau_{\text{th}}}\big)
$$

Two lines of code, and it is one of the strongest "this is a real uncooled camera" cues you can add. Most simulators omit it.

**FPA temperature coupling.** Bolometer response depends on the FPA's own temperature. In a TEC-less core this drifts with ambient and self-heating, which is the source of shutterless-camera drift. NETD depends not only on intrinsic material properties and ROIC design but also on operational parameters including integration time, bias conditions and frame rate, which influence responsivity, noise spectral density, thermal time constant effects and saturation behaviour — and FPA temperature can itself be treated as an optimisation parameter [R25].

Model as: $\text{gain}(T_{\text{FPA}}), \text{offset}(T_{\text{FPA}})$ as low-order polynomials, with $T_{\text{FPA}}$ driven by its own lumped thermal model from ambient plus power dissipation. §9.5 gives that thermal model, and the term it must not omit.

### 9.3 The datasheet-facing figures of merit

**Responsivity** $\mathcal{R} = V_{\text{out}}/\Phi$ [V/W]. **NEP** $= V_n/\mathcal{R}$ [W] — the power giving SNR = 1. **Specific detectivity**

$$
D^* = \frac{\sqrt{A_d \Delta f}}{\mathrm{NEP}}\quad[\text{cm·Hz}^{1/2}\text{·W}^{-1}]
$$

### 9.4 NETD

The two forms you will meet:

**Thermal-detector / datasheet form.** For an idealised system with no absorption in the medium or optics:

$$
\mathrm{NETD} = \frac{4F^2 V_n}{\mathcal{R}\,A_d\,L'} = \frac{4F^2}{A_d L'}\,\mathrm{NEP}
$$

where $A_d$ is the effective absorbing area, $V_n$ the total noise voltage in the system bandwidth, $\mathcal{R}$ responsivity, $F$ focal ratio, and $L'$ the change in power per unit area radiated by the object within the spectral band [R26]. NETD is defined as the object temperature change that makes the output SNR change by unity.

**Photon-count form (recommended for implementation, unambiguous):**

$$
\mathrm{NETD} = \frac{\sigma_{N,\text{total}}}{\partial N_e/\partial T},
\qquad
\frac{\partial N_e}{\partial T} = \eta A_d t_{\text{int}}\frac{\pi\tau_{\text{opt}}}{4F^2+1}\left(\frac{\partial L_q}{\partial T}\right)_B
$$

with $\sigma_{N,\text{total}}^2 = N_e + \sigma_{\text{dark}}^2 + \sigma_{\text{read}}^2 + \sigma_{1/f}^2$.

**Use NETD as a calibration handle, not as a noise generator.** The workflow is:

1. Build the full physical chain with your best parameter estimates.
2. Compute predicted NETD at a 300 K reference.
3. Scale the noise terms so predicted NETD matches the datasheet number.
4. Now the *spatial and spectral* structure of the noise is physical, and its *magnitude* matches the real device.

This is what makes a simulator both physical and matched to a specific camera. Note that NETD derivations differ in whether target and atmosphere temperatures are treated as independent; a modified systematic approach that decouples them applies across a wider range of target temperatures and atmospheric conditions [R27]. If you compare against published NETD numbers, check which convention was used.

---

### 9.5 The camera is in the weather too

§9.2 leaves $T_{\text{FPA}}$ to "its own lumped thermal model from ambient plus power dissipation". On anything that moves — a drone, a vehicle, a mast in wind — that model is missing its largest term: **forced convection over the camera body**.

The effect is measurable and it is not small. A UAV-borne LWIR camera hovering 15–20 min over fixed targets, checked against calibrated ground radiometers, shows its bias move from **−1.02 °C to +3.86 °C as wind rises from 0.8 to 8.5 m s⁻¹** [R44]. The ground reference sees the same surfaces at the same time, so wind-driven changes in the *true* surface temperature cancel in that difference and what remains is the instrument. An independent study of the same class of camera finds wind lowering the image mean by about 5.5 °C, raising its spatial standard deviation and deepening vignetting, with 20–40 min of warm-up before the core is stable at all [R45]; a third correlates measured temperature directly with FPA temperature and corrects on it [R46].

Model it as a second lumped-capacitance node — the physics of §6.4 applied to the camera instead of to the scene:

$$
C_{\text{cam}}\frac{dT_{\text{cam}}}{dt}
= P_{\text{diss}} + \alpha_{\text{cam}}Q_{\text{sol}} - h_c(v)\,A_{\text{cam}}\big(T_{\text{cam}}-T_{\text{air}}\big)
$$

with $h_c(v)$ the same forced-convection correlation §6.2 uses for surfaces, $T_{\text{FPA}}$ following $T_{\text{cam}}$ through a first-order lag, and §11.2's existing drift term converting $(T_{\text{FPA}}-T_{\text{FPA}}^{\text{cal}})$ into apparent scene temperature. Nothing downstream is new: the shutterless-drift path already exists and is simply never driven.

**One weather object, extended to the camera.** $T_{\text{air}}$, $v$ and $Q_{\text{sol}}$ come from the *same* weather series that drives §6 and §7. A scene must not fly a camera through still air while its surfaces are being wind-cooled.

**Fidelity, stated plainly.** The airflow field around a particular airframe is not computable in this model. $h_c(v)A_{\text{cam}}$ is one lumped coefficient fitted to a published bias-versus-wind curve that was measured on a different airframe, in a different attitude, looking down rather than up. It reproduces the **sign, the order of magnitude and the time constant**, and it is wrong in detail — a Level-B empirical fit in the sense of §4.2, and it must be switchable off. It earns its place anyway: at +3.86 °C the effect is some seventy times a Boson's NETD, larger than most of what §10 models carefully, and a simulator that omits it renders drone footage that is rock-steady in a way real drone footage never is.

### 9.6 Water and snow on the window

In rain or snow the camera's own front window collects drops. In LWIR a drop is opaque — water's 1/e depth
is 16 µm at 10 µm — and it sits far out of focus. It does not image as a shape. It blocks a fraction
$a$ of the beam footprint of every pixel whose rays cross it, and adds its own emission there:

$$
L_{\text{obs}}(\mathbf x) = \big(1-a(\mathbf x)\big)\,L_{\text{scene}}(\mathbf x) + a(\mathbf x)\,L_B(T_{\text{drop}})
$$

$a(\mathbf x)$ is the drop area inside pixel $\mathbf x$'s footprint on the window divided by that
footprint's area. At the entrance pupil every footprint is the whole aperture and $a$ is uniform. On a
window ahead of the pupil the footprints move with field angle, so a drop shades a soft-edged region of
the frame. Measured behind a LWIR windscreen camera, a drop lowered responsivity to about 90 % without
blurring the image [R115], which is this form with $a \approx 0.1$.

$T_{\text{drop}}$ is the window's temperature: near the wet bulb once wind and evaporation have settled
it, which ties this section to §9.5's camera node. Snow on the window is opaque in every band. In the
reflective bands a drop is transparent and refracts instead, a blur this model does not carry.
Hydrophobic coatings that bead the water are not modelled.

---

<a name="10-noise"></a>
## 10. Noise

### 10.1 Physical sources

| Source | Statistics | Scales as | Where |
|---|---|---|---|
| Photon shot | Poisson | $\sqrt{N_e}$ | all bands (dominant in MWIR/SWIR by day) |
| Dark shot | Poisson | $\sqrt{i_d t_{\text{int}}/q}$ | cooled devices, hot FPAs |
| Johnson/thermal | Gaussian | $\sqrt{4k_BT R\Delta f}$ | bolometers (dominant) |
| Temperature fluctuation | Gaussian | $\sqrt{4k_BT^2G_{\text{th}}\Delta f}$ | bolometers — the physical floor |
| Read / ROIC | Gaussian | constant | all |
| 1/f | pink | $\propto 1/f$ | bolometers, ROIC |
| FPN (gain + offset) | fixed spatial | scene-dependent | all — **dominant after NUC decay** |

### 10.2 The 3D noise model — use this, not a single sigma

The NVESD three-dimensional noise decomposition splits total noise into components along temporal ($t$), vertical ($v$) and horizontal ($h$) directions. In the standard nomenclature: $\sigma_{TVH}$ is random spatio-temporal noise from detector temporal noise; $\sigma_{VH}$ is random spatial (bi-directional fixed-pattern) noise from pixel processing, detector-to-detector non-uniformity and 1/f; $\sigma_V$ is fixed row noise (line-to-line non-uniformity); $\sigma_H$ is fixed column noise from scan effects and detector-to-detector non-uniformity [R28].

Full set and how to synthesise each:

| Component | Physical meaning | Synthesis |
|---|---|---|
| $\sigma_{TVH}$ | random spatio-temporal (pixel temporal noise) | i.i.d. Gaussian per pixel per frame |
| $\sigma_{VH}$ | fixed 2-D pattern (pixel non-uniformity, 1/f) | one fixed 2-D Gaussian field, slowly drifting |
| $\sigma_V$ | fixed row noise | one Gaussian value per row, fixed |
| $\sigma_H$ | fixed column noise | one Gaussian value per column, fixed |
| $\sigma_{TV}$ | temporal row noise (ROIC line noise) | new Gaussian per row per frame |
| $\sigma_{TH}$ | temporal column noise | new Gaussian per column per frame |
| $\sigma_T$ | frame-to-frame bounce | one Gaussian per frame (global) |

$$
\sigma_{\text{total}}^2=\sigma_T^2+\sigma_V^2+\sigma_H^2+\sigma_{TV}^2+\sigma_{TH}^2+\sigma_{VH}^2+\sigma_{TVH}^2
$$

This decomposition is directly implementable, and it is what real EO/IR test equipment reports, so **you can parameterise it from measurements of the actual camera you intend to model.** Programs of record combine signal intensity transfer function (SITF), 3-D noise, IFOV and MTF measurements into a measured-system component usable directly in the Night Vision Integrated Performance Model [R29].

Practical starting ratios for an uncooled LWIR core after NUC:
$\sigma_{TVH}:\sigma_{VH}:\sigma_H:\sigma_V \approx 1.0 : 0.3 : 0.15 : 0.08$, all in NETD units. For a cooled MWIR core, $\sigma_{VH}$ is smaller and $\sigma_{TVH}$ dominates.

Directional noise (row/column striping) is exactly what makes thermal imagery look *thermal*, and it is what most detection networks latch onto. Getting it wrong is the biggest single contributor to sim-to-real gap in IR perception — larger than radiometric error.

### 10.3 Temporal structure

Real IR noise is not white in time. Temporal stripe noise arises when the ADC is affected by electronic thermal noise, introducing fluctuating stripe structures along the temporal dimension; it is commonly modelled as a 1-D Gaussian distributed along one image axis [R30]. Implement the fixed-pattern terms with a slow random walk (correlation time of seconds to minutes) rather than freezing them: that produces the characteristic "pattern breathing" between NUC events.

### 10.4 Bad pixels

Every real FPA has them. Model:
- **Dead** (stuck low), **hot** (stuck high), **flickering** (random telegraph noise), **blinking** (intermittent).
- Typical: 0.05–0.5% of pixels, clustered slightly (use a Poisson cluster process, not uniform).
- Cameras replace them with neighbour interpolation, which leaves a **detectable smoothed footprint**. Simulate the defect *and* the replacement — the replacement artefact is what a detector actually sees.
- The replacement map is a **calibration-time artefact**. The camera interpolates over the defect map it
  was shipped with, so a pixel that fails afterwards is not on the map and reaches the output as an
  isolated stuck-high or stuck-low pixel. A single bright dot in an otherwise featureless sky frame is
  usually this, and a sky-target detector must be trained on frames that contain it. Simulate two
  populations: factory defects (replaced, smoothed footprint) and late defects (not replaced).

---

<a name="11-signal-chain"></a>
## 11. Signal chain

Everything after the ADC. This stage is where most of the *visual* character of a thermal image is created, and it is the part most simulators get wrong by skipping.

### 11.1 Order of operations

```
raw DN → bad-pixel replace → NUC (2-pt gain/offset) → temporal filter
       → [radiometric linearisation → T_app]  (radiometric mode)
       → AGC / DRC → gamma → polarity → palette → 8-bit output
```

Note the fork: a **radiometric** camera exposes the linear branch (calibrated $T_{\text{app}}$ per pixel); a **non-radiometric** core exposes only the AGC branch. Emit both — perception stacks usually consume the 8-bit AGC image, while your validation needs the linear one. §11.5 continues the radiometric branch past $T_{\text{app}}$ to the number the camera actually reports.

**The raw DN must hold the coldest scene.** A microbolometer's DN is referenced to the shutter,
$\mathrm{DN} \propto \Phi_{\text{scene}} - \Phi_{\text{shutter}}$ plus a mid-scale pedestal, so the low end
of the 14-bit range sits far below any natural scene: zero radiance is still on scale. A camera
datasheet's "scene dynamic range" (e.g. −40 °C … +140 °C high gain) is the range over which the
radiometry is *specified*, not an ADC floor. A model that puts DN 0 at the lower end of the datasheet
range clips every clear sky colder than it — the zenith sky in dry air reads −50 … −70 °C in LWIR — to a
single code, which erases the sky's structure from the display branch and, worse, hands the AGC one
enormous histogram bin. The ADC transfer's lower bound must lie below the coldest sky the scene can
produce (spec issue S53).

### 11.2 Two-point NUC

$$
\mathrm{DN}^{\text{corr}}_{ij} = G_{ij}\big(\mathrm{DN}_{ij}-O_{ij}\big),\qquad
G_{ij}=\frac{\overline{\mathrm{DN}^{H}}-\overline{\mathrm{DN}^{L}}}{\mathrm{DN}^{H}_{ij}-\mathrm{DN}^{L}_{ij}},\quad
O_{ij}=\mathrm{DN}^{L}_{ij}
$$

calibrated against two blackbody temperatures. This is the standard two-point technique, and NETD and correctability are the two parameters used to characterise an FPA against a blackbody radiator [R31].

**Model the residual, not the ideal.** Simulate:
- coefficients calibrated at $T_{\text{FPA}}^{\text{cal}}$, applied at current $T_{\text{FPA}}$ → drift $\propto (T_{\text{FPA}}-T^{\text{cal}}_{\text{FPA}})$;
- residual non-uniformity growing between shutter events (flat-field correction, FFC);
- the FFC event itself: a shutter closes, the image freezes for 0.5–1 s, then the pattern resets.

**The shutter is a radiance reference, not a reset button.** At the FFC the shutter fills each pixel's
cone at $L_B(T_{\text{shutter}})$ while the out-of-cone housing view of §8.2 is unchanged, so the
offset snapshot is

$$
O_{ij} = \mathcal{S}\Big[A_d\,\Omega_{\text{eff}}\mathrm{RI}_{ij}\,L_B(T_{\text{shutter}})
       + A_d\big(\pi-\Omega_{\text{eff}}\mathrm{RI}_{ij}\big)L_B\!\big(T^{\text{FFC}}_{\text{housing}}\big)\Big]
$$

Subtract it from the §8.2 power at a later time $t$, apply the factory gain $G_{ij}$ (ideally
$1/\mathrm{RI}_{ij}$), and what remains on a uniform scene is three terms:

$$
\mathrm{DN}^{\text{corr}}_{ij} \propto
\underbrace{\Omega_{\text{eff}}\Big[\tau_{\text{opt}}L_{\text{scene}} + (1-\tau_{\text{opt}})L_B(T_{\text{housing}}) - L_B(T_{\text{shutter}})\Big]}_{\text{uniform: the signal}}
+\underbrace{\Big(\tfrac{\pi}{\mathrm{RI}_{ij}} - \Omega_{\text{eff}}\Big)\Big[L_B\big(T_{\text{housing}}(t)\big) - L_B\big(T^{\text{FFC}}_{\text{housing}}\big)\Big]}_{\text{radial: housing drift since the FFC}}
+\underbrace{\big(1 - G_{ij}\mathrm{RI}_{ij}\big)\,\Omega_{\text{eff}}\,\tau_{\text{opt}}\Big[L_{\text{scene}} - L_B(T_{\text{shutter}})\Big]}_{\text{radial: gain-map error}}
$$

What this predicts, and a clear-sky reference frame confirms:

- The residual after an FFC is a **smooth radial bowl**, not a white per-pixel field. Its depth is the
  housing's radiance change since the event, weighted by $\pi/\mathrm{RI}_{ij} - \Omega_{\text{eff}}$,
  so it is deepest in the corners: a housing that has *cooled* since the shutter closed (a camera
  carried outside, or a lens radiating to a cold sky) darkens the corners and leaves a bright centre;
  one that has warmed does the opposite. The white $(T_{\text{FPA}} - T^{\text{cal}}_{\text{FPA}})$
  bullet above is a different, multiplicative mechanism — pixel responsivity drift [R50] — and stays.
- A scene far from the shutter temperature multiplies any gain-map error. The clear sky is the
  extreme case, which makes it both the worst scene for non-uniformity and the right Tier 4 scene for
  measuring it: point the camera at nothing and the residual is all that is left.
- $T_{\text{shutter}}$ is a state, not a constant. The shutter sits beside the FPA and follows it with
  its own lag [R49]; when it is not at the housing temperature the uniform term carries a radiometric
  bias that a two-point NUC cannot see.

That freeze is a real behavioural artefact that a perception stack must survive. Simulating it is worth more than another decimal place of radiometry. Some cameras avoid it: shutterless approaches process consecutive scene and internal-shutter images to stabilise response [R32], and scene-based methods estimate fixed-pattern noise from pixel-sized translations of the FPA, requiring neither shutter nor elaborate calibration and being invariant to noise magnitude and robust to unknown camera and inter-scene movement [R33]. If you model a shutterless core, use a slow-drift + scene-based-correction residual instead of periodic freezes.

### 11.3 AGC / dynamic range compression

A 14–16 bit radiometric image must become 8 bits. The mapping choice changes the image drastically and **is part of the sensor model**, not a display detail.

**Linear AGC with percentile clipping:**
$$
y = \mathrm{clip}\!\left(\frac{x - x_{p_{\text{lo}}}}{x_{p_{\text{hi}}} - x_{p_{\text{lo}}}},0,1\right)^{1/\gamma}
$$
with $p_{\text{lo}},p_{\text{hi}}$ typically 0.5% / 99.5%.

**Plateau equalisation (what most thermal cores actually use):** histogram equalisation with the per-bin count clipped at a plateau value $P$ before integrating the CDF. Low $P$ → approaches linear; high $P$ → full HE. This is the algorithm behind the characteristic thermal "look."

**$P$ only clips a histogram with a spike.** $P$ is a fraction of $N_{\text{pixels}}$ per bin (FLIR's
default is 7 % [R51]), so it limits a bin only when one DN value holds more than that share of the
frame — a *uniform* background such as a clear sky a few counts wide. A background that is itself
spread over thousands of DN (cumulus clutter spanning 60 K at ~170 DN/K) never reaches the plateau,
and the operator degenerates to full HE: grey shades are handed out in proportion to pixel count. A
small target then gets the share of the ramp that it has of the frame. FLIR says it in so many words:
"an image with 60 % sky will devote 60 % of the available 8 bit shades to the sky" [R51]. Measured
on this repository's Phantom 4 clip (spec issue S52), a drone covering 0.6 % of the frame received
**three** of 256 grey levels for a 15 → 38 °C spread of part temperatures. That is correct plateau
behaviour; it is the wrong operator to call the camera's default.

**Information-based equalisation (a Boson's factory default).** The frame is split into a low-pass
image $x_{LP}$ (an edge-preserving smoother; FLIR's control is a range sigma, *Smoothing Factor*) and
$x_{HP} = x - x_{LP}$. Plateau equalisation runs on $x_{LP}$, and the histogram is then re-weighted
so that bins holding high-pass content receive more shades:

$$
h(b) = \min\big(h_{LP}(b),\,P N\big) + \beta_{\text{info}}\,\frac{\sum_b \min(h_{LP}, PN)}{\sum_{i}|x_{HP,i}|}\sum_{i\in b}|x_{HP,i}|
$$

so a textured target on a smooth sky earns shades in proportion to its detail, not its area.
$\beta_{\text{info}}$ (the relative weight of the information histogram) is **not published**; FLIR
describes the weighting only qualitatively, so the form above is a flagged approximation of a
proprietary operator, and its free parameter must be fitted to public footage before it is called a
Boson (Tier 4). The high-pass layer is added back after the mapping at the local slope of the
transfer, $y = \mathrm{LUT}(x_{LP}) + g_{\text{DDE}}\,\mathrm{LUT}'(x_{LP})\,x_{HP}$, which is what DDE
means on this core: detail is shown at the gain of its surroundings, and *Detail Headroom* reserves
$h$ of the range at each end so that $g_{\text{DDE}}\,x_{HP}$ does not rail.

**Linear Percent.** Every equalising mode is blended with the min–max linear map,
$\mathrm{LUT} = (1-\lambda)\,\mathrm{LUT}_{\text{eq}} + \lambda\,\mathrm{LUT}_{\text{lin}}$. $\lambda$
restores the ordering of *how much* hotter one object is than another (FLIR's example: a stove and a
person both "hot" under pure HE), at the price of shades spent on empty DN [R51]. On a sky scene it
is also what spreads a drone's motors away from its shell.

**Why it matters for automotive:** a hot exhaust entering frame collapses contrast on everything else, because AGC is global. That failure mode is real, is a genuine hazard for perception, and only appears in simulation if you model AGC. Add ROI-weighted and locally-adaptive variants as options.

**The families are shared; the parameters are the camera.** Across vendors the uncooled-core AGC is
one construction with differently named controls: a high clip (plateau) and a **low clip** — a
constant added to every occupied bin, so a sparsely populated temperature keeps a floor of shades —
in the Lepton family [R52]; Linear Percent, tail rejection and a **max gain** capping the transfer's
slope in the Boson family [R51]; auto-gain with a "maximal allowed stretching" and an equalisation
strength in InGaAs SWIR cores [R53]; CLAHE-style local equalisation in the literature for the rest.
Model the families (linear, global plateau, local tiled, detail-weighted) and the shared controls
(linear percent, low clip, max gain), and express a specific camera as a parameter set of them.
Max gain matters most for the sky lane: a clear sky a few DN wide is the blandest scene there is,
and uncapped equalisation stretches its noise across the whole ramp.

**A Boson's factory values are published.** [R51] p. 5 shows them in the camera's own control
panel: Information-Based on, Plateau 7 %, Linear Percent 20 %, Max Gain 1.38, Detail Headroom 12,
DDE 0.95, Smoothing Factor 1250, Tail Rejection 0, ACE 0.97, Damping 85. A Boson config carries
these, not values tuned for a picture (ADR 0152 maps each one and lists the three that do not map).

**A band's display chain follows its detector.** In the emissive bands the scene radiance moves by a
small factor between night and noon, so exposure is fixed and the display work is the AGC. In the
reflective bands (NIR, SWIR) it moves by five to six decades, and the camera is exposure-driven like
a visible camera: **auto-exposure** sets the integration time and switches gain modes, and only then
does a lighter AGC run [R53]. A silicon NIR sensor's display is the visible-camera chain — black
level, linear stretch, display gamma — not a bolometer's equaliser. Do not give a reflective-band
camera an uncooled LWIR core's ISP.

**Digital detail enhancement (DDE):** high-pass boost added back over the compressed image. Model as unsharp mask with a configurable gain — it is why real thermal images look "crunchy."

### 11.4 Polarity and palette

White-hot / black-hot inversion, plus colour LUTs (Ironbow, Rainbow, Lava, Arctic). Trivial, but expose them: many downstream models are sensitive to palette, and mismatched palette between training and deployment is a classic silent failure.

---

### 11.5 Radiometric retrieval — what the camera believes about $\varepsilon$

The radiometric branch of §11.1 produces apparent temperature, $T_{\text{app}}=L_B^{-1}(L)$: the temperature of the blackbody that would emit the measured radiance. A real radiometric camera does not stop there. It applies operator-set parameters — an emissivity, a reflected (background) temperature, an atmospheric transmittance and temperature — and inverts [R43]

$$
W_{\text{tot}}=\varepsilon\,\tau_{\text{atm}}W_{\text{obj}}
+(1-\varepsilon)\,\tau_{\text{atm}}W_{\text{refl}}
+(1-\tau_{\text{atm}})W_{\text{atm}}
$$

for $W_{\text{obj}}$, reporting $T_{\text{meas}}=L_B^{-1}(W_{\text{obj}})$.

**Those parameters are the camera's beliefs, not the scene's truth** — the structure §11.2 already uses for the housing temperature. Set them equal to the truth and the retrieval returns the kinetic temperature; set them wrong and the error is the product. Computed in this project's own Boson band, for a 300 K target against a 250 K background:

| true $\varepsilon$ | camera assumes | error in $T_{\text{meas}}$ |
|---|---|---|
| 0.90 | 0.90 | 0, by construction |
| 0.90 | 0.89 | +0.43 K |
| 0.90 | 0.95 | −2.04 K |
| 0.85 | 0.95 | −4.11 K |
| 0.09 | 0.95 | −43.7 K |

The error grows with target-to-background contrast: the same 0.95-on-0.90 mistake costs −2.98 K on a 330 K target. Note that 0.01 of emissivity error is already 0.43 K, some nine times a Boson's NETD — while the common rule of thumb that it costs "about 1 % of the reading" would predict ≈ 3 K here, overstating it sevenfold by ignoring how steep $\partial L/\partial T$ is in the LWIR (§3.4).

**Emit both.** $T_{\text{app}}$ is what the radiance says; $T_{\text{meas}}$ is what an operator would read off the screen. They differ by exactly the quantity this section is about, and a simulator that reports only one of them cannot be used to study measurement error at all.

---

<a name="12-band-specialisation"></a>
## 12. Band specialisation: one core, four cameras

This is the scalability answer. **Nothing in §2–§11 is band-specific.** A band is a data file.

### 12.1 What changes per band

| Aspect | NIR | SWIR | MWIR | LWIR |
|---|---|---|---|---|
| Range (µm) | 0.75–1.0 | 0.9–1.7 | 3.0–5.0 | 7.5–13.5 |
| Detector | Si CMOS | InGaAs | InSb / MCT / T2SL | VOx / a-Si bolometer |
| Cooling | none | TEC | 77–150 K | none |
| Detector model | photon | photon | photon | **bolometer** |
| Self-emission | negligible | negligible | **significant** | dominant |
| Illumination needed | yes | yes | partial | **no** |
| Night source | moon, IR LED | **airglow** | self-emission | self-emission |
| Typical $F$ | 1.4–2.8 | 1.4–4 | 2–4 | **1.0–1.4** |
| Typical pitch | 3–5 µm | 10–15 µm | 10–15 µm | 12–17 µm |
| Glass | transparent | transparent | opaque-ish | opaque |
| Water | absorbing (dark) | strongly absorbing | reflective | ε≈0.96, emissive |
| Vegetation | **very bright** (red edge) | bright | moderate | ε≈0.97 |
| Solar glint | strong | strong | **strong** | negligible |
| Atmospheric limiter | scattering | water vapour | CO₂ 4.2–4.4 µm | water continuum |
| Fog penetration | poor | poor | fair | **best** |
| Dominant noise | shot | shot | shot | Johnson + FPN |

### 12.2 Configuration schema

A band is fully described by a file like this. Everything above is either derived from it or shared.

```yaml
sensor:
  name: "flir_boson_640_lwir"
  band:
    lambda_min_um: 7.5
    lambda_max_um: 13.5
    spectral_response: "spectra/responses/boson_vox.csv"   # λ (µm), R(λ) peak-normalised; relative to data/
    regime: emissive                                # emissive | reflective | mixed

  optics:
    f_number: 1.0
    focal_length_mm: 14.0
    transmittance: 0.92
    housing_temp_mode: coupled                      # fixed | ambient | coupled
    cold_shield_efficiency: 1.0                     # cooled systems only
    distortion:
      model: brown_conrady                          # or kannala_brandt / ftheta
      coeffs: [-0.28, 0.09, 0.0, 0.0, 0.0]
    vignetting_cos4: true

  fpa:
    type: bolometer                                 # bolometer | photon
    width: 640
    height: 512
    pitch_um: 12.0
    fill_factor: 0.90
    frame_rate_hz: 60
    bit_depth: 16
    # bolometer
    thermal_time_constant_ms: 10.0
    tcr_per_k: -0.021
    # photon (unused when type == bolometer)
    quantum_efficiency: null
    well_capacity_e: null
    integration_time_ms: null
    dark_current_model: null

  noise:
    netd_mk_at_300k: 50.0                           # anchor: scales everything
    ratios_3d:                                      # relative to sigma_TVH
      tvh: 1.00
      vh:  0.30
      h:   0.15
      v:   0.08
      tv:  0.05
      th:  0.05
      t:   0.02
    fpn_drift_tau_s: 120.0
    bad_pixel_fraction: 0.0015
    bad_pixel_cluster_lambda: 1.6

  nuc:
    mode: shuttered                                 # shuttered | shutterless | ideal
    ffc_interval_s: 180
    ffc_freeze_ms: 700
    residual_gain_ppm_per_k: 900
    residual_offset_mk_per_k: 45

  isp:
    agc: plateau_equalization                       # linear | plateau_equalization | none
    plateau: 0.012
    clip_percentiles: [0.005, 0.995]
    gamma: 1.0
    dde_gain: 0.35
    polarity: white_hot
    palette: gray                                   # gray | ironbow | rainbow | lava

  outputs:
    radiance_linear: true                           # float32, W/m2/sr
    apparent_temperature: true                       # float32, K
    dn_16: true
    display_8: true
```

To add SWIR: copy the file, change `regime: reflective`, `type: photon`, load an InGaAs $R(\lambda)$, enable the night-illumination model, and enable glass transmission in the material resolver. No core code changes. **That is the test of whether your architecture is actually scalable** — if adding a band requires touching the radiance kernel, the architecture is wrong.

### 12.3 Material schema

Materials should be **spectral** where data exists, not per-band scalar, if you ever want MWIR right, or want a camera whose range is not a nominal band's to read its own value. The schema takes the one authored quantity (emissivity or reflectance, §4.1) in up to three forms, layered wavelength by wavelength:

```yaml
materials:
  car_paint_black:
    thermal:
      density_kg_m3: 7800          # substrate-dominated
      specific_heat_j_kgk: 470
      conductivity_w_mk: 45
      thickness_m: 0.0012
      solar_absorptivity: 0.94
    optical:
      spectral_emissivity: "spectra/car_paint_black.csv"   # λ, ε(λ) 0.3–15 µm
      # or segments: [{file: "sw.csv", quantity: reflectance}, "lw.csv"]
      emissivity_per_band: { mwir: 0.88 }   # only where the curve has no data
      emissivity: 0.90                     # grey: everything else
      roughness_per_band:  { nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12 }
      angular_model: { type: fresnel, n_k_file: "nk/acrylic_paint.csv" }
      transmittance_per_band: { nir: 0.0, swir: 0.0, mwir: 0.0, lwir: 0.0 }
```

For a camera with response R(λ), the band value is the Planck × response average (§3.2, ADR 0010) of

$$
\varepsilon(\lambda)=\begin{cases}
\varepsilon_{\text{curve}}(\lambda) & \text{where a segment has data}\\
\varepsilon_{B} & \text{else, the camera band's per-band value}\\
\varepsilon_{\text{grey}} & \text{else}
\end{cases}
$$

- **Any non-empty combination of the three forms is valid.** A grey value alone is a grey body in every band. A table alone is the old four numbers. A curve alone is a spectral material.
- **The camera's own response is used**, so the same curve gives a 6–13 µm camera and a 7.5–13.5 µm camera different values. A material without a curve gives both the same value, because nothing more is known.
- **Gaps are never extrapolated.** A band the curve does not cover and nothing fills is refused.
- **Each number is authored once.** A per-band value for a band the curve already fully covers is refused, since the curve would silently overrule it.
- **The share of the band weight the curve supplied is reported** as `curve_fraction`. Below 1, the value assumes the material is flat across the gap.

ADR 0175 records the choice.

Note `roughness_per_band` decreasing with wavelength — that is §4.3 made concrete, and it is the parameter that makes vehicles reflect the sky correctly in LWIR.

---

<a name="13-isaac-sim"></a>
## 13. Isaac Sim 6.0 implementation mapping

### 13.1 What NVIDIA officially recommends

As of Isaac Sim 6.0 GA there is still no built-in IR camera. On the open feature request (`isaac-sim/IsaacSim` Discussion #298), the maintainer's recommended approach is [R34]:

1. **Encode temperature as emission** — assign an OmniPBR material per surface with emission enabled, using the emissive colour channel to carry the object's temperature value.
2. **Capture the thermal signal** — use the `PtSelfIllumination` AOV to capture only the self-illumination component, giving a clean per-pixel temperature readout without contamination from reflected or bounced light. Alternatively render `HdrColor` with max bounces set to 0 to suppress global illumination.
3. **Implement the sensor model in SPG** — feed the AOV into a Sensor Processing Graph implementing NETD noise, spectral response curves, vignetting, and other LWIR/MWIR sensor behaviour.

The framework is `omni.rtx.spg`, which enables running custom GPU code as post-processing passes on RTX render outputs (AOVs), with all computation staying on the GPU and no CPU-side data transfer in the processing pipeline [R35].

**Two honest caveats on this advice:**

- The discussion describes chaining SPG stages "using Python or Warp kernels" [R34]; the actual SPG documentation describes **CUDA kernels + Lua launch scripts + USD shader definitions**, compiled at runtime with NVRTC [R35]. If you were confused trying to follow the discussion, that is why. §13.4 gives an alternative Python/Warp path that does work.
- Step 1 is only a *temperature transport* trick, and it is a lossy one. Emission channels are colour channels — you must handle precision and the fact that the renderer's emission is not physically your radiance (§13.3).

### 13.2 How SPG actually works

Every SPG shader is three files [R35]:

| File | Role |
|---|---|
| `.cu` | CUDA kernel — the GPU transform, an `extern "C" __global__` function |
| `.cu.lua` | Lua launch script — validates inputs, allocates outputs, returns launch config; called every frame |
| `.usda` | USD shader definition — declares inputs/outputs, references the CUDA source |

Wiring: `RenderVar.omni:rtx:aov` → `Shader.inputs:X` → `Shader.outputs:Y` → `RenderVar.omni:rtx:aov.connect`. Shaders chain directly output-to-input with no intermediate RenderVar. Execution order comes from the dependency graph, not from `orderedVars` order. Kit must run with `--enable omni.rtx.spg`.

Three things in the API are directly useful to you:

- **`cuda.static(fn, ...)`** caches a computation and reuses it unless its arguments change. This is exactly how you upload your Planck LUT once instead of per frame — the docs show precisely this pattern for a Gaussian kernel [R35].
- **`rtx.frameId`** global — use it to seed temporal noise so frames are reproducible.
- **`cuda.image(w,h,dtype)`** for image outputs, **`cuda.empty(shape,dtype)`** for LUTs and statistics buffers.

Known limitations to design around [R35]: only local `.cu`/`.cu.lua`/`.usda` files; SPG shader nodes must not be nested under a Material prim; standard-library nodes handle 2D textures only, with integer texture formats (`SINT`/`UINT`) and buffer-backed resources unsupported. Also, `display_render_var` only works for RGBA unorm textures — for float AOVs, capture to disk with `FileCapture` instead of expecting them in the viewport.

### 13.3 The AOV design — the part that decides whether this works

Do **not** try to make the renderer produce IR radiance. Make it produce a **G-buffer of physical quantities** and do all radiometry in SPG. Concretely:

| AOV | Carries | Format | Notes |
|---|---|---|---|
| `PtSelfIllumination` (or custom) | encoded surface temperature | **float32** | see encoding below |
| `DiffuseAlbedo` or material-ID | emissivity lookup key | uint8/float | index into a material table |
| `Normal` | surface normal (world or view) | float16 ok | for $\varepsilon(\theta)$ and sky-view factor |
| `DistanceToCamera` | path length $d$ | float32 | atmospheric attenuation |
| `AmbientOcclusion` (or custom) | sky-view factor $V_s$ | float16 ok | reflected-sky term |
| `MotionVectors` | image-plane velocity | float16 ok | motion MTF, bolometer smear |
| `SemanticSegmentation` | ground truth | uint | training labels |

**Temperature encoding — get this right or nothing else matters.** Emissive colour channels are commonly fp16 or unorm. At 300 K, fp16 spacing is 0.25 K, five times coarser than a 50 mK NETD; an 8-bit unorm channel over 200–400 K gives 0.78 K per code. Either destroys the sensor entirely. Encode instead:

$$
c = \frac{T - T_{\text{ref}}}{T_{\text{span}}},\qquad T_{\text{ref}}=200\ \mathrm{K},\ T_{\text{span}}=800\ \mathrm{K}
$$

into a **float32** AOV, or split across RGB channels of a fp16 target (coarse in R, fine in G) if float32 is unavailable. Verify empirically: render a ramp of known temperatures, decode, and check round-trip error is < 10 mK. **Do this before writing any other code.**

### 13.4 The pipeline

```
                 RTX renderer
                      │
     ┌────────────────┼────────────────┬──────────────┬───────────────┐
  T_encoded      materialID         Normal      DistToCam      SkyView
     │                │                │             │             │
     └────────────────┴───────┬────────┴─────────────┴─────────────┘
                              ▼
              ┌──────────────────────────────────┐
   band LUTs  │  SPG#1  band_radiance.cu         │   ε(θ) from Fresnel/empirical
   (cuda.     │  L = ε(θ)·B_LUT(T)               │   L_env from sky model + V_s
    static)   │      + (1-ε(θ))·L_env            │
              └──────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  SPG#2  atmosphere.cu            │   τ = exp(-γ·d)
              │  L' = τ·L + (1-τ)·B(T_air)       │
              └──────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  SPG#3  optics.cu                │   π·τ_opt/(4F²+1), cos⁴θ,
              │  Φ = f(L'), + self-emission      │   + Φ_self(T_housing)
              └──────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  SPG#4  detector.cu              │   bolometer IIR (τ_th) or
              │  DN_ideal = S(Φ)                 │   photon → e⁻ → well → ADC
              └──────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  SPG#5  noise.cu                 │   3D noise: TVH,VH,V,H,TV,TH,T
              │  + NUC residual + bad pixels     │   seeded by rtx.frameId
              └──────────────┬───────────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  SPG#6  isp.cu                   │   AGC/plateau EQ, DDE,
              │  → DN16 (linear) + DISP8         │   gamma, polarity, palette
              └──────────────┬───────────────────┘
                             ▼
                    IrRadiance / IrApparentT / IrDN16 / IrDisplay8  → ROS 2
```

Six kernels, each independently testable. MTF and downsampling happen either as SPG#0 (supersample-then-box-filter) or by rendering at 3–4× resolution and letting SPG#3 do the box filter — the latter is simpler and gives correct aliasing (§8.3).

### 13.5 Kernel sketch

```cuda
// band_radiance.cu — SPG stage 1
extern "C" __global__ void band_radiance(
    int width, int height,
    float T_ref, float T_span,
    float L_sky, float L_ground,          // band radiances, W/m^2/sr
    cudaTextureObject_t  encT,            // float32: (T - T_ref)/T_span
    cudaTextureObject_t  matId,           // material index
    cudaTextureObject_t  normalTex,       // world normal + view dir dot in .w
    cudaTextureObject_t  skyView,         // V_s in [0,1]
    const float*  __restrict__ planckLUT, // L_B(T), N entries over [T0,T1]
    int    lutN, float lutT0, float lutT1,
    const float2* __restrict__ matEps,    // per material: (eps0, angular a)
    cudaSurfaceObject_t  outRadiance)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float T = tex2D<float>(encT, x, y) * T_span + T_ref;

    // Planck band radiance by LUT with linear interpolation
    float u  = (T - lutT0) / (lutT1 - lutT0) * (lutN - 1);
    u = fminf(fmaxf(u, 0.f), lutN - 1.001f);
    int   i0 = (int)u;  float fr = u - i0;
    float B  = planckLUT[i0] * (1.f - fr) + planckLUT[i0 + 1] * fr;

    // directional emissivity, empirical model (level B)
    int   m        = (int)tex2D<float>(matId, x, y);
    float2 ep      = matEps[m];                 // (eps0, a)
    float  cosTh   = fabsf(tex2D<float4>(normalTex, x, y).w);
    float  omc     = 1.f - cosTh;
    float  eps     = ep.x * (1.f - ep.y * omc*omc*omc*omc);   // p = 4

    // reflected environment via sky-view factor
    float Vs   = tex2D<float>(skyView, x, y);
    float Lenv = Vs * L_sky + (1.f - Vs) * L_ground;

    float L = eps * B + (1.f - eps) * Lenv;
    surf2Dwrite<float>(L, outRadiance, x * sizeof(float), y);
}
```

with the launch script uploading the LUT once:

```lua
function band_radiance(inputs, outputs)
    local h = inputs["EncodedT"].shape[1]
    local w = inputs["EncodedT"].shape[2]
    outputs["Radiance"] = cuda.image(w, h, cuda.float)

    -- LUT built once, reused unless the band changes
    local lut = cuda.static(function(t0, t1, n) 
        local t = {}
        for i = 0, n-1 do t[#t+1] = planck_band(t0 + (t1-t0)*i/(n-1)) end
        return cuda.array(t, cuda.float)
    end, inputs["lutT0"].value, inputs["lutT1"].value, inputs["lutN"].value)

    return cuda.kernel({
        args = { cuda.int(w), cuda.int(h),
                 cuda.float(inputs["T_ref"]), cuda.float(inputs["T_span"]),
                 cuda.float(inputs["L_sky"]), cuda.float(inputs["L_ground"]),
                 cuda.TextureObject(inputs["EncodedT"]),
                 cuda.TextureObject(inputs["MatId"]),
                 cuda.TextureObject(inputs["Normal"]),
                 cuda.TextureObject(inputs["SkyView"]),
                 lut,
                 cuda.int(inputs["lutN"].value),
                 cuda.float(inputs["lutT0"]), cuda.float(inputs["lutT1"]),
                 cuda.array(inputs["MatEps"]),
                 cuda.SurfaceObject(outputs["Radiance"]) },
        block = {32, 32},
        grid  = {math.ceil(w/32), math.ceil(h/32)},
    })
end
```

*(`planck_band` here stands for your band integral — in practice generate the LUT offline and load it as a binary asset rather than computing it in Lua.)*

### 13.6 The Python/Warp path — start here

If SPG's CUDA+Lua+USD triple is more friction than you want for a first prototype, there is a simpler route that gets you the same physics:

1. Create the camera with `isaacsim.sensors.experimental.rtx` — `RtxCamera` for authoring, `CameraSensor` for runtime, which attaches Replicator annotators and provides `get_data()` returning numpy or **warp arrays** [R36].
2. Attach annotators for the AOVs in §13.3.
3. Run stages 1–6 as **Warp kernels** on the returned warp arrays (they are already on the GPU — no round trip if you keep everything in warp).
4. Publish over ROS 2.

Trade-off: one extra buffer hop and Python-side per-frame dispatch, versus in-pipeline execution. For a 640×512 sensor this costs well under a millisecond. **Prototype here, port hot stages to SPG once the physics is verified.** Verifying physics inside a runtime-compiled CUDA kernel wired through USD is not where you want to spend your debugging budget.

### 13.7 The reflection problem

This is the real limitation of the emission-encoding approach and you should know it up front. `PtSelfIllumination` deliberately gives you a temperature buffer *without* bounced light — which is what you want for a clean $T$ readout, and which also means you get **no reflections at all**. But §4 and §5.3 established that reflections are a large part of what LWIR actually looks like, especially on vehicles and glass.

Options, cheapest first:

1. **Sky-view factor only** (§5.3a). Ambient occlusion AOV as $V_s$, blend sky and ground radiance. No specular reflections, but gets the broad behaviour and the cold-roof effect. **Start here.**
2. **Low-res T-cubemap.** Render a second, small render product (e.g. 64×64 per face) of the encoded-T field from a probe near the camera; convert to band radiance and use it as a reflection probe in SPG. Adds specular sky/building reflections. Good value.
3. **Second render product with bounces enabled**, where emission is proportional to band radiance rather than temperature. Physically the most correct available, requires care to keep the two encodings from mixing, and costs a full extra render.
4. **Warp-based ray cast** against the T-field for mirror directions on flagged specular materials only. Surgical and cheap if only a few materials need it.

### 13.8 Performance

The SPG kernels are trivial — six element-wise passes over 640×512 is microseconds. Your cost is the renderer: supersampling 4× means 2560×2048, and a second render product for reflections doubles it again. Budget accordingly, and use `TiledCameraSensor` if you need many cameras for data generation [R36].

---

<a name="14-unreal"></a>
## 14. Unreal Engine mapping

The model transfers directly; only the plumbing changes.

| Component | Isaac Sim | Unreal Engine |
|---|---|---|
| Temperature transport | encoded-T AOV | custom G-buffer / Custom Data channel, or a Scene Capture with a T-only material |
| Band radiance | SPG CUDA kernel | post-process material with LUT texture (float32 `PF_R32_FLOAT`) |
| Emissivity table | `cuda.array` | material parameter collection / texture atlas |
| Sky-view factor | AO AOV | DFAO or a baked sky-visibility channel |
| Atmosphere | SPG stage | post-process using SceneDepth |
| Noise + NUC | SPG stage | post-process with a noise texture + persistent render target |
| Bolometer time constant | frame IIR in SPG | persistent RT ping-pong in post-process |
| Thermal solver | Python tick | Blueprint/C++ actor component on a fixed tick |

Where your existing Unreal build already matches this document:

- **LUT-based Planck** — correct, keep it. Just confirm the LUT texture format is float32, not fp16 (§3.2).
- **Kirchhoff for environment reflection** — correct in principle. Upgrade path: add the angular term (§4.2 Level B is two instructions) and per-band roughness (§4.3).
- **AGC windowing, palette, noise, polarity in a post-process pass** — correct architecture; §10.2 and §11 tell you what to put inside it. Replacing a single-sigma noise term with the 3D decomposition is the highest-value change you can make.
- **Newton's law of cooling in a Blueprint component** — right solver, wrong scope. Keep it for scripted actors, add the §6.1 energy balance as a second solver behind the same interface (§6.5).

The one thing that does **not** transfer: Unreal's post-process stack works in fp16 by default. Force float32 render targets for the temperature and radiance buffers or you will silently lose the sensor's entire dynamic sensitivity.

---

<a name="15-validation"></a>
## 15. Validation protocol

A simulator you haven't validated is a renderer with physics-flavoured variable names. Five tiers, in order.

### Tier 1 — Unit tests (no scene, no renderer)

| Test | Pass criterion |
|---|---|
| Planck vs. photon form | $B_q\cdot hc/\lambda = B$ to 1e-12 relative |
| Band integral vs. Stefan-Boltzmann | $\int_0^\infty B\,d\lambda = \sigma T^4/\pi$ to 1e-6 |
| LUT vs. direct quadrature | max error < 5 mK equivalent over 200–1000 K |
| $L_B^{-1}(L_B(T)) = T$ | < 1 mK round trip |
| $\partial L/\partial T$ vs. finite difference | < 1e-5 relative |
| Kirchhoff closure | $\varepsilon+\rho+\tau = 1$ for every material, every band |
| Energy balance equilibrium | steady-state $T_s$ analytic vs. numeric, < 0.1 K |
| Temperature encode/decode round trip | < 10 mK |

### Tier 2 — Radiometric bench (synthetic blackbody)

Reproduce standard lab characterisation *inside the simulator*, then against the real camera:

- **SITF** (signal intensity transfer function): image simulated blackbodies at 10 temperatures spanning the range; DN vs. $T$ must be smooth and match the real camera's measured curve. SITF, 3D noise, IFOV and MTF together form the measured-system description used in NV-IPM [R29].
- **NETD**: image two blackbodies differing by a few K; $\mathrm{NETD}=\Delta T\cdot\sigma_{\text{noise}}/\Delta \mathrm{DN}$. Must match datasheet within 10%. Check that it **falls** with scene temperature (§3.4) — if it's flat, your noise is in the wrong domain.
- **3D noise**: image a uniform blackbody for 100+ frames, decompose into $\sigma_{TVH},\sigma_{VH},\sigma_V,\sigma_H$ [R28]. Compare component-by-component with the real camera.
- **MTF**: slant-edge target. A practical setup for a real reference measurement: a Boson-class core viewing an aluminium sheet at ~5.5° tilt in front of a hotplate at 100 °C, with a cardboard enclosure to block stray flux and prevent the camera's own signature reflecting back [R37]. Reproduce the same geometry in sim.

### Tier 3 — Phenomenology (does it behave like the world?)

Qualitative but decisive. Each of these should emerge without being scripted:

- [ ] Diurnal cycle runs; thermal crossover (contrast collapse) appears near dawn and dusk
- [ ] Overcast sky flattens the image; clear night sky darkens vehicle roofs and glass
- [ ] Wet asphalt reads colder than dry; shaded ground reads colder than sunlit
- [ ] A departed vehicle leaves a warm tyre trace and a cool body shadow
- [ ] Fog kills visible and SWIR before LWIR — and MWIR and SWIR no later than the visible (§7.2, §7.7)
- [ ] Rain degrades every band alike: no LWIR advantage in rain (§7.8)
- [ ] From above, a radiation fog reads at its top's temperature, colder than the ground air, with the
      ground seen through the layer's depth rather than the camera's range
- [ ] Falling snow is visible as flakes to an uncooled LWIR camera; falling rain is not (§7.8)
- [ ] A wet road mirrors the cold sky at grazing angles; it dries from the sunlit side first (§4.6)
- [ ] A moving cumulus shadow leaves a cool footprint that lags the shadow (§6.7)
- [ ] Clouds have rough outlines, sharp edges and no sampling static at the horizon, in both bands (§7.5)
- [ ] Humid clear air degrades LWIR more than SWIR
- [ ] MWIR shows solar glint at midday and looks like LWIR at night
- [ ] SWIR at night is usable from airglow alone, without any modelled light source
- [ ] A hot exhaust entering frame collapses global AGC contrast
- [ ] FFC event freezes the image and resets the fixed pattern
- [ ] Fast lateral motion smears edges in LWIR (bolometer $\tau_{\text{th}}$) but not in cooled MWIR

### Tier 4 — Comparison against real data

Public reference points, both suggested by practitioners on the Isaac Sim feature request [R34]:

- **FLIR ADK** (Boson 640-based, designed for automotive) — used in the *Novel Sensors for Autonomous Driving* dataset (Carmichael et al., 2024).
- **FLIR Hadron 640** — deployed on small UAS.

Method: recreate a dataset scene's geometry, materials, weather and time-of-day, render, and compare
(a) **apparent-temperature histograms** by semantic class — the shape and separation matter more than absolute offsets;
(b) **radiometric contrast** between labelled object classes and their local background;
(c) **noise power spectral density** of a flat region — the single most diagnostic comparison, because it exposes wrong 3D-noise ratios instantly;
(d) **edge spread function** on real edges.

Acceptance targets for L2: apparent-temperature bias < 2 K per class, contrast ratio within 25%, noise PSD shape matching within a factor of 2 across spatial frequency.

### Tier 5 — Task-level (does it transfer?)

Train a detector on synthetic only, evaluate on real. Then train real-only and evaluate on synthetic. Measure both gaps.

Three cautions worth stating plainly, because they cost people months:

1. **Low-level statistics dominate transfer, not radiometry.** Networks overfit to noise structure, NUC residual pattern and AGC behaviour far more than to being 3 K off on a car door. If Tier 4(c) fails, no amount of Tier 2 accuracy saves you.
2. **AGC must match at train and test time.** Same algorithm, same plateau, same palette. A model trained on linear-AGC synthetic data and deployed on plateau-EQ real data will underperform for reasons that look like a physics problem and are not.
3. **Mixed training beats synthetic-only** in essentially every published result. Plan for synthetic as augmentation and rare-case coverage — night, fog, occluded pedestrians, thermal crossover — rather than as a replacement.

---

<a name="16-parameters"></a>
## 16. Reference parameter tables

### 16.1 Reference camera — FLIR Boson 640 (LWIR)

A good default target: it is the sensor in the FLIR ADK automotive module and appears in public robotics datasets [R34].

| Parameter | Value | Source |
|---|---|---|
| Detector | VOx uncooled microbolometer | [R24] |
| Resolution | 640×512 (also 320×256) | [R24] |
| Pitch | 12 µm | [R24] |
| NETD | <60 mK (consumer) / <50 mK (professional) / <40 mK (industrial) | [R38] |
| Spectral band | ~7.5–13.5 µm | typical for VOx cores |
| Frame rate | 60 Hz (9 Hz export variant) | [R38] |
| Example lens | 14 mm, 32° HFOV | [R38] |
| Operating temperature | −40 °C to +80 °C | [R38] |
| Thermal time constant | ~8–12 ms | typical VOx |
| TCR | ≈ −2 %/K | typical VOx |

Note the NETD grading: the *same* detector is binned into three products. Model NETD as a config value, not a constant, and you can simulate all three by editing one line.

### 16.2 Material properties (starting values — replace with measurements)

| Material | $\varepsilon_{\text{LWIR}}$ | $\varepsilon_{\text{MWIR}}$ | $\alpha_{\text{sol}}$ | $\rho$ (kg/m³) | $c_p$ (J/kg·K) | $k$ (W/m·K) |
|---|---|---|---|---|---|---|
| Asphalt (dry) | 0.94 | 0.92 | 0.90 | 2200 | 920 | 0.75 |
| Concrete | 0.92 | 0.90 | 0.65 | 2300 | 880 | 1.4 |
| Car paint (black) | 0.90 | 0.88 | 0.94 | 7800 | 470 | 45 |
| Car paint (white) | 0.90 | 0.88 | 0.28 | 7800 | 470 | 45 |
| Bare aluminium | 0.09 | 0.06 | 0.15 | 2700 | 900 | 205 |
| Rusted steel | 0.85 | 0.82 | 0.80 | 7800 | 470 | 45 |
| Glass (windshield) | 0.88 | 0.85 | 0.10 | 2500 | 840 | 1.0 |
| Rubber (tyre) | 0.95 | 0.94 | 0.94 | 1100 | 2000 | 0.16 |
| Human skin | 0.98 | 0.97 | 0.65 | 1050 | 3500 | 0.37 |
| Cotton clothing | 0.95 | 0.93 | 0.70 | 300 | 1300 | 0.06 |
| Vegetation (leaf) | 0.97 | 0.96 | 0.50 | 700 | 3000 | 0.30 |
| Soil (dry) | 0.92 | 0.90 | 0.75 | 1500 | 800 | 0.30 |
| Soil (wet) | 0.96 | 0.95 | 0.85 | 2000 | 1500 | 1.5 |
| Water | 0.96 | 0.98 | 0.93 | 1000 | 4180 | 0.60 |
| Snow | 0.99 | 0.98 | 0.15 | 300 | 2100 | 0.20 |

Glass and bare metal are the two rows most likely to break naïve simulators — glass because of the band transition (§4.4), aluminium because $\varepsilon=0.09$ means it is a mirror, not a surface.

**These are surface states, unwritten.** Each row is one surface condition of the named substance and the table does not say which — see §4.5. "Bare aluminium" at $\varepsilon=0.09$ is an *oxidised* surface: polished aluminium is 0.04 and anodised 0.834–0.856 [R39], a span this single row cannot represent. The two paint rows carrying equal $\varepsilon$ and differing only in $\alpha_{\text{sol}}$ is correct and deliberate (§4.5a), not an oversight. Treat the table as starting values for one unstated finish each, and replace them from a spectral library that covers all four bands [R41][R42] rather than by editing numbers in place.

### 16.3 Physical constants

| Constant | Value |
|---|---|
| $h$ | 6.626 070 15 × 10⁻³⁴ J·s |
| $c$ | 2.997 924 58 × 10⁸ m/s |
| $k_B$ | 1.380 649 × 10⁻²³ J/K |
| $\sigma$ | 5.670 374 419 × 10⁻⁸ W·m⁻²·K⁻⁴ |
| $c_{1L}=2hc^2$ | 1.191 042 97 × 10⁸ W·µm⁴·m⁻²·sr⁻¹ |
| $c_{1q}=2c$ | 5.995 849 × 10²⁶ ph·s⁻¹·µm³·m⁻²·sr⁻¹ |
| $c_2=hc/k_B$ | 1.438 776 9 × 10⁴ µm·K |
| Wien | $\lambda_{\max}T=2897.77$ µm·K |
| $q$ | 1.602 176 634 × 10⁻¹⁹ C |

### 16.4 Suggested build order

1. Planck LUT + band integral + inversion, with Tier 1 unit tests. *(No renderer.)*
2. Temperature encode/decode round trip through an Isaac Sim AOV. Verify < 10 mK. **Stop and fix if this fails.**
3. Constant-$\varepsilon$, no-reflection, no-atmosphere radiance kernel → 16-bit linear output. Simulate a blackbody; verify SITF.
4. Detector + 3D noise + NETD calibration. Tier 2 tests.
5. AGC + palette + display output. First image that looks like a thermal camera.
6. Energy-balance thermal solver with weather and 24 h spin-up. Tier 3 diurnal tests.
7. Angular emissivity + sky-view-factor reflections. Tier 3 reflection tests.
8. Atmosphere. Tier 3 weather tests.
9. Bolometer time constant, NUC/FFC, bad pixels.
10. Second band (MWIR is the hardest — do SWIR first to prove the reflective path).
11. Tier 4 comparison against real data.

Steps 1–5 give a defensible LWIR camera. Steps 6–9 are what separate it from a post-process filter. Step 10 validates the architecture.

---

<a name="17-sources"></a>
## 17. Sources

**Rendering, radiometry, materials**

- [R1] ter Heerdt, Keustermans, De Boi, Vanlanduit, *A Unified Complex-Fresnel Model for Physically Based Long-Wave Infrared Imaging and Simulation*, J. Imaging 12(1):33, Jan 2026. https://doi.org/10.3390/jimaging12010033 — complex-IOR Fresnel with branch-safe $A/B$ reformulation; Kirchhoff emissivity; LWIR validation against a heated K9 sphere. **Most directly useful single paper for your material model.**
- [R2] Willers et al., *Signature Modelling and Radiometric Rendering Equations in Infrared Scene Simulation Systems*, SPIE. https://www.researchgate.net/publication/253464716 — compares OSSIM/OSMOSIS and DIRSIG rendering equations; the "one spectral scene, many bandpasses" architecture.
- [R39] Gustavsen & Berdahl, *Spectral emissivity of anodized aluminum and the thermal transmittance of aluminum window frames*, LBNL. https://www.osti.gov/biblio/835335 — normal spectral emissivity measured 4.5–40 µm: anodised 0.834–0.856, untreated all-aluminium cavities 0.055–0.82, polished 0.04 at 8 µm. **The measurement behind §4.5.**
- [R40] *A Study on the Infrared Radiation Properties of Anodized Aluminum*, J. Korean Inst. Surface Engineering. https://koreascience.kr/article/JAKO200211921391533.page — $\varepsilon$ against anodic film thickness: 0.15 at 1 µm, 0.45 at 2 µm, 0.91 above ~15 µm.
- [R47] *Spectral emissivity of oxidized and roughened metal surfaces*, Int. J. Heat and Mass Transfer. https://www.sciencedirect.com/science/article/abs/pii/S0017931017325802 — roughness and oxidation both raise $\varepsilon$, oxidation dominating.
- [R41] Meerdink, Hook, Roberts & Abbott, *The ECOSTRESS spectral library version 1.0*, Remote Sensing of Environment, 2019. https://www.sciencedirect.com/science/article/abs/pii/S0034425719302081 — 3400+ spectra over **0.35–15.4 µm**, i.e. all four bands in one curve; ~72 man-made entries; free at https://speclib.jpl.nasa.gov. Reflectance in percent, wavelength in µm.
- [R42] MODIS UCSB Emissivity Library. https://icess.eri.ucsb.edu/modis/EMIS/html/em.html — 123 laboratory emissivity spectra, 3–14 µm; emissivity directly rather than reflectance.
- [R20] SPIE Optipedia, *Detector Footprint Modulation Transfer Function*. https://spie.org/publications/spie-publication-resources/optipedia-free-optics-information/tt52_21_detector_footprint_mtf

**DIRSIG / thermal modelling**

- [R3] RIT DIRSIG, *Thermal Modality Handbook*. https://dirsig.cis.rit.edu/docs/new/thermal.html — heat transfer basics, thermodynamic property set, DHE = 1 − DHR, temperature model options.
- [R6] RIT DIRSIG, *Temperature Solvers*. https://dirsig.cis.rit.edu/docs/new/temp_solvers.html — THERM 1-D slab formulation and spin-up.
- [R4] ThermoAnalytics, *Dynamic EO/IR Satellite Signature Prediction with High-Fidelity Thermal Simulation*. https://blog.thermoanalytics.com/blog/dynamic-electro-optical/infrared-satellite-signature-prediction-with-high-fidelity-thermal-simulation
- [R5] DeMars et al., *High-Fidelity Simulation of Dynamic Thermal Satellite Signatures with MuSES*, AMOS 2023. https://amostech.com/TechnicalPapers/2023/Poster/Demars.pdf
- [R14] *Agentic AI-Enabled Framework for Thermal Comfort and Building Energy Assessment*, arXiv. https://arxiv.org/pdf/2604.21787 — transient surface energy balance form.
- [R15] *Directional Characteristics of Thermal-Infrared Beaming from Atmosphereless Planetary Surfaces*, arXiv. https://arxiv.org/pdf/1211.1844 — facet energy balance with shadow and illumination-cosine terms.

**Atmosphere**

- [R11] *Absorption-Feature-Guided Distance-Decoupled Estimation for LWIR Hyperspectral Passive Ranging*, arXiv. https://arxiv.org/pdf/2606.31824 — the three-term radiance model and $\tau(\lambda;d)=[\tau_{1\mathrm{m}}]^d$.
- [R12] Spectral Sciences, *Modeling and Analysis of LWIR Signature Variability*. https://www.spectral.com/wp-content/uploads/2019/08/Modeling_and_Analysis.pdf
- [R13] *Refining Atmosphere Profiles for Aerial Target Detection Models*, Sensors 21:7067. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8588161/ — sky apparent temperature vs. elevation, LWIR vs. MWIR.
- [R16] MODTRAN model description (US patent 5,315,513). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5315513
- [R17] *Effects of climate change on EO/IR propagation using CMIP6*, PubMed 40281099. https://pubmed.ncbi.nlm.nih.gov/40281099/ — representative MODTRAN run configuration for ground-level horizontal paths.
- [R18] Berk et al., *MODTRAN4 Radiative Transfer Modeling for Atmospheric Correction*. https://www.spectral.com/wp-content/uploads/2017/04/MODTRAN4_Radiative_Transfer.pdf

**Night illumination (SWIR/NIR)**

- [R7] *The Night Glows Brighter in the Near-IR*, Photonics Spectra, 2012. https://www.photonics.com/Article.aspx?AID=50540 — airglow spectrum peaks 1–1.8 µm; comparable to moonlight.
- [R8] *Shortwave infrared for night vision applications: illumination levels and sensor performance*, SPIE 9641. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9641/1/10.1117/12.2193738.short
- [R10] *Passive SWIR airglow illuminated imaging compared with NIR-visible*, ResearchGate 253463654 — reported airglow irradiance range 3.5–39 nW/cm².

**Detector, noise, performance**

- [R9] Optris, *NETD – Thermal Sensitivity*. https://optris.com/us/knowledge-library/thermal-sensitivity/ — NETD vs. scene temperature, measured.
- [R21] Optris, *Modulation Transfer Function in Infrared Imaging*. https://optris.com/lexicon/modulation-transfer-function-mtf/
- [R22] Cascade MTF model for sampled IR imaging systems — see Science.gov IR imaging topic index. https://www.science.gov/topicpages/i/ir+imaging+system.html
- [R25] *Methodological enhancement of NETD optimization through active FPA temperature control*, SPIE 14036, 2026. https://doi.org/10.1117/12.3094960
- [R26] Thermopile IR sensor patent (US 6,335,478) — the $\mathrm{NETD}=4F^2V_n/(\mathcal{R}A_dL')$ form. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6335478
- [R27] *A modified and systematic approach to NETD derivation for infrared focal plane arrays*, Infrared Physics 30(1):71, 1990. https://www.sciencedirect.com/science/article/abs/pii/002008919090043U
- [R28] 3-D noise nomenclature ($\sigma_{TVH},\sigma_{VH},\sigma_V,\sigma_H$) as used in scene-based NUC (US 10,621,702). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10621702
- [R29] *Infrared Imaging Systems: Design, Analysis, Modeling and Testing XXIX*, SPIE 10625 — SITF + 3D noise + IFOV + MTF feeding NV-IPM. https://spie.org/Publications/Proceedings/Volume/10625
- [R30] *Exploring Video Denoising in Thermal Infrared Imaging: Physics-Inspired Noise Generator*, ResearchGate 380032672 — temporal stripe noise modelling.
- [R31] *Infrared Focal Plane Array Characterization by Means of a Blackbody Radiator*, ResearchGate 220843421 — NETD and correctability via two-point calibration.
- [R32] Pust, *Radiometric calibration of infrared imagers using an internal shutter as an equivalent external blackbody*.
- [R43] FLIR, *The Ultimate Infrared Handbook for R&D Professionals*. http://www.flirmedia.com/MMC/THG/Brochures/T559243/T559243_EN.pdf — the measurement equation §11.5 inverts.
- [R44] *Quantifying Within-Flight Variation in Land Surface Temperature from a UAV-Based Thermal Infrared Camera*, Drones 7(10):617, 2023. https://doi.org/10.3390/drones7100617 — bias −1.02 → +3.86 °C over 0.8–8.5 m s⁻¹ of wind, against calibrated Apogee SI-111 ground truth. **The anchor for §9.5.**
- [R45] *A Case Study of Vignetting Nonuniformity in UAV-Based Uncooled Thermal Cameras*, Drones 6(12):394, 2022. https://doi.org/10.3390/drones6120394 — wind lowers image mean ~5.5 °C and deepens vignetting; 20–40 min warm-up to stability.
- [R46] *Removing temperature drift and temporal variation in thermal infrared images of a UAV uncooled thermal infrared imager*, ISPRS J. Photogrammetry and Remote Sensing, 2023. https://www.sciencedirect.com/science/article/abs/pii/S0924271623002265 — measured temperature correlated with FPA temperature.
- [R33] Ness, Oved, Kakon (RAFAEL), *Derivative Based Focal Plane Array Nonuniformity Correction*, arXiv. https://arxiv.org/pdf/1702.06118
- [R19] Low-radiance IR airborne calibration reference (US 9,234,796) — worked optics self-emission decomposition. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9234796
- [R37] *Resonant Anti-Reflection Metasurface for Infrared Transmission Optics*, arXiv 2306.05405 — practical Boson slant-edge MTF setup. https://arxiv.org/pdf/2306.05405

**Isaac Sim**

- [R34] *Infrared Camera Support*, isaac-sim/IsaacSim Discussion #298. https://github.com/isaac-sim/IsaacSim/discussions/298 — NVIDIA's recommended SPG approach; FLIR ADK / Hadron as reference hardware.
- [R35] NVIDIA, *RTX Sensor Processing Graphs [omni.rtx.spg]*. https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/latest/Overview.html — CUDA/Lua/USD structure, `cuda.static`, stdlib nodes, limitations.
- [R36] NVIDIA, *isaacsim.sensors.experimental.rtx* and *Camera Sensors*. https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html
- [R23] NVIDIA, camera distortion models (OpenCV pinhole/fisheye, f-theta, Kannala-Brandt). https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_camera_depth_sensors.html

**Hardware reference**

- [R24] FLIR, *Boson Thermal Imaging Core* datasheet. https://groupgets-files.s3.amazonaws.com/boson/documents/Boson%20datasheet,%20102-2013-40,%20Rev%20340.pdf
- [R38] Teledyne FLIR Boson 640 product listings (NETD grades, lens options, frame rates). https://www.oemcameras.com/products/20640a032-htm
- [R48] H. Budzier, G. Gerlach, *Calibration of uncooled thermal infrared cameras*, J. Sens. Sens. Syst. 4, 187–197, 2015. https://jsss.copernicus.org/articles/4/187/2015/ — radiometric camera model of a microbolometer core: the detector signal is the scene radiance relative to the housing, with the lens, housing and shutter as radiance terms.
- [R49] C. Tempelhahn, H. Budzier, V. Krause, G. Gerlach, *Shutter-less calibration of uncooled infrared cameras*, J. Sens. Sens. Syst. 5, 9–16, 2016. https://jsss.copernicus.org/articles/5/9/2016/ — the housing and shutter radiation as explicit, temperature-measured terms; the field-dependent share of the housing seen by each pixel.
- [R50] P. W. Nugent, J. A. Shaw, N. J. Pust, *Correcting for focal-plane-array temperature dependence in microbolometer infrared cameras lacking thermal stabilization*, Opt. Eng. 52(6), 061304, 2013. https://doi.org/10.1117/1.OE.52.6.061304 — responsivity and offset drift as functions of FPA temperature, the multiplicative mechanism distinct from the housing pedestal.
- [R51] FLIR, *FLIR Camera Adjustments — Boson Application Note*, 102-2013-100-01 Rev 220, June 2018. https://tesscorn-thermalimaging.com/wp-content/uploads/2024/08/Boson-CameraAdjustments-AppNote-2.pdf — the Boson's AGC: plateau value as a fraction of the ROI's pixels per bin (default 7 %), Information-Based Equalization as the factory default mode, Linear Percent, Tail Rejection, Max Gain, Damping Factor, DDE and Detail Headroom.
- [R52] FLIR, *Lepton Software Interface Description Document (IDD)*, 110-0144-04. https://cdn.sparkfun.com/assets/0/6/d/2/e/16465-FLIRLepton-SoftwareIDD.pdf — AGC HEQ: clip limit high (bin population cap), clip limit low (constant added to every non-zero bin), linear percent, dampening factor (IIR), ROI.
- [R53] Xenics, *Smart onboard image enhancement algorithms for SWIR day and night vision camera*, 2015. https://www.researchgate.net/publication/283861475_Smart_onboard_image_enhancement_algorithms_for_SWIR_day_and_night_vision_camera — auto-exposure positions the histogram by integration time and switches gain and read-out modes; auto-gain and histogram equalisation follow, with a maximal allowed stretching and equalisation strength.
- [R54] University of Wisconsin lidar group, *Visible vs. Infrared Optical Depths* (thesis chapter), citing Platt et al. 1980, Minnis et al. 1990 and 1993. http://lidar.ssec.wisc.edu/papers/ww_thes/node17.htm — ratio of visible extinction to 10.6 µm absorption efficiency, 2:1 in the large-particle limit, 2.13 measured.
- [R55] D. H. DeSlover, W. L. Smith, P. K. Piironen, E. W. Eloranta, *A methodology for measuring cirrus cloud visible-to-infrared spectral optical depth ratios*, J. Atmos. Oceanic Technol. 16, 251–262 (1999) — ratios of 2–3 by cloud type.
- [R56] J. A. Shaw, P. W. Nugent, *Physics principles in radiometric infrared imaging of clouds in the atmosphere*, Eur. J. Phys. 34, S111–S121 (2013). https://www.montana.edu/jshaw/documents/Physics_IRCloudImaging_EJP2013.pdf — LWIR optical depth half the visible; ε = 1 − exp(−0.79 τ); opaque at 0.1 mm of liquid.
- [R57] J.-L. Brenguier et al., *Cloud optical thickness and liquid water path — does the k coefficient vary with droplet concentration?*, Atmos. Chem. Phys. 11, 9771–9786 (2011) — τ = 3 LWP / (2 ρ_w r_e).
- [R58] G. L. Stephens, *Radiation profiles in extended water clouds. II: Parameterization schemes*, J. Atmos. Sci. 35, 2123–2132 (1978); tabulated with later measurements in Stephens, AT622 notes §16, Table 16.1. https://reef.atmos.colostate.edu/~odell/AT622/stephens_notes/AT622_section16.pdf — LWIR mass absorption 0.130 (down) and 0.158 (up) m²/g in flux form.
- [R59] M.-D. Chou, K.-T. Lee, S.-C. Tsay, Q. Fu, *Parameterization for cloud longwave scattering for use in atmospheric models*, J. Climate 12, 159–169 (1999). https://modis-images.gsfc.nasa.gov/_docs/Chou_et_al._(1999).pdf — scaled absorption, errors under 2 % of the flux.
- [R60] G. Hong, P. Yang, B. A. Baum, A. J. Heymsfield, K.-M. Xu, *Parameterization of shortwave and longwave radiative properties of ice clouds for use in climate models*, J. Climate 22, 6287–6312 (2009), Tables A1–A2.
- [R61] G. W. Petty, *A First Course in Atmospheric Radiation*, 2nd ed., Sundog (2006), ch. 8 — the Schwarzschild emission–absorption equation along a path.

**Weather: cloud shape and rendering, fog, haze, rain, snow and wet surfaces** (added 2026-10-01; the
search notes are `docs/research/2026-10-01-weather-in-the-infrared.md`)

- [R62] S. Lovejoy, *Area–perimeter relation for rain and cloud areas*, Science 216, 185–187 (1982). doi:10.1126/science.216.4542.185 — D = 1.35 over 1–1.2 × 10⁶ km².
- [R63] G. Zhao, L. Di Girolamo, *Statistics on the macrophysical properties of trade wind cumuli over the tropical western Atlantic*, J. Geophys. Res. 112, D10204 (2007). doi:10.1029/2006JD007371 — D = 1.28 and size exponent 2.19 from 15 m ASTER.
- [R64] R. A. J. Neggers, H. J. J. Jonker, A. P. Siebesma, *Size statistics of cumulus cloud populations in large-eddy simulations*, J. Atmos. Sci. 60, 1060–1074 (2003). doi:10.1175/1520-0469(2003)60<1060:SSOCCP>2.0.CO;2 — b = 1.70 below a scale break.
- [R65] R. Wood, P. R. Field, *The distribution of cloud horizontal sizes*, J. Climate 24, 4800–4816 (2011). doi:10.1175/2011JCLI4056.1 — chord exponent 1.66 ± 0.04.
- [R66] J. T. Dawe, P. H. Austin, *Statistical analysis of an LES shallow cumulus cloud ensemble using a cloud tracking algorithm*, Atmos. Chem. Phys. 12, 1101–1119 (2012). doi:10.5194/acp-12-1101-2012 — 1.88–1.96, break near 1 km.
- [R67] H. E. Gerber, G. M. Frick, J. B. Jensen, J. G. Hudson, *Entrainment, mixing, and microphysics in trade-wind cumulus*, J. Meteor. Soc. Japan 86A, 87–106 (2008). doi:10.2151/jmsj.86A.87 — LWC steps over ~30 cm at 10 cm resolution.
- [R68] Y. Wang, B. Geerts, J. French, *Dynamics of the cumulus cloud margin: an observational study*, J. Atmos. Sci. 66, 3660–3677 (2009). doi:10.1175/2009JAS3129.1 — decline over the outer ~10 % (1624 passes).
- [R69] V. P. Ghate, M. A. Miller, P. Zhu, *Differences between nonprecipitating tropical and trade wind marine shallow cumuli*, Mon. Wea. Rev. 144, 681–701 (2016). doi:10.1175/MWR-D-15-0110.1 — depth/chord 0.9 (Manus), 0.4 (Azores).
- [R70] M. Hess, P. Koepke, I. Schult, *Optical properties of aerosols and clouds: the software package OPAC*, Bull. Amer. Meteor. Soc. 79, 831–844 (1998) — cumulus visible extinction 0.05–0.12 m⁻¹; haze scale heights.
- [R71] R. Wood, *Stratocumulus clouds*, Mon. Wea. Rev. 140, 2373–2423 (2012). doi:10.1175/MWR-D-11-00121.1 — longwave liquid-water absorption 100–160 m² kg⁻¹.
- [R72] A. Schneider, N. Vos, *The real-time volumetric cloudscapes of Horizon: Zero Dawn*, SIGGRAPH 2015 Advances in Real-Time Rendering. https://advances.realtimerendering.com/s2015/ — envelope, Perlin–Worley base, Worley detail, remap erosion.
- [R73] A. Schneider, *Nubis, Evolved: real-time volumetric clouds for skies, environments, and VFX*, SIGGRAPH 2022 Advances. https://www.guerrilla-games.com/read/nubis-evolved
- [R74] A. Schneider, *Nubis³: methods (and madness) to model and render immersive real-time voxel-based clouds*, SIGGRAPH 2023 Advances. https://www.guerrilla-games.com/read/nubis-cubed — 0.5 m effective precision from 8 m voxels.
- [R75] W. I. Gustafson Jr. et al., *The Large-Eddy Simulation (LES) Atmospheric Radiation Measurement (ARM) Symbiotic Simulation and Observation (LASSO) activity for continental shallow convection*, Bull. Amer. Meteor. Soc. 101, E462–E479 (2020). doi:10.1175/BAMS-D-19-0065.1; data doi:10.5439/1342961 (CC BY 4.0) — WRF-LES, 100 m horizontal, 30 m vertical below 5 km, 25 km periodic domain, QCLOUD every 10 min. https://adc.arm.gov/lassobrowser
- [R76] N. Villefranque et al., *A path-tracing Monte Carlo library for 3-D radiative transfer in highly resolved cloudy atmospheres*, J. Adv. Model. Earth Syst. 11 (2019), arXiv:1902.01137; htrdr Atmosphere Starter Pack (GPLv3+). https://www.meso-star.com/projects/htrdr/htrdr-atmosphere-spk.html — Meso-NH SGP cumulus at 25 m isotropic, liquid water and temperature in K.
- [R77] Walt Disney Animation Studios, *Cloud Data Set* (2017), CC BY-SA 3.0. https://www.disneyanimation.com/resources/clouds/ — 1987 × 1351 × 2449 voxels and four lower resolutions; density in arbitrary units.
- [R78] S. Hillaire, *Physically based sky, atmosphere and cloud rendering in Frostbite*, SIGGRAPH 2016 Physically Based Shading course notes. https://sebh.github.io/publications/ — sky ambient in clouds, dual-lobe phase, multiple-scattering octaves.
- [R79] S. Hillaire, *Physically-based & unified volumetric rendering in Frostbite*, SIGGRAPH 2015 Advances. https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite — energy-conserving step integration.
- [R80] T. Porter, T. Duff, *Compositing digital images*, Computer Graphics 18(3), 253–259 (SIGGRAPH 1984) — associated (premultiplied) colour.
- [R81] M. Wrenninge, *Art-directable multiple volumetric scattering*, SIGGRAPH 2015 Talks. doi:10.1145/2775280.2792512 — thick media need upwards of 100 scattering orders.
- [R82] M. Wrenninge, C. Kulla, V. Lundqvist, *Oz: the great and volumetric*, SIGGRAPH 2013 Talks, art. 46 — octaves with a = b = c = ½.
- [R83] NVIDIA, *RTX path tracing mode*, Omniverse materials and rendering documentation. https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html — `ptvol/*` settings (`maxBounces` default 2, `maxCollisionCount` 1024), VDB materials on a cube mesh; volumes do not emit.
- [R84] E. P. Shettle, R. W. Fenn, *Models for the aerosols of the lower atmosphere and the effects of humidity variations on their optical properties*, AFGL-TR-79-0214 (1979) — advection and radiation fog size distributions.
- [R85] R. G. Pinnick et al., *Vertical structure in atmospheric fog and haze and its effects on visible and infrared extinction*, J. Atmos. Sci. 35, 2020–2032 (1978).
- [R86] R. G. Pinnick, S. G. Jennings, P. Chylek, H. J. Auvermann, *Verification of a linear relation between IR extinction, absorption and liquid water content of fogs*, J. Atmos. Sci. 36, 1577–1586 (1979).
- [R87] C. Klein, A. Dabas, *Relationship between optical extinction and liquid water content in fogs*, Atmos. Meas. Tech. 7, 1277–1287 (2014). doi:10.5194/amt-7-1277-2014 — c_e(11 µm) = 0.31.
- [R88] A. Breton et al., *Free-space optical transmission measurements from 0.532 to 10 µm in real controlled fog*, Opt. Lett. 51(16), 4729–4732 (2026). doi:10.1364/OL.609352 — 4 µm no better than 1.55 µm; 10 µm > 20 dB in continental fog.
- [R89] K. Nurowska, P. Makuch, K. M. Markowicz, *Measurement report: microphysical and optical characteristics of radiation fog*, Atmos. Chem. Phys. 25, 13493–13525 (2025). doi:10.5194/acp-25-13493-2025 — tops 83–115 m; r_eff 8–10 µm near the surface; LWC peak near 80 % of depth.
- [R90] E. G. Wærsted et al., *Radiation in fog: quantification of the impact on fog liquid water based on ground-based remote sensing*, Atmos. Chem. Phys. 17, 10811–10835 (2017). doi:10.5194/acp-17-10811-2017
- [R91] WMO, *Guide to Instruments and Methods of Observation*, WMO-No. 8, Vol. I ch. 9 — MOR at 5 % of a collimated beam; quoted in S. Liandrat et al., *A review of Cerema PAVIN fog & rain platform*, ITS World Congress 2022.
- [R92] WMO, *Aerodrome Reports and Forecasts: a Users' Handbook to the Codes*, WMO-No. 782 (2019) — fog, mist and haze thresholds.
- [R93] D. Lin, M. Katurji, L. E. Revell, B. Khan, A. Sturman, *Investigating multiscale meteorological controls and impact of soil moisture heterogeneity on radiation fog in complex terrain*, Atmos. Chem. Phys. 23, 14451–14479 (2023). doi:10.5194/acp-23-14451-2023
- [R94] F. X. Kneizys et al., *Users guide to LOWTRAN 7*, AFGL-TR-88-0177 (1988); Fortran source (TNRAIN, RNSCAT) at https://github.com/space-physics/lowtran — Marshall–Palmer rain at Q = 2, absorption share 0.5, rain at air temperature; boundary-layer haze profiles.
- [R95] T. S. Chu, D. C. Hogg, *Effects of precipitation on propagation at 0.63, 3.5, and 10.6 microns*, Bell Syst. Tech. J. 47, 723–759 (1968).
- [R96] J. S. Marshall, W. McK. Palmer, *The distribution of raindrops with size*, J. Meteor. 5, 165–166 (1948).
- [R97] FLIR Systems, *Seeing through fog and rain with a thermal imaging camera*, technical note TN_0001. http://www.flirmedia.com/MMC/CVS/Tech_Notes/TN_0001_EN.pdf — detection ranges by ICAO visibility category.
- [R98] ITU-R Recommendations P.1814-1 (2025) and P.1817-1 (2012), propagation data and prediction methods for terrestrial free-space optical links — rain and snow attenuation laws; rain-rate visibility code.
- [R99] M. Al Naboulsi, H. Sizun, F. de Fornel, *Propagation of optical and infrared waves in the atmosphere*, URSI General Assembly 2005, F01P.7 — Carbonneau's rain law and the dry/wet snow laws.
- [R100] D. J. Segelstein, *The complex refractive index of water*, M.S. thesis, Univ. Missouri–Kansas City (1981); G. M. Hale, M. R. Querry, Appl. Opt. 12, 555–563 (1973) — k = 0.0508 at 10 µm.
- [R101] D. Sadot, N. S. Kopeika, *Imaging through the atmosphere: practical instrumentation-based theory and verification of aerosol modulation transfer function*, J. Opt. Soc. Am. A 10, 172–179 (1993); R. F. Lutomirski, Appl. Opt. 17, 3915–3921 (1978).
- [R102] K. Loftus, R. D. Wordsworth, *The physics of falling raindrops in diverse planetary atmospheres*, J. Geophys. Res. Planets 126, e2020JE006653 (2021); after G. D. Kinzer, R. Gunn, J. Meteor. 8, 71–83 (1951) — drops at the wet bulb.
- [R103] S. Zhang, C. Meurey, J.-C. Calvet, *Identification of soil-cooling rains in southern France from soil temperature and soil moisture observations*, Atmos. Chem. Phys. 19, 5005–5020 (2019). doi:10.5194/acp-19-5005-2019
- [R104] M. A. Seagraves, *Visible and infrared extinction in falling snow*, Appl. Opt. 25, 1166–1169 (1986). doi:10.1364/AO.25.001166
- [R105] R. M. Rasmussen et al., *The estimation of snowfall rate using visibility*, J. Appl. Meteor. 38, 1542–1563 (1999).
- [R106] S. G. Warren, *Optical properties of ice and snow*, Phil. Trans. R. Soc. A 377, 20180161 (2019). doi:10.1098/rsta.2018.0161; after M. Hori et al., Remote Sens. Environ. 100, 486–502 (2006).
- [R107] K. Garg, S. K. Nayar, *Vision and rain*, Int. J. Comput. Vis. 75, 3–27 (2007). doi:10.1007/s11263-006-0028-6
- [R108] B. Oswald-Tranta, *Temperature reconstruction of infrared images with motion deblurring*, J. Sens. Sens. Syst. 7, 13–20 (2018). doi:10.5194/jsss-7-13-2018
- [R109] D. Atlas, R. C. Srivastava, R. S. Sekhon, *Doppler radar characteristics of precipitation at vertical incidence*, Rev. Geophys. 11, 1–35 (1973) — fit to Gunn & Kinzer (1949).
- [R110] J. D. Locatelli, P. V. Hobbs, *Fall speeds and masses of solid precipitation particles*, J. Geophys. Res. 79, 2185–2197 (1974).
- [R111] J. Lekner, M. C. Dorf, *Why some things are darker when wet*, Appl. Opt. 27, 1278–1280 (1988).
- [R112] X. Zhong et al., *Investigating the effects of surface moisture content on thermal infrared emissivity of urban underlying surfaces*, Constr. Build. Mater. 327, 127023 (2022). doi:10.1016/j.conbuildmat.2022.127023
- [R113] A. Rankin et al., *Unmanned ground vehicle perception using thermal infrared cameras*, Proc. SPIE 8045 (2011). https://robotics.jpl.nasa.gov/media/documents/spie-2011-rankin-final.pdf — puddles reflect the sky in LWIR.
- [R114] F. Ritter, M. Berkelhammer, D. Beysens, *Dew frequency across the US from a network of in situ radiometers*, Hydrol. Earth Syst. Sci. 23, 1179–1197 (2019). doi:10.5194/hess-23-1179-2019
- [R115] G. Jobert et al., *Windshield integration of thermal and color fusion for automatic emergency braking in low visibility conditions*, arXiv:2410.04928 (2024) — a drop on the window lowers LWIR responsivity to ~90 % without blur.
- [R116] S. Hillaire, *A scalable and production ready sky and atmosphere rendering technique*, Computer Graphics Forum 39(4), 13–22 (EGSR 2020). doi:10.1111/cgf.14050 — the sky-view table, the multiple-scattering table and the aerial-perspective volume.

**Related open work**

- TCIsaacSim (reza-shahriari) — the thermal-camera Isaac Sim repo you found; same author who opened Discussion #298. Useful as a starting scaffold; note that it predates Isaac Sim 6.0's SPG framework.

---

## Appendix A — Where this model can legitimately be criticised

State these limitations up front in any documentation you write. It is what separates an engineering model from a demo.

1. **No 3-D conduction.** The two-node lumped model cannot represent lateral heat flow, internal components, or objects whose interior differs strongly from their skin. An engine bay is scripted, not solved. MuSES-class fidelity is out of scope [R4][R5].
2. **Band-averaged atmosphere.** Beer-Lambert with band-averaged $\gamma$ ignores spectral fine structure, and is wrong near band edges and for long slant paths. Acceptable under ~500 m; not acceptable for airborne work [R18].
3. **Emissivity is grey within a band.** Real $\varepsilon(\lambda)$ has structure — quartz reststrahlen, vegetation features. Fine for broadband imaging, wrong for anything spectral.
4. **Reflection is approximate** unless you implement §13.7 option 3. Sky-view-factor blending gets the mean right and the specular structure wrong.
5. **No polarisation.** Thermal emission is partially polarised at grazing angles and this is a real discriminant. Out of scope here; DIRSIG supports it if you need it.
6. **No turbulence.** Scintillation and image dancing over long hot paths are unmodelled. Irrelevant under 500 m, significant beyond ~2 km.
7. **NETD is anchored, not predicted.** §9.4 calibrates noise magnitude to a datasheet number rather than deriving it from first principles. That is a deliberate and defensible choice — but it means the model cannot predict the NETD of a detector that doesn't exist yet.
8. **Weather is prescribed, not simulated.** Wind, humidity and cloud come from a data file. There is no coupling back from the scene to the atmosphere.
9. **Emissivity is temperature-independent.** $\varepsilon(\lambda)$ is authored once per material; the real quantity is $\varepsilon(\lambda,T)$, metals rising with $T$ and non-metals falling (§4.5). Below ~600 K the error sits inside the authored values' own uncertainty; for plumes and fire it does not.
10. **The sensor's environmental coupling is an empirical fit.** §9.5 reproduces the sign, magnitude and time constant of wind-driven camera drift from a lumped coefficient anchored on published UAV measurements [R44][R45]. It is not a thermal model of any particular airframe and must not be quoted as one.
11. **Surface state is authored, not derived.** Emissivity depends on finish, and on coating thickness in the thin-film regime (§4.5b); neither is recoverable from a mesh. The library names a state and cites it — there is no model here that predicts $\varepsilon$ from geometry.
12. **Weather media do not scatter light into the beam.** §7.6 removes what a fog, a cloud or a rain shaft scatters out of a pixel and does not add what they scatter in, except rain's diffraction halo (§7.8). That is right for the thermal bands near air temperature and wrong for sunlit fog, rain and cloud in the reflective bands, where in-scattered sunlight is first order. The wet-surface film optics (§4.6) and the precipitation near field (§7.8) are approximations held to their limits and to a few measurements, not validated models.
