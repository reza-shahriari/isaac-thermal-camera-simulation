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

Materials must be **per-band spectral**, not per-band scalar, if you ever want MWIR right:

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
      roughness_per_band:  { nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12 }
      angular_model: { type: fresnel, n_k_file: "nk/acrylic_paint.csv" }
      transmittance_per_band: { nir: 0.0, swir: 0.0, mwir: 0.0, lwir: 0.0 }
```

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
- [ ] Fog kills visible and SWIR before LWIR
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
