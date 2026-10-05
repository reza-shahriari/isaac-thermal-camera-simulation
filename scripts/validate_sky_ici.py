#!/usr/bin/env python3
"""XD.6: the layered clear sky against ARM's calibrated Infrared Cloud Imager, per night and angle.

    python scripts/validate_sky_ici.py [--out outputs/validation/ici]

Reads ``data/validation/ici_sgp2023_clear_sky.csv`` (no raw data needed), runs
:func:`irsim.validation.ici.model_clear_sky` for every clear image under the project's ESTIMATED
VOx response, and writes ``ici_clear_sky.json`` (per night: surface T, PWV, the model/measured
ratio and the instrument's own model/measured ratio at each elevation) and ``ici_clear_sky.png``
(radiance against elevation, measurement as points, both models as lines, one panel per night).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "outputs" / "validation" / "ici"))
    parser.add_argument("--preset", default="us_standard_clear")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from irsim.atmosphere.library import load_atmosphere_preset
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.validation.ici import load_ici_clear_sky, model_clear_sky, surface_rh_for_pwv

    images = [im for im in load_ici_clear_sky(REPO / "data/validation/ici_sgp2023_clear_sky.csv")]
    clear = [im for im in images if im.clear]
    preset = load_atmosphere_preset(args.preset)
    response = load_spectral_response(REPO / "data/spectra/responses/boson_vox.csv")
    lut = BandLUT.build(response, t0_k=150.0, t1_k=400.0, n=5001)
    nights = sorted({im.utc[:10] for im in clear})
    els = clear[0].elevations_deg

    report: dict[str, object] = {
        "preset": args.preset,
        "response": "boson_vox.csv (ESTIMATED stand-in; the ICI's response shape is unpublished)",
        "nights": {},
    }
    fig, axes = plt.subplots(1, len(nights), figsize=(4.2 * len(nights), 3.8), sharey=False)
    for ax, night in zip(axes, nights, strict=True):
        group = [im for im in clear if im.utc.startswith(night)]
        measured = np.array([im.radiance for im in group])
        modelled = np.array([model_clear_sky(im, preset, response, lut) for im in group])
        theirs = np.array([im.ici_model for im in group])
        t_c = float(np.mean([im.t_surface_k for im in group])) - 273.15
        pwv = float(np.mean([im.pwv_cm for im in group]))
        _, rh_needed = surface_rh_for_pwv(
            t_c + 273.15, pwv, preset.profile.water_vapour_scale_height_m
        )
        ratio = np.nanmean(modelled / measured, axis=0)
        theirs_ratio = np.nanmean(theirs / measured, axis=0)
        report["nights"][night] = {  # type: ignore[index]
            "images": len(group),
            "t_surface_c": round(t_c, 1),
            "pwv_cm": round(pwv, 2),
            "surface_rh_needed": round(rh_needed, 2),
            "elevations_deg": els.tolist(),
            "measured_w_m2_sr": np.round(np.nanmean(measured, axis=0), 2).tolist(),
            "model_over_measured": np.round(ratio, 3).tolist(),
            "ici_model_over_measured": np.round(theirs_ratio, 3).tolist(),
        }
        for row in measured:
            ax.plot(els, row, ".", color="0.55", ms=3)
        ax.plot(els, np.nanmean(modelled, axis=0), "-", color="C3", label="irsim layered sky")
        ax.plot(
            els, np.nanmean(theirs, axis=0), "--", color="C0", label="ICI's own clear-sky model"
        )
        ax.plot([], [], ".", color="0.55", label=f"ICI measured ({len(group)} images)")
        ax.set_title(f"{night}   {t_c:.1f} °C   PWV {pwv:.2f} cm", fontsize=9)
        ax.set_xlabel("elevation (deg)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("clear-sky radiance  W/(m² sr)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Clear LWIR sky: irsim vs ARM Infrared Cloud Imager, SGP 2023", fontsize=10)
    fig.tight_layout()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ici_clear_sky.png", dpi=130)
    (out / "ici_clear_sky.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for night, r in report["nights"].items():  # type: ignore[attr-defined]
        print(
            f"{night}: {r['t_surface_c']} C, PWV {r['pwv_cm']} cm -- model/measured "
            f"{min(r['model_over_measured']):.2f}..{max(r['model_over_measured']):.2f}, ICI's own "
            f"{min(r['ici_model_over_measured']):.2f}..{max(r['ici_model_over_measured']):.2f}"
        )
    print(f"wrote {out / 'ici_clear_sky.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
