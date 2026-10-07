"""HU.11 part 2 -- a construction worker and a cyclist, each only a kit, an asset and scenes.

Neither added a line of Python: that is the roadmap's criterion, and ``test_kits.py`` checks both
kits against the rig and their assets. What is checked here is what each occupation is *for* in
an image, and would be wrong while still rendering plausibly:

* the cyclist's lycra is thin, so her bare forearms, hands, knees and calves carry no insulation
  and her shorts sit nearer her skin than a worker's trousers sit to his;
* the worker's hi-vis vest is dyed fluorescent orange, so it takes more sun than the silver
  tape on it, and at night the two are one temperature.

Garment insulation is ISO 9920's: a T-shirt 0.08 clo, short shorts 0.06, work trousers 0.24, with
the vest's 0.05 on the T-shirt's body (ADR 0196). Helmets: 0.12 on the head segment for a hard
hat's suspended shell (the helmet-system headform value of ADR 0197), 0.10 for a vented cycling
helmet (ESTIMATED: helmets cut scalp heat transfer by up to 30 %, Brühwiler et al. 2006,
doi:10.1080/02640410500457877).

docs/physics-model.md §6.1, §16.2; roadmap HU.11; ADR 0195, ADR 0196, ADR 0198.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

pytest.importorskip("jos3")

from irsim.config.humans import JOS3_SEGMENTS, load_body_schema  # noqa: E402
from irsim.config.scene import SceneConfig  # noqa: E402
from irsim.materials.mapping import load_asset_mapping  # noqa: E402
from irsim.scene import Scene  # noqa: E402
from irsim.thermal.human_body import clo_by_segment  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def _advance(stem: str) -> tuple[dict[str, float], float]:
    with open(REPO / "configs" / "scenes" / f"{stem}.yaml", encoding="utf-8") as fh:
        scene = Scene.from_config(SceneConfig.model_validate(yaml.safe_load(fh)))
    return scene.advance_targets(0.0, 30.0), scene.weather_at(0.0).t_air_k


def test_the_cyclist_is_bare_where_lycra_stops() -> None:
    h = load_asset_mapping("cyclist").human
    assert h is not None
    clo = dict(zip(JOS3_SEGMENTS, clo_by_segment(h, load_body_schema().garment_slots), strict=True))
    for seg in ("LArm", "RArm", "LHand", "RHand", "LLeg", "RLeg", "Neck"):
        assert clo[seg] == 0.0, seg
    for seg in ("Chest", "Back", "LShoulder", "Pelvis", "LThigh"):
        assert 0.0 < clo[seg] <= 0.08, seg


def test_thin_shorts_sit_nearer_the_skin_than_work_trousers() -> None:
    cyc, air = _advance("cyclist_summer_night")
    wrk, _ = _advance("construction_worker_summer_night")
    # both above the 17 C air, the 0.06-clo shorts warmer than the 0.24-clo trousers
    assert air < wrk["construction_worker.garment_legs"] < cyc["cyclist.garment_legs"]
    # and the bare face the warmest surface on either
    assert cyc["cyclist.skin_Head"] > max(v for k, v in cyc.items() if "garment" in k)


def test_the_orange_vest_takes_more_sun_than_its_tape_and_at_night_they_agree() -> None:
    noon, _ = _advance("construction_worker_summer_noon")
    assert noon["construction_worker.garment_torso"] > noon["construction_worker.equipment_tape"]
    night, _ = _advance("construction_worker_summer_night")
    assert night["construction_worker.equipment_tape"] == pytest.approx(
        night["construction_worker.garment_torso"], abs=1e-9
    )
