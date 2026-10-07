#!/usr/bin/env python3
"""Rig any humanoid mesh with Make-It-Animatable: Mixamo bone weights per vertex, to an .npz (HU.9).

    ~/miniconda3/envs/mia/bin/python scripts/autorig_mia.py \\
        --mia-root ~/tools/Make-It-Animatable --input scan.glb --out scan.mia.npz

**Runs in Make-It-Animatable's own environment**, never the project interpreter: it pins torch
2.1.2 + CUDA 12.1, PyTorch3D and numpy 1.26, which Isaac Sim's Python must not receive
(``pip-into-isaac-python-check-pins-first``). Make-It-Animatable (Guo et al., CVPR 2025;
github.com/jasongzy/Make-It-Animatable, **MIT**) predicts, for a humanoid mesh in any pose, the
52 deforming bones of the Mixamo skeleton and every vertex's skinning weights, in about a second
on a GPU. Install::

    git clone --recursive https://github.com/jasongzy/Make-It-Animatable ~/tools/Make-It-Animatable
    conda create -n mia python=3.11 && conda activate mia
    pip install -r requirements.txt
    hf download jasongzy/Make-It-Animatable --include 'output/best/new/**' --local-dir .

This script drives the tool's own pipeline functions in-process (``prepare_input`` ->
``preprocess`` -> ``infer`` -> ``vis`` with its weight clean-up) and stops before its FBX export:
the project needs the weights, not an animation. It writes

* ``vertices`` -- the mesh's vertices **in the input file's frame** (trimesh's loader merges
  duplicates, so they are *not* the file's vertex list: ``rig_from_mia.py`` maps them back by
  position);
* ``weights`` -- (N, 52) float32, rows summing to 1;
* ``bones`` -- the 52 Mixamo names, ``mixamorig:`` prefixed, in column order;
* ``joints_head`` / ``joints_tail`` -- each bone's ends in the input file's frame, and ``parents``
  (each bone's parent index, -1 for the root);
* ``seconds`` -- wall time of the inference, for the roadmap's "under 2 min" criterion.

docs/physics-model.md §6.1; roadmap HU.9; ADR 0201.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mia-root", type=pathlib.Path, required=True)
    ap.add_argument("--input", type=pathlib.Path, required=True, help=".glb / .obj / .ply")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the .npz to write")
    ap.add_argument("--cpu", action="store_true", help="force the CPU (the roadmap's criterion)")
    ap.add_argument("--no-fingers", action="store_true", help="fold finger weights into the hand")
    a = ap.parse_args(argv)
    # CUDA numbers cards fastest-first, the reverse of nvidia-smi's PCI order on this machine, so
    # pin by bus order and honour $IRSIM_GPU as every Isaac entry point does (irsim_isaac.env):
    # the default card 0 is the A6000, and torch 2.1 cannot drive the RTX 5090 at all.
    # the tool imports Gradio, whose HTTP client refuses a socks:// proxy at import; nothing
    # here touches the network, so the variable is dropped for this process only
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.cpu else os.environ.get("IRSIM_GPU", "0")
    src = a.input.resolve()
    out = a.out.resolve()
    root = a.mia_root.expanduser().resolve()
    os.chdir(root)  # the tool loads its weights from paths relative to its root
    sys.path.insert(0, str(root))

    import app  # Make-It-Animatable's app.py
    import numpy as np
    import torch

    app.init_models()
    # The pipeline's steps return dicts keyed by the web interface's components; building that
    # interface needs the tool's example files, so placeholders stand in for the keys instead.
    for name in ("state", "output_joints_coarse", "output_normed_input", "output_sample",
                 "output_joints", "output_bw", "output_rest_lbs", "output_rest_vis",
                 "output_anim", "output_anim_vis"):  # fmt: skip
        setattr(app, name, object())
    # ...and its progress toasts (gr.Info / Warning / Success) need a live page: print them
    for toast in ("Info", "Warning", "Success"):
        setattr(app.gr, toast, lambda *msg, **_: print("[make-it-animatable]", *msg))
    db = app.DB()
    t0 = time.perf_counter()
    app.prepare_input(str(src), db=db, export_temp=True)
    vertices = np.array(db.mesh.vertices, dtype=np.float64)  # input frame, before normalising
    app.preprocess(db)
    app.infer(False, db)
    app.vis(True, "LeftArm", a.no_fingers, db)
    seconds = time.perf_counter() - t0

    inv = db.global_transform.inverse()
    head = inv.transform_points(torch.as_tensor(db.joints).float()[None]).squeeze(0).numpy()
    tail = inv.transform_points(torch.as_tensor(db.joints_tail).float()[None]).squeeze(0).numpy()
    bones = [name for name, _ in sorted(app.bones_idx_dict_joints.items(), key=lambda kv: kv[1])]
    tree = app.KINEMATIC_TREE_ADD if app.joints_additional else app.KINEMATIC_TREE
    parents = np.asarray(tree.parent_indices, dtype=np.int32)
    weights = np.asarray(db.bw, dtype=np.float32)
    if weights.shape != (len(vertices), len(bones)):
        raise SystemExit(f"weights {weights.shape} do not match {len(vertices)} x {len(bones)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=vertices.astype(np.float32),
        faces=np.asarray(db.faces, dtype=np.int32),
        weights=weights,
        bones=np.array(bones),
        joints_head=head.astype(np.float32),
        joints_tail=tail.astype(np.float32),
        parents=parents,
        seconds=np.float64(seconds),
        device=np.array(str(app.device)),
    )
    print(
        f"[autorig_mia] {len(vertices)} vertices, {len(bones)} bones, {seconds:.1f} s on "
        f"{app.device} -> {out}"
    )
    app.clear(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
