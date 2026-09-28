# See it all on a real model

[`blender_addon/demo/make_phantom4_demo.py`](../../../blender_addon/demo/make_phantom4_demo.py)
does the whole tutorial, headless, on the DJI Phantom 4. It needs the git-ignored
`3d_models/phantom4.fbx`.

```bash
blender -b --factory-startup --python blender_addon/demo/make_phantom4_demo.py -- \
    --out outputs/blender_addon_demo --render --structure --export-scratch /tmp/phantom_export
```

What it does, and what it measured on the last run:

1. Imports the model: 2,486,459 faces in 41 parts. The check takes 0.34 s and reads 46.4 m across,
   which it recognises as centimetres. Scaled, it is 0.4105 × 0.4637 × 0.2067 m.
2. Loads the hand-written map ([step 4](04-existing-asset.md)): 100 % coverage. The one
   mirror-like material flagged is the chrome trim, chosen on purpose.
3. Adds the flight battery inside the body as a hidden part ([step 7](07-hidden-parts.md)), laid
   fore and aft, then finds the connections ([step 6](06-connections.md)): 103 contacts and 87
   facing pairs in 9.9 s.
4. Saves `phantom4_thermal_demo.blend`. It opens in the thermal view with the sidebar out and the
   connections shown.
5. With `--export-scratch`, exports into a scratch copy of the repository and runs the project's
   audit: 41/41 parts mapped.
6. With `--render`, draws the pictures in this tutorial on the CPU.

![The model as it looks, and its LWIR emissivity with the hand-written map](images/phantom4_look_and_emissivity.webp)
