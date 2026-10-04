# Start from an existing asset

If the model already has an asset config (the Phantom 4 has a hand-written one in
[`configs/assets/phantom4.yaml`](../../../configs/assets/phantom4.yaml)), click **From an existing
asset…** under the library and pick it.

- Every Blender material whose name is in that map gets its material, whether the name matches as
  written or in the form USD uses (`[DJI_Phantom_4_Pro]Glass` matches `_DJI_Phantom_4_Pro_Glass`).
- Materials you have already assigned are kept, unless you tick **Replace existing assignments**.
- If the asset records a scale (the Phantom 4 was in centimetres), the message says so. Apply it
  from the size check ([step 8](08-check.md)).
- If the asset has a `parts:` block, **Hidden parts and contacts too** (on by default) brings back
  its hidden parts, as boxes placed where the config puts them and inside the part that holds
  them, with their component or their own numbers, and its contacts between parts your scene has,
  as confirmed connections with their joints and areas. Anything already in the scene is kept.

A model you exported with this add-on reopens with everything in place, including its hidden
parts and connections: open `3d_models/<name>/<name>.blend`.
