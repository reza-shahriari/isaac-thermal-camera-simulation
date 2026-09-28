# Start from an existing asset

If the model already has an asset config (the Phantom 4 has a hand-written one in
[`configs/assets/phantom4.yaml`](../../../configs/assets/phantom4.yaml)), click **From an existing
asset…** under the library and pick it.

- Every Blender material whose name is in that map gets its material, whether the name matches as
  written or in the form USD uses (`[DJI_Phantom_4_Pro]Glass` matches `_DJI_Phantom_4_Pro_Glass`).
- Materials you have already assigned are kept, unless you tick **Replace existing assignments**.
- If the asset records a scale (the Phantom 4 was in centimetres), the message says so. Apply it
  from the size check ([step 8](08-check.md)).

A model you exported with this add-on reopens with everything in place, including its hidden
parts and connections: open `3d_models/<name>/<name>.blend`.
