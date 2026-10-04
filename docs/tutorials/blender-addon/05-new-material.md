# Add a material the library lacks

Click **New material…**, or **New from this…** to start from the selected material's values. Fill
in:

- **Name**: lower case with underscores, for example `nylon_black`.
- **Description**, **Source** (*estimated*, *literature* or *measured*) and **Reference**: where
  the numbers came from. Even "estimated from similar polymers" helps the next person.
- **Surface state**, for example *as manufactured*, *painted* or *anodised*. Emissivity belongs to
  the surface, not to the substance under it.
- **Heat storage**: density, specific heat, conductivity, the thickness of the skin that holds the
  heat, and solar absorptivity.
- **Emissivity as**: the form opens on **Curve**, because a measured curve is how the library's
  materials are now stored.
  - **Curve**: a measured spectrum. Pick a CSV with wavelength in µm and the value, one pair per
    line. Say whether the file holds emissivity or the reflectance of an opaque surface. A second
    file can cover another range, for example a short-wave reflectance file beside a long-wave
    emissivity file. **Check curve** reads the file with irsim's own loader and says which bands
    it covers. The per-band ε you type is used only where the curve has no data. irsim leaves
    out the bands the curve already covers, and says so.
  - **One value**: the same ε in every band, for a surface that behaves as a grey body and that
    nobody has measured.
  - **Per band (no curve)**: one ε for each of NIR, SWIR, MWIR and LWIR, when you have numbers
    but no curve. Starting from an existing material opens on this form, with its values.

  A curve is the only form under which two cameras with different wavelength ranges, say
  6–13 µm and 7.5–13.5 µm, see the material differently. Use one whenever you have measured data.
- **Transmittance τ** per band. **Reflectance ρ is shown, not typed in**: it is always 1 − ε − τ.
  That is how irsim keeps every material physically consistent.

When you click **Check and create**, irsim itself checks the material. If a band does not add up,
or the name is taken, nothing is written and the message says why. If the check passes, the file
appears as `configs/materials/<name>.yaml`; commit it like any other change. A picked curve is
copied to `data/spectra/materials/<name>.csv`, with a first line saying where it came from; commit
that file too.

In the library panel, each material says how it is authored (curve, per band, one value). For a
curve, it also shows how much of each band the curve supplied. **Plot across the bands** draws the
curve, with any per-band or single values dashed where the curve has no data. With *Assign it to the
selected parts* ticked, the new material is assigned to them straight away.
