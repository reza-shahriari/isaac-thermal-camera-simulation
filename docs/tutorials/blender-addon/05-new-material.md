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
- **Per band**: emissivity ε and transmittance τ. **Reflectance ρ is shown, not typed in**: it is
  always 1 − ε − τ. That is how irsim keeps every material physically consistent.

When you click **Check and create**, irsim itself checks the material. If a band does not add up,
or the name is taken, nothing is written and the message says why. If the check passes, the file
appears as `configs/materials/<name>.yaml`; commit it like any other change. With *Assign it to the
selected parts* ticked, the new material is assigned to them straight away.
