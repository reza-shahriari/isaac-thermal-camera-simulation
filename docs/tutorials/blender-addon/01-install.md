# Install the add-on

You do this once. The add-on is loaded straight from the repository, so a `git pull` also updates
it.

1. In Blender: **Edit ▸ Preferences ▸ Get Extensions**.
2. Open the **Repositories** drop-down (top right) and click **+ ▸ Add Local Repository**.
3. Tick **Custom Directory** and choose the repository's `blender_addon` folder. Click
   **Create**.
4. Go to **Add-ons**, find **irsim Thermal Materials** and tick it.
5. Expand it. **irsim repository** and **irsim Python** are usually found on their own; the line
   under each says what is being used. If either says *not found*, set it by hand:
   - the repository is the folder with `configs/`, `src/` and `blender_addon/` in it;
   - the Python is Isaac Sim's `python.sh`, not the `kit/python/bin/python3` inside it.
6. Click **Test connection**. You should see *OK: 23 materials in …/configs/materials* (the
   number grows as the library does).

The add-on appears as an **irsim** tab in the 3D viewport's sidebar (press **N** to show the
sidebar).
