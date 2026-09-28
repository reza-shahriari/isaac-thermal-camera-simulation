#!/usr/bin/env bash
# Open the frame viewer on rendered frames: click a pixel, read everything saved under it.
# Usage: scripts/viewer.sh [folder] [--port N]   (folder defaults to outputs/; `make viewer` calls this)
# Finds a Python that can run it by itself -- $PYTHON, then Isaac Sim's python.sh, then python3 --
# so nobody has to remember an interpreter path to look at a picture. Guide: docs/frame-viewer.md
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
folder="${1:-$repo/outputs}"
shift || true

# `src/` on the path, so a checkout that was never `pip install`ed still runs.
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

candidates=(
    "${PYTHON:-}"
    "${ISAAC_PYTHON:-$HOME/IsaacSim/_build/linux-x86_64/release/python.sh}"
    python3
    python
)
for py in "${candidates[@]}"; do
    [ -n "$py" ] || continue
    command -v "$py" >/dev/null 2>&1 || [ -x "$py" ] || continue
    if "$py" -c "import irsim_viewer" >/dev/null 2>&1; then
        exec "$py" -m irsim_viewer "$folder" "$@"
    fi
done

echo "viewer: no Python here can import irsim_viewer (it needs NumPy and PyYAML)." >&2
echo "        Point PYTHON at the project interpreter, e.g." >&2
echo "        PYTHON=\$HOME/IsaacSim/_build/linux-x86_64/release/python.sh scripts/viewer.sh" >&2
exit 1
