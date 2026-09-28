#!/usr/bin/env bash
# Run the add-on's end-to-end check in a headless Blender. CPU only; writes nothing into the
# repository (see blender_smoke.py). Usage: blender_addon/tests/run_blender_smoke.sh [--skip-audit]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER="${BLENDER:-blender}"
SCRATCH="${SCRATCH:-$(mktemp -d -t irsim_smoke_XXXX)}"
echo "scratch: $SCRATCH"
"$BLENDER" -b --factory-startup --python-exit-code 1 \
    --python "$HERE/blender_smoke.py" -- --scratch "$SCRATCH" "$@"
