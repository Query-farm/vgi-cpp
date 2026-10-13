#!/usr/bin/env bash
# Regenerate the public schemas and worker registry together, using the
# canonical atomic driver and the SDK's vgi::generated namespace.
set -euo pipefail

VGI_CPP_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGI_PYTHON="${VGI_PYTHON:-$HOME/Development/vgi-python}"
[[ -d "$VGI_PYTHON" ]] || { echo "vgi-python not found at $VGI_PYTHON (set VGI_PYTHON)" >&2; exit 2; }
VGI_CPP_ROOT="$VGI_CPP_CHECKOUT" uv run --project "$VGI_PYTHON" python \
    "$VGI_PYTHON/scripts/regen_generated.py" --only vgi-c++ --require-all
