#!/usr/bin/env bash
# Regenerate the protocol headers under src/generated/ from vgi-python.
#
# The headers are generated into `vgi::generated`, not the `duckdb::vgi::
# generated` the generators default to: VGI is a wire protocol, not a DuckDB
# feature, and a standalone worker links no DuckDB and has no business
# declaring symbols in its namespace. The default stays `duckdb` so the
# extension's own copies are unaffected.
#
# Regenerating pulls in whatever protocol version vgi-python is currently at.
# If that is ahead of the engine you test against, the version gate will
# refuse every request — check `VGI_PROTOCOL_VERSION` in the diff.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGI_PYTHON="${VGI_PYTHON:-$HOME/Development/vgi-python}"
NS="${VGI_CPP_NAMESPACE:-vgi::generated}"

[[ -d "$VGI_PYTHON" ]] || { echo "vgi-python not found at $VGI_PYTHON (set VGI_PYTHON)" >&2; exit 2; }

run() { (cd "$VGI_PYTHON" && uv run --project . python -m "$1" --namespace "$NS"); }

run vgi.codegen.cpp_schemas          > "$ROOT/src/generated/vgi_protocol_schemas.hpp"
run vgi.codegen.cpp_constants        > "$ROOT/src/generated/vgi_protocol_constants.hpp"
run vgi.codegen.cpp_protocol_version > "$ROOT/src/generated/vgi_protocol_version.hpp"

echo "regenerated into namespace $NS"
grep -h "VGI_PROTOCOL_VERSION = " "$ROOT/src/generated/vgi_protocol_version.hpp"
