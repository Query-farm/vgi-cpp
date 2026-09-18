#!/usr/bin/env bash
# Regenerate the protocol headers under include/vgi/generated/ from vgi-python.
#
# These headers are public (installed with the rest of include/vgi/): a
# client SDK (e.g. vgi-sqlite) links vgi::vgi purely to get at these request
# and result schemas, without needing any worker-side symbol.
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

run vgi.codegen.cpp_schemas          > "$ROOT/include/vgi/generated/vgi_protocol_schemas.hpp"
run vgi.codegen.cpp_constants        > "$ROOT/include/vgi/generated/vgi_protocol_constants.hpp"
run vgi.codegen.cpp_protocol_version > "$ROOT/include/vgi/generated/vgi_protocol_version.hpp"
# The wire routing key (`vgi.v2`): generated, not hand-spelled, for the reason
# the generator gives. A version bump that misses this tree is a red build; a
# rename that missed it was a silent misroute -- the worker answered on a name
# no client sent, and nothing failed until integration.
run vgi.codegen.cpp_protocol_name    > "$ROOT/include/vgi/generated/vgi_protocol_names.hpp"

# Rewrite the provenance banner. Two reasons, both real:
#
#   * The generator names ~/Development/vgi as the destination, which is the
#     extension's copy, not this one, and omits the --namespace flag these
#     headers depend on. A reader who follows it puts the wrong file in the
#     wrong namespace.
#   * Its command spans two lines with a trailing backslash, and a backslash at
#     the end of a `//` comment is a line continuation — GCC warns
#     (-Wcomment) and swallows the next line. Clang does not, so it only shows
#     up on the Linux CI legs.
for f in vgi_protocol_schemas.hpp vgi_protocol_constants.hpp vgi_protocol_version.hpp \
         vgi_protocol_names.hpp; do
    python3 - "$ROOT/include/vgi/generated/$f" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
src = re.sub(r"// To regenerate:\n(//   [^\n]*\n)+",
             "// To regenerate:\n//   scripts/regenerate_protocol.sh\n", src, count=1)
open(p, "w").write(src)
PY
done

echo "regenerated into namespace $NS"
grep -h "VGI_PROTOCOL_VERSION = " "$ROOT/include/vgi/generated/vgi_protocol_version.hpp"
grep -h "VGI_PROTOCOL_NAME = " "$ROOT/include/vgi/generated/vgi_protocol_names.hpp"
