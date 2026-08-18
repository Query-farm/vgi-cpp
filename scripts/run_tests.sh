#!/usr/bin/env bash
# Run the VGI integration suite in ~/Development/vgi against the C++ worker.
#
# The suite is the definition of done for this SDK, so this script is the
# primary feedback loop, not a convenience. Modelled on
# ~/Development/vgi-rust/scripts/run_tests.sh so the two ports are compared
# under the same conditions.
#
# Usage:
#   scripts/run_tests.sh                                  # full in-scope suite
#   scripts/run_tests.sh scalar                           # one category
#   scripts/run_tests.sh test/sql/integration/scalar/upper_case.test
#   scripts/run_tests.sh --no-build ...                   # skip cmake --build

set -uo pipefail

VGI_CPP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGI_EXT="${VGI_EXT:-$HOME/Development/vgi}"
UNITTEST="$VGI_EXT/build/release/test/unittest"
BUILD_DIR="${VGI_CPP_BUILD:-$VGI_CPP/build-release}"
BIN="$BUILD_DIR/example-worker/vgi-example-worker"
# Overridable so two runs (e.g. parallel agents) do not clobber each other's
# run.log, worker.log and wrapper scripts.
CACHE="${VGI_CPP_TEST_CACHE:-/tmp/vgi-cpp-test-cache}"
mkdir -p "$CACHE"

BRANCH_DIR="${VGI_TEST_BRANCH_DIR:-$CACHE/branches}"
mkdir -p "$BRANCH_DIR"

BUILD=1
if [[ "${1:-}" == "--no-build" ]]; then BUILD=0; shift; fi

if [[ ! -x "$UNITTEST" ]]; then
  echo "[harness] $UNITTEST missing — build the extension first:"
  echo "          cd $VGI_EXT && GEN=ninja make release"
  exit 1
fi

if [[ $BUILD == 1 ]]; then
  echo "[harness] building release worker..."
  cmake --build "$BUILD_DIR" -j8 2>&1 | tail -3
  if [[ ! -x "$BIN" ]]; then echo "[harness] build failed: $BIN missing"; exit 1; fi
fi

# The extension swallows the worker's stderr, which is where every diagnostic
# a worker can emit goes (stdout is the Arrow-IPC channel). Without this
# wrapper a worker-side failure shows up only as a generic query error.
: > "$CACHE/worker.log"
WRAP="$CACHE/worker-wrap.sh"
cat > "$WRAP" <<EOF
#!/usr/bin/env bash
exec "$BIN" "\$@" 2>>"$CACHE/worker.log"
EOF
chmod +x "$WRAP"

# One binary serves every catalog the suite attaches, switched by
# VGI_WORKER_CATALOG_NAME — same shape as the Rust fixture.
mk_wrapper() { # name catalog [extra-env...]
  local f="$CACHE/worker-$1.sh"
  { echo '#!/usr/bin/env bash'
    echo "export VGI_WORKER_CATALOG_NAME=$2"
    shift 2
    for kv in "$@"; do echo "export $kv"; done
    echo "exec \"$BIN\" \"\$@\" 2>>\"$CACHE/worker.log\""
  } > "$f"
  chmod +x "$f"
  echo "$f"
}

W_VERSIONED=$(mk_wrapper versioned versioned)
W_VERSIONED_TABLES=$(mk_wrapper versioned_tables versioned_tables)
W_ATTACH_OPTIONS=$(mk_wrapper attach_options attach_options)
W_BAD_PROTOCOL=$(mk_wrapper bad_protocol example VGI_PROTOCOL_VERSION_OVERRIDE=99.0.0)

ARGS=()
if [[ $# -ge 1 ]]; then
  case "$1" in
    test/*) ARGS=("$1");;
    *)      ARGS=("test/sql/integration/$1/*");;
  esac
else
  ARGS=("test/sql/integration/*")
fi

echo "[harness] running: ${ARGS[*]}"
( cd "$VGI_EXT" && env \
  VGI_TEST_BRANCH_DIR="$BRANCH_DIR" \
  VGI_TEST_WORKER="$WRAP" \
  VGI_VERSIONED_WORKER="$W_VERSIONED" \
  VGI_VERSIONED_TABLES_WORKER="$W_VERSIONED_TABLES" \
  VGI_ATTACH_OPTIONS_WORKER="$W_ATTACH_OPTIONS" \
  VGI_BAD_PROTOCOL_WORKER="$W_BAD_PROTOCOL" \
  VGI_TEST_BEARER_TOKEN="test-secret-token" \
  "$UNITTEST" "${ARGS[@]}" ) > "$CACHE/run.log" 2>&1
RC=$?

grep -oE 'test/sql/integration/[A-Za-z0-9_/]+\.test(_slow)?' "$CACHE/run.log" \
  | sort -u > "$CACHE/allmentioned" 2>/dev/null
awk '/unexpectedly|FAILED:|Mismatch on/{print}' "$CACHE/run.log" \
  | grep -oE 'test/sql/integration/[A-Za-z0-9_/]+\.test(_slow)?' | sort -u \
  > "$CACHE/failures" 2>/dev/null

echo "===== TAIL ====="
tail -8 "$CACHE/run.log"
echo "===== FAILURES ($(wc -l < "$CACHE/failures" | tr -d ' ')) ====="
head -40 "$CACHE/failures" 2>/dev/null
echo "(log: $CACHE/run.log  worker stderr: $CACHE/worker.log)"
exit $RC
