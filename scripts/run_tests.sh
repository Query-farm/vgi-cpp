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
FULL_RUN=0
if [[ $# -eq 0 ]]; then FULL_RUN=1; fi

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

# HTTP-mode workers. The engine talks to these over a URL rather than by
# spawning them, so each is a long-lived process the harness owns for the run.
#
# Port 0 and read the port back: a fixed port collides with a previous run that
# has not finished closing its socket, which surfaces as an unrelated test
# failing to attach. The worker prints `PORT:<n>` once bound, which is the only
# race-free way to learn an ephemeral port.
HTTP_PIDS=()
stop_http_workers() {
  for pid in ${HTTP_PIDS[@]+"${HTTP_PIDS[@]}"}; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
  done
}
trap stop_http_workers EXIT

start_http_worker() { # name catalog [extra-env...]
  local name=$1 catalog=$2; shift 2
  local out="$CACHE/http-$name.out"
  : > "$out"
  # Started from the engine's directory, not the harness's. A spawned worker
  # inherits the engine's cwd; a long-lived HTTP one would otherwise resolve
  # the relative paths the COPY tests use — `duckdb_unittest_tempdir/...` —
  # against wherever this script happened to run, and report "cannot open"
  # for a file the engine had just written.
  ( cd "$VGI_EXT" || exit 1
    for kv in "$@"; do export "${kv?}"; done
    export VGI_WORKER_CATALOG_NAME="$catalog"
    # A spawned worker inherits the engine's environment; a long-lived HTTP one
    # has to be handed the same variables explicitly. `VGI_TEST_BRANCH_DIR` in
    # particular is read by both sides — the test writes a file there and the
    # catalog fixture names the same path — so a worker without it points its
    # branches at a directory the test never wrote to.
    export VGI_TEST_BRANCH_DIR="$BRANCH_DIR"
    export VGI_TEST_BEARER_TOKEN="test-secret-token"
    exec "$BIN" --http 0 >"$out" 2>>"$CACHE/worker.log" ) &
  HTTP_PIDS+=($!)

  # Bounded wait. A worker that never prints a port is a worker that failed to
  # start, and hanging here would blame the first test that used it.
  local port=""
  for _ in $(seq 1 100); do
    port=$(sed -n 's/^PORT:\([0-9]*\)$/\1/p' "$out" | head -1)
    [[ -n "$port" ]] && break
    sleep 0.1
  done
  if [[ -z "$port" ]]; then
    echo "[harness] HTTP worker '$name' never reported a port; see $CACHE/worker.log" >&2
    return 1
  fi
  echo "http://127.0.0.1:$port"
}

# Opt-in, because each one is a process held open for the whole run and the
# suite is useful without them. VGI_HTTP=1 turns the group on.
HTTP_ENV=()
if [[ "${VGI_HTTP:-0}" == "1" ]]; then
  echo "[harness] starting HTTP workers..."
  H_EXAMPLE=$(start_http_worker example example) || exit 1
  H_VERSIONED=$(start_http_worker versioned versioned) || exit 1
  H_VERSIONED_TABLES=$(start_http_worker versioned_tables versioned_tables) || exit 1
  echo "[harness] example=$H_EXAMPLE versioned=$H_VERSIONED tables=$H_VERSIONED_TABLES"
  # VGI_HTTP_TRANSPORT is a flag: it says VGI_TEST_WORKER is itself a URL, so
  # the whole suite runs over HTTP rather than by spawning a subprocess.
  HTTP_ENV=(
    VGI_TEST_WORKER="$H_EXAMPLE"
    VGI_HTTP_TRANSPORT=1
    VGI_VERSIONED_HTTP_WORKER="$H_VERSIONED"
    VGI_VERSIONED_TABLES_HTTP_WORKER="$H_VERSIONED_TABLES"
  )
fi

# The bearer fixture needs a protected HTTP worker while the rest of the suite
# needs an anonymous worker. Run it separately during a full suite instead of
# exporting its token into the launcher run, where bearer_token is invalid.
H_BEARER=""
if [[ $FULL_RUN == 1 ]]; then
  H_BEARER=$(start_http_worker bearer example \
    VGI_BEARER_TOKENS=test-secret-token=test-principal) || exit 1
fi

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
  ${HTTP_ENV[@]+"${HTTP_ENV[@]}"} \
  VGI_VERSIONED_WORKER="$W_VERSIONED" \
  VGI_VERSIONED_TABLES_WORKER="$W_VERSIONED_TABLES" \
  VGI_ATTACH_OPTIONS_WORKER="$W_ATTACH_OPTIONS" \
  VGI_BAD_PROTOCOL_WORKER="$W_BAD_PROTOCOL" \
  "$UNITTEST" "${ARGS[@]}" ) > "$CACHE/run.log" 2>&1
RC=$?

if [[ $FULL_RUN == 1 ]]; then
  ( cd "$VGI_EXT" && env \
    VGI_TEST_WORKER="$H_BEARER" \
    VGI_TEST_BEARER_TOKEN="test-secret-token" \
    "$UNITTEST" "test/sql/integration/bearer_auth/*" ) >> "$CACHE/run.log" 2>&1
  RC=$(( RC | $? ))
fi

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
