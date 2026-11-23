#!/usr/bin/env bash
# © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#
# The extension's blob-vendor lane (Query-farm/vgi test/sql/external_storage/,
# contract: docs/testing/blob-test-catalog.md) against the C++ example worker
# over HTTP, with vgi-rpc-python's fake vendor. Each file gets its own worker
# (run_tests.py --server-per-test), so a file's `# server-env:` lines apply.
#
# Covered: the protocol 3.4 argument-reference files (arguments_ref*.test)
# and the stateless externalization files (scalar, table, result_integrity).
# Not covered: the files that need blob_buffer_echo / blob_tio /
# blob_store_values and vendor-backed function storage
# (VGI_STORAGE_EXTERNALIZE), which this SDK does not implement.
#
# Expected skips: arguments_ref_state_ref.test (VGI_WORKER_INIT_STATE_REF is
# not set: this SDK is single-replica in 3.4, so its init StateRef lives in
# memory and is never stored) and nothing else.
#
# Required environment:
#   VGI_SRC          a Query-farm/vgi checkout (protocol 3.4 or later)
#   HAYBARN_UNITTEST the unittest binary built from that revision
#   VGI_WORKER_BIN   the built vgi-example-worker
# Optional:
#   PYTHON           Python 3.13+ for scripts/run_tests.py (default python3)
#   VGI_VENDOR_LANE_FILES  override the file list (space-separated globs)
set -uo pipefail

: "${VGI_SRC:?path to a Query-farm/vgi checkout}"
: "${HAYBARN_UNITTEST:?path to the unittest binary}"
: "${VGI_WORKER_BIN:?path to the built vgi-example-worker}"
PYTHON="${PYTHON:-python3}"

# run_tests.py runs <root>/build/release/test/unittest; point it at the runner
# unless the checkout has a build of its own.
UNITTEST="$VGI_SRC/build/release/test/unittest"
if [ ! -e "$UNITTEST" ]; then
  mkdir -p "$(dirname "$UNITTEST")"
  ln -s "$HAYBARN_UNITTEST" "$UNITTEST"
fi

cd "$VGI_SRC" || exit 1
# shellcheck source=/dev/null
source test/start_fake_blob_vendor.sh
trap stop_fake_blob_vendor EXIT

export VGI_WORKER_ARGUMENTS_REF=1 VGI_FAKE_VENDOR_DELETE_ROUTE=1 VGI_HTTP_TRANSPORT=1

# shellcheck disable=SC2206
FILES=(${VGI_VENDOR_LANE_FILES:-"test/sql/external_storage/arguments_ref*" \
  test/sql/external_storage/scalar.test \
  test/sql/external_storage/table.test \
  test/sql/external_storage/result_integrity.test})

rc=0
"$PYTHON" scripts/run_tests.py --build release -j 1 \
  --server-per-test "$VGI_WORKER_BIN --http 0" \
  --allow-skip 'require-env VGI_WORKER_INIT_STATE_REF' \
  "${FILES[@]}" || rc=$?
echo "fake blob vendor counters: $(curl -s "$VGI_BLOB_VENDOR_URL/__stats__" | tr -s ' ')"
exit "$rc"
