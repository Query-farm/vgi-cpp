#!/usr/bin/env bash
# © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#
# Direct S3 storage (VGI_STORAGE_BACKEND=s3, docs/direct-storage.md) end to
# end: the C++ example worker, built with the s3 feature, keeps large values
# in an S3-compatible bucket, and the extension's native runner drives it over
# HTTP. Three worker configurations:
#
#   static     static keys, checksum sha256: results + an argument via a grant
#   session    an STS session token (AssumeRole) with its expiry: the same
#   none       VGI_STORAGE_S3_CHECKSUM=none: results only (no grants)
#
# Each run must leave new objects under <prefix>rpc/, and a missing bucket
# must stop the worker at startup with VGI_STORAGE_S3_VERIFY=1. A skipped test
# fails the lane.
#
# Required environment:
#   HAYBARN_UNITTEST  the native runner (unittest) for the vgi test suite
#   VGI_WORKER_BIN    vgi-example-worker built with -DVGI_RPC_WITH_S3=ON
#   S3_ENDPOINT       the S3 endpoint, e.g. http://127.0.0.1:9000
#   S3_ACCESS_KEY / S3_SECRET_KEY  root keys (create buckets, AssumeRole)
# Optional:
#   PYTHON            Python with boto3 (default python3)
#   LANE_CONFIGS      subset of "static session none"
set -uo pipefail

: "${HAYBARN_UNITTEST:?path to the native runner}"
: "${VGI_WORKER_BIN:?path to vgi-example-worker built with the s3 feature}"
: "${S3_ENDPOINT:?S3 endpoint URL}"
: "${S3_ACCESS_KEY:?}" "${S3_SECRET_KEY:?}"
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS="$HERE/direct-s3"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BUCKET="vgi-cpp-direct-s3-$(date +%s)"

s3py() {
  "$PYTHON" - "$@" <<'PY'
import json, os, sys
import boto3
endpoint, key, secret = os.environ["S3_ENDPOINT"], os.environ["S3_ACCESS_KEY"], os.environ["S3_SECRET_KEY"]
kw = dict(endpoint_url=endpoint, region_name="us-east-1", aws_access_key_id=key, aws_secret_access_key=secret)
op, *args = sys.argv[1:]
if op == "mkbucket":
    boto3.client("s3", **kw).create_bucket(Bucket=args[0])
elif op == "count":
    r = boto3.client("s3", **kw).list_objects_v2(Bucket=args[0], Prefix=args[1])
    print(r.get("KeyCount", 0))
elif op == "session":
    c = boto3.client("sts", **kw).assume_role(
        RoleArn="arn:aws:iam::000000000000:role/vgi-cpp-lane", RoleSessionName="vgi-cpp-lane",
        DurationSeconds=3600)["Credentials"]
    print(json.dumps({"id": c["AccessKeyId"], "secret": c["SecretAccessKey"],
                      "token": c["SessionToken"], "expires": c["Expiration"].isoformat()}))
PY
}

s3py mkbucket "$BUCKET" || { echo "::error::cannot create bucket $BUCKET"; exit 1; }

# Starts the worker with the storage env in "$@"; sets WORKER_PID and VGI_TEST_WORKER.
start_worker() {
  local log="$WORK/worker-$1.log"; shift
  env -u VGI_BLOB_VENDOR_URL \
    VGI_STORAGE_S3_BUCKET="$BUCKET" VGI_STORAGE_S3_ENDPOINT="$S3_ENDPOINT" \
    VGI_STORAGE_ALLOW_PLAIN_HTTP=1 VGI_STORAGE_S3_VERIFY=1 AWS_EC2_METADATA_DISABLED=true \
    VGI_RPC_MAX_REQUEST_BYTES=32768 VGI_RPC_MAX_UPLOAD_BYTES=1073741824 \
    VGI_RPC_EXTERNALIZE_THRESHOLD_BYTES=1048576 \
    "$@" "$VGI_WORKER_BIN" --http 0 > "$log" 2>&1 &
  WORKER_PID=$!
  local port=""
  for _ in $(seq 1 120); do
    port="$(sed -n 's/.*PORT:\([0-9]*\).*/\1/p' "$log" | head -1)"
    [ -n "$port" ] && break
    kill -0 "$WORKER_PID" 2>/dev/null || break
    sleep 0.25
  done
  if [ -z "$port" ]; then
    echo "::error::worker did not start"; cat "$log"; return 1
  fi
  grep "direct storage" "$log" || true
  export VGI_TEST_WORKER="http://127.0.0.1:$port"
}

stop_worker() { kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; }

# One native-runner invocation per file (it takes a single filter). The
# no-skip config turns an error the runner would treat as "skip" into a
# failure; a reported skip fails the lane too.
run_tests() {
  local config="$1"; shift
  local rc=0 file out
  for file in "$@"; do
    out="$WORK/run-$config-$(basename "$file").out"
    # The pre-signed URLs are plain http on loopback.
    (cd "$WORK" && VGI_TEST_ALLOW_LOOPBACK_EXTERNAL_LOCATIONS=1 \
      "$HAYBARN_UNITTEST" --test-dir "$TESTS" --test-config "$TESTS/noskip.json" "$file") > "$out" 2>&1 || rc=1
    tail -3 "$out"
    if grep -qi "skipped" "$out"; then
      echo "::error::$config: $file was skipped"; rc=1
    fi
    [ "$rc" -eq 0 ] || grep -B2 -A8 "FAILED\|unexpectedly" "$out" | head -40
  done
  return $rc
}

rc=0
for config in ${LANE_CONFIGS:-static session none}; do
  prefix="lane-$config/"
  before="$(s3py count "$BUCKET" "${prefix}rpc/")"
  case "$config" in
    static)
      start_worker "$config" VGI_STORAGE_S3_PREFIX="$prefix" \
        VGI_STORAGE_S3_ACCESS_KEY_ID="$S3_ACCESS_KEY" VGI_STORAGE_S3_SECRET_ACCESS_KEY="$S3_SECRET_KEY" || { rc=1; continue; }
      files=(test/results.test test/arguments.test) ;;
    session)
      creds="$(s3py session)" || { echo "::error::STS AssumeRole failed"; rc=1; continue; }
      field() { "$PYTHON" -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$creds" "$1"; }
      start_worker "$config" VGI_STORAGE_S3_PREFIX="$prefix" \
        VGI_STORAGE_S3_ACCESS_KEY_ID="$(field id)" VGI_STORAGE_S3_SECRET_ACCESS_KEY="$(field secret)" \
        VGI_STORAGE_S3_SESSION_TOKEN="$(field token)" VGI_STORAGE_S3_SESSION_EXPIRES="$(field expires)" \
        || { rc=1; continue; }
      files=(test/results.test test/arguments.test) ;;
    none)
      start_worker "$config" VGI_STORAGE_S3_PREFIX="$prefix" VGI_STORAGE_S3_CHECKSUM=none \
        VGI_STORAGE_S3_ACCESS_KEY_ID="$S3_ACCESS_KEY" VGI_STORAGE_S3_SECRET_ACCESS_KEY="$S3_SECRET_KEY" || { rc=1; continue; }
      files=(test/results.test) ;;
  esac
  echo "== direct-s3 / $config"
  run_tests "$config" "${files[@]}" || rc=1
  stop_worker
  after="$(s3py count "$BUCKET" "${prefix}rpc/")"
  if [ "$after" -le "$before" ]; then
    echo "::error::$config: nothing was stored under ${prefix}rpc/ (results stayed inline?)"; rc=1
  else
    echo "$config: $((after - before)) object(s) under ${prefix}rpc/"
  fi
done

# A missing bucket stops the worker at startup, saying why.
log="$WORK/worker-missing.log"
env -u VGI_BLOB_VENDOR_URL VGI_STORAGE_S3_BUCKET=no-such-bucket-vgi-cpp \
  VGI_STORAGE_S3_ENDPOINT="$S3_ENDPOINT" VGI_STORAGE_ALLOW_PLAIN_HTTP=1 VGI_STORAGE_S3_VERIFY=1 \
  VGI_STORAGE_S3_ACCESS_KEY_ID="$S3_ACCESS_KEY" VGI_STORAGE_S3_SECRET_ACCESS_KEY="$S3_SECRET_KEY" \
  AWS_EC2_METADATA_DISABLED=true timeout 60 "$VGI_WORKER_BIN" --http 0 > "$log" 2>&1
if [ $? -eq 0 ] || ! grep -q "does not exist" "$log"; then
  echo "::error::a missing bucket did not stop the worker"; cat "$log"; rc=1
else
  echo "missing bucket: refused at startup"
fi
exit "$rc"
