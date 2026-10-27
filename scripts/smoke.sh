#!/usr/bin/env bash
# Smoke test of a built wheel in a clean environment. Usage: scripts/smoke.sh dist/oaax_conformance-X.Y.Z-py3-none-any.whl
# Asserts on what the tool really prints and on exact exit codes: a wheel without its specs or with a broken import must fail here.
set -uo pipefail
wheel=${1:?usage: smoke.sh WHEEL}
fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }
run() { uv run -q --isolated --no-project --python 3.12 --with "$wheel" "$@"; }

version=$(run oaax-conformance --version 2>&1) || fail "--version exited non-zero: $version"
case "$version" in oaax-conformance\ [0-9]*) ;; *) fail "unexpected --version output: $version" ;; esac

# the specs of the checks are data files of the wheel: without them nothing can run
count=$(run python -c "from oaax_conformance.specdocs import load; print(len(load()))" 2>&1) || fail "the specs cannot be loaded: $count"
[ "$count" -ge 90 ] 2>/dev/null || fail "only '$count' specs found in the wheel (expected at least 90)"

err=$(run oaax-conformance runtime /nonexistent.so 2>&1 >/dev/null); rc=$?
[ "$rc" -eq 3 ] || fail "a missing library should exit 3, got $rc: $err"
printf '%s' "$err" | grep -q "no such library" || fail "unexpected error text: $err"
printf '%s' "$err" | grep -q Traceback && fail "a traceback is not a clean error: $err"
echo "smoke test OK: $version, $count specs"
