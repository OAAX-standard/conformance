#!/usr/bin/env bash
# Smoke test of a built wheel in a clean environment. Usage: scripts/smoke.sh WHEEL [RUNTIME_LIB]
# Asserts on what the tool really prints and on exact exit codes: a wheel without its specs or with a broken import must fail here.
# With RUNTIME_LIB (the published reference CPU runtime) it also installs the wheel as a tool and runs the whole runtime tier with the
# harness inside the wheel: exit 1, 62 checks counted, 4 failed (the README's known findings). CI runs this in a bare container (no compiler).
set -uo pipefail
wheel=${1:?usage: smoke.sh WHEEL [RUNTIME_LIB]}
runtime_lib=${2:-}
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

if [ -n "$runtime_lib" ]; then
  # SMOKE_NO_COMPILER=1 (CI): the point is that nothing is compiled on this machine
  if [ -n "${SMOKE_NO_COMPILER:-}" ] && command -v gcc cc cmake >/dev/null 2>&1; then fail "a compiler is installed: the test would not prove the wheel is self-contained"; fi
  tools=$(mktemp -d)
  export UV_TOOL_DIR="$tools/tools" UV_TOOL_BIN_DIR="$tools/bin"
  uv tool install -q --python 3.12 "$wheel[e2e]" || fail "uv tool install failed"
  harness=$(find "$UV_TOOL_DIR" -path '*/oaax_conformance/bin/oaax-harness') || true
  [ -x "$harness" ] || fail "the harness is not in the installed wheel, or not executable: '$harness'"
  out=$("$UV_TOOL_BIN_DIR/oaax-conformance" runtime "$runtime_lib" --model @tiny-conv --quiet 2>&1); rc=$?
  [ "$rc" -eq 1 ] || fail "the reference CPU runtime should exit 1 (known findings), got $rc: $(printf '%s' "$out" | tail -5)"
  summary=$(printf '%s\n' "$out" | grep -E '^[0-9]+ pass') || fail "no result summary: $(printf '%s' "$out" | tail -5)"
  checks=$(printf '%s\n' "$summary" | awk '{ n = 0; for (i = 1; i < NF; i++) if ($i ~ /^[0-9]+$/ && $(i + 1) ~ /^(pass|warn|fail|blocked|n\/a),?$/) n += $i; print n }')
  [ "$checks" -eq 62 ] || fail "expected 62 checks counted, got '$checks': $summary"
  case "$summary" in *" 4 fail"*) ;; *) fail "expected the 4 known failures: $summary" ;; esac
  version="$version; runtime tier: $summary"
fi
echo "smoke test OK: $version, $count specs"
