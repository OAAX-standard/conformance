# Smoke test of a built Windows wheel (the counterpart of smoke.sh). Usage: scripts/smoke.ps1 WHEEL [RUNTIME_DLL]
# With RUNTIME_DLL (the published reference CPU runtime for Windows) it installs the wheel as a tool and runs the whole runtime tier with the
# harness inside the wheel: exit 1, 62 checks counted, 4 failed (the README's known findings).
param([Parameter(Mandatory)][string]$Wheel, [string]$RuntimeDll = "")
$ErrorActionPreference = "Continue"
function Fail($msg) { Write-Error "SMOKE TEST FAILED: $msg"; exit 1 }

$tools = Join-Path ([IO.Path]::GetTempPath()) ("oaax-smoke-" + [guid]::NewGuid())
$env:UV_TOOL_DIR = "$tools\tools"; $env:UV_TOOL_BIN_DIR = "$tools\bin"
uv tool install -q --python 3.12 "$Wheel[e2e]"
if ($LASTEXITCODE) { Fail "uv tool install failed" }
$exe = "$tools\bin\oaax-conformance.exe"

$version = (& $exe --version 2>&1) -join ""
if ($LASTEXITCODE -or $version -notmatch "^oaax-conformance \d") { Fail "unexpected --version output: $version" }

$err = (& $exe runtime C:\nonexistent.dll 2>&1 | Out-String)
if ($LASTEXITCODE -ne 3) { Fail "a missing library should exit 3, got ${LASTEXITCODE}: $err" }
if ($err -match "Traceback" -or $err -notmatch "no such library") { Fail "unexpected error text: $err" }

if ($RuntimeDll) {
  $harness = Get-ChildItem $env:UV_TOOL_DIR -Recurse -Filter oaax-harness.exe | Select-Object -First 1
  if (-not $harness) { Fail "the harness is not in the installed wheel" }
  $out = (& $exe runtime $RuntimeDll --model "@tiny-conv" --quiet 2>&1 | Out-String)
  if ($LASTEXITCODE -ne 1) { Fail "the reference CPU runtime should exit 1 (known findings), got ${LASTEXITCODE}: $out" }
  $summary = ($out -split "`r?`n" | Where-Object { $_ -match "^\d+ pass" } | Select-Object -First 1)
  if (-not $summary) { Fail "no result summary: $out" }
  $checks = ([regex]::Matches($summary, "(\d+) (pass|warn|fail|blocked|n/a)") | ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Sum).Sum
  if ($checks -ne 62) { Fail "expected 62 checks counted, got ${checks}: $summary" }
  if ($summary -notmatch " 4 fail") { Fail "expected the 4 known failures: $summary" }
  $version = "$version; runtime tier: $summary"
}
Write-Host "smoke test OK: $version"
