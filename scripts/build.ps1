# Build the C harness and the mock runtimes into .\build (Windows; Visual Studio or any compiler CMake finds)
$ErrorActionPreference = "Continue"  # cmake prints warnings on stderr: only the exit codes decide
Set-Location (Join-Path $PSScriptRoot "..")
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build build --config Release --parallel
exit $LASTEXITCODE
