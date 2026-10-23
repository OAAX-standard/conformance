#!/usr/bin/env bash
# Build the C harness and the mock runtimes into ./build
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
