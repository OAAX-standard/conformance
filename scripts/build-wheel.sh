#!/usr/bin/env bash
# Build the release wheel of THIS machine's architecture (x86_64 or aarch64) with the C harness inside, in a manylinux container
# so the binary runs on old and new distros alike. Needs Docker. Usage: scripts/build-wheel.sh   (writes ./dist)
set -euo pipefail
cd "$(dirname "$0")/.."
arch=$(uname -m)
case "$arch" in x86_64 | aarch64) ;; *) echo "unsupported architecture: $arch" >&2; exit 1 ;; esac
image=quay.io/pypa/manylinux_2_28_${arch}:2026.10.26-1
# the binary only needs glibc 2.17 symbols (CI checks it with `auditwheel show`): that is the honest tag, with its old alias
plat=manylinux_2_17_${arch}.manylinux2014_${arch}
mkdir -p dist
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --pretty=%ct)}" \
  -e OAAX_WHEEL_PLAT="$plat" -e OAAX_REQUIRE_HARNESS=1 -v "$PWD":/src -w /src "$image" \
  uv build --wheel --python /opt/python/cp313-cp313/bin/python --out-dir dist
