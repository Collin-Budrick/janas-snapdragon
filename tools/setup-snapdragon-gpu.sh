#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Native ARM64 Linux/WSL shader tools for Janas's Windows ARM64 GPU build.
# Downloads and build outputs stay in a user cache; no driver, package,
# system PATH, or Janas CPU build settings are changed.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Usage: bash tools/setup-snapdragon-gpu.sh

Build pinned glslang and fetch pinned Vulkan headers in ARM64 Linux/WSL.
Requires git, cmake >= 3.22.1, make, a native C/C++ compiler and internet.

Optional environment:
  JANAS_TOOLCHAIN_CACHE  cache root (default ~/.cache/janas-toolchains)
  JANAS_GPU_SETUP_JOBS   parallel compiler jobs (default 4)

The final lines give PATH and JANAS_CFLAGS values for build.sh. This script
does not build Janas, install drivers/packages, or alter global settings.
EOF
    exit 0
fi
if [[ $# != 0 ]]; then
    echo 'setup-snapdragon-gpu: no arguments expected (use --help)' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
    echo 'setup-snapdragon-gpu: run in native ARM64 Linux/WSL' >&2
    exit 2
fi
for tool in git cmake make cc c++; do
    command -v "$tool" >/dev/null || {
        echo "setup-snapdragon-gpu: missing prerequisite: $tool" >&2
        exit 2
    }
done

cache="${JANAS_TOOLCHAIN_CACHE:-$HOME/.cache/janas-toolchains}"
jobs="${JANAS_GPU_SETUP_JOBS:-4}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo 'setup-snapdragon-gpu: JANAS_GPU_SETUP_JOBS must be positive' >&2
    exit 2
fi
mkdir -p "$cache"
cache="$(cd "$cache" && pwd)"

# Tags are convenient for fetching; the immutable commits verify the source.
glslang_tag=16.6.0
glslang_commit=e1b562a8bed273a02f30b59b66a5d499793cede5
headers_tag=vulkan-sdk-1.4.363.0
headers_commit=6802bb4733b63ed5efd3adb308a6c885ef180ea1
glslang_src="$cache/glslang-$glslang_tag-src"
glslang_build="$cache/glslang-$glslang_tag-build-aarch64"
glslang_prefix="$cache/glslang-$glslang_tag-linux-aarch64"
headers="$cache/vulkan-headers-1.4.363"

source_tree() {
    local url="$1" tag="$2" commit="$3" dir="$4"
    if [[ ! -e "$dir" ]]; then
        git -c advice.detachedHead=false clone --depth 1 --branch "$tag" \
            "$url" "$dir"
    fi
    if [[ ! -d "$dir/.git" || "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
        echo "setup-snapdragon-gpu: unexpected source revision in $dir" >&2
        exit 2
    fi
    if ! git -C "$dir" diff --quiet HEAD --; then
        echo "setup-snapdragon-gpu: source has local modifications: $dir" >&2
        exit 2
    fi
}

source_tree https://github.com/KhronosGroup/glslang.git \
    "$glslang_tag" "$glslang_commit" "$glslang_src"
source_tree https://github.com/KhronosGroup/Vulkan-Headers.git \
    "$headers_tag" "$headers_commit" "$headers"

# The standalone compiler needs no SPIRV-Tools checkout with optimization off.
# A Makefile generator avoids requiring a separately installed Ninja binary.
cmake -S "$glslang_src" -B "$glslang_build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF -DENABLE_HLSL=OFF \
    -DGLSLANG_TESTS=OFF -DGLSLANG_ENABLE_INSTALL=OFF \
    -DENABLE_GLSLANG_BINARIES=ON
cmake --build "$glslang_build" --target glslang-standalone --parallel "$jobs"
mkdir -p "$glslang_prefix/bin"
install -m 755 "$glslang_build/StandAlone/glslang" "$glslang_prefix/bin/glslang"
ln -sfn glslang "$glslang_prefix/bin/glslangValidator"

validator="$glslang_prefix/bin/glslangValidator"
"$validator" --version
probe="$(mktemp -d "$cache/gpu-probe.XXXXXX")"
trap 'rm -rf "$probe"' EXIT
cat >"$probe/integer-dot.comp" <<'EOF'
#version 450
#extension GL_EXT_integer_dot_product : require
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer Result { int value; } result;
void main() { result.value = dotPacked4x8EXT(0x01010101, 0x01010101); }
EOF
"$validator" -V --target-env vulkan1.3 "$probe/integer-dot.comp" \
    -o "$probe/integer-dot.spv"
printf '#include <vulkan/vulkan.h>\n#ifndef VK_VERSION_1_3\n#error Vulkan 1.3 headers required\n#endif\n' |
    cc -I"$headers/include" -x c -E - >/dev/null

printf '\nGPU prerequisites verified: glslang %s and Vulkan headers 1.4.363.\n' "$glslang_tag"
printf 'Use these values for a GPU-enabled build.sh clean-build:\n'
printf '  PATH=%q:$PATH\n' "$glslang_prefix/bin"
printf '  JANAS_CFLAGS=%q\n' "-I$headers/include"
