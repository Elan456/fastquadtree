#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYODIDE_HOST_PYTHON="${PYODIDE_HOST_PYTHON:-3.13}"
PYODIDE_XBUILDENV_VERSION="${PYODIDE_XBUILDENV_VERSION:-20260401}"
PYODIDE_XBUILDENV_CHANNEL="${PYODIDE_XBUILDENV_CHANNEL:-nightly}"
PYODIDE_EMSCRIPTEN_VERSION="${PYODIDE_EMSCRIPTEN_VERSION:-4.0.9}"
PYODIDE_XBUILDENV_ROOT="${PYODIDE_XBUILDENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/fastquadtree-pyodide}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.93.0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
PYODIDE_EXPECTED_WHEEL_GLOB="${PYODIDE_EXPECTED_WHEEL_GLOB:-fastquadtree-*-cp39-abi3-pyemscripten_2025_0_wasm32.whl}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

need_cmd uv
need_cmd rustup

printf 'Using Pyodide xbuildenv root: %s\n' "$PYODIDE_XBUILDENV_ROOT"
printf 'Using Pyodide host Python: %s\n' "$PYODIDE_HOST_PYTHON"
printf 'Using Pyodide xbuildenv channel: %s\n' "$PYODIDE_XBUILDENV_CHANNEL"
printf 'Using Pyodide xbuildenv version: %s\n' "$PYODIDE_XBUILDENV_VERSION"
printf 'Using Emscripten version: %s\n' "$PYODIDE_EMSCRIPTEN_VERSION"
printf 'Using Rust toolchain: %s\n' "$RUST_TOOLCHAIN"

install_args=(
    xbuildenv
    install
    "$PYODIDE_XBUILDENV_VERSION"
    --path
    "$PYODIDE_XBUILDENV_ROOT"
)
if [ "$PYODIDE_XBUILDENV_CHANNEL" = "nightly" ]; then
    install_args+=(--nightly)
fi

uv run --python "$PYODIDE_HOST_PYTHON" --with pyodide-build pyodide "${install_args[@]}"
uv run --python "$PYODIDE_HOST_PYTHON" --with pyodide-build pyodide \
    xbuildenv install-emscripten \
    --version "$PYODIDE_EMSCRIPTEN_VERSION" \
    --path "$PYODIDE_XBUILDENV_ROOT"

rustup toolchain install "$RUST_TOOLCHAIN"
rustup target add wasm32-unknown-emscripten --toolchain "$RUST_TOOLCHAIN"

EMSDK_ENV="$PYODIDE_XBUILDENV_ROOT/$PYODIDE_XBUILDENV_VERSION/emsdk/emsdk_env.sh"
if [ ! -f "$EMSDK_ENV" ]; then
    printf 'Expected emsdk activation script at %s\n' "$EMSDK_ENV" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$EMSDK_ENV" >/dev/null 2>&1

export PYODIDE_XBUILDENV_PATH="$PYODIDE_XBUILDENV_ROOT"
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"
export UV_CACHE_DIR

cd "$PROJECT_ROOT"
mkdir -p dist
rm -f dist/fastquadtree-*-pyemscripten_*_wasm32.whl
uv run --python "$PYODIDE_HOST_PYTHON" --with pyodide-build pyodide build . --outdir dist

printf '\nBuilt wheel:\n'
shopt -s nullglob
matches=("$PROJECT_ROOT"/dist/$PYODIDE_EXPECTED_WHEEL_GLOB)
if [ "${#matches[@]}" -eq 0 ]; then
    printf 'No wheel matching expected Pyodide target was produced: %s\n' \
        "$PYODIDE_EXPECTED_WHEEL_GLOB" >&2
    exit 1
fi
printf '%s\n' "${matches[@]}" | sort
