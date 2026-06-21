#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYODIDE_ENV_VERSION="${PYODIDE_ENV_VERSION:-20260401}"
PYODIDE_XBUILDENV_ROOT="${PYODIDE_XBUILDENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/fastquadtree-pyodide}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.93.0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

need_cmd pyodide
need_cmd rustup

printf 'Using Pyodide xbuildenv root: %s\n' "$PYODIDE_XBUILDENV_ROOT"
printf 'Using Pyodide nightly env: %s\n' "$PYODIDE_ENV_VERSION"
printf 'Using Rust toolchain: %s\n' "$RUST_TOOLCHAIN"

if [ ! -d "$PYODIDE_XBUILDENV_ROOT/$PYODIDE_ENV_VERSION" ]; then
    pyodide xbuildenv install "$PYODIDE_ENV_VERSION" --nightly --path "$PYODIDE_XBUILDENV_ROOT"
fi

pyodide xbuildenv install-emscripten --path "$PYODIDE_XBUILDENV_ROOT"
rustup toolchain install "$RUST_TOOLCHAIN"
rustup target add wasm32-unknown-emscripten --toolchain "$RUST_TOOLCHAIN"

EMSDK_ENV="$PYODIDE_XBUILDENV_ROOT/$PYODIDE_ENV_VERSION/emsdk/emsdk_env.sh"
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
pyodide build .

printf '\nBuilt wheel:\n'
find "$PROJECT_ROOT/dist" -maxdepth 1 -type f -name 'fastquadtree-*-pyemscripten_*_wasm32.whl' -print | sort
