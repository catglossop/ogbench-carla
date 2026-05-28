#!/usr/bin/env bash
# Install the Fail2Drive static content pack into a packaged CARLA install
# (e.g. /home/carla/carla-0-9-16). The pack contains only loose .uasset
# files -- no binaries, no plugin code, no PostProcessing material swaps --
# so it is safe to drop into a stock CARLA install without affecting any
# bench2drive features.
#
# Contents:
#   Content/AfricanAnimalsPack/        animal meshes + materials
#   Content/AnimalVarietyPack/         animal meshes + materials
#   Content/FarmAnimalsPack/           animal meshes + materials
#   Content/ImageAssets/               image-on-object props
#   Content/StopOcclusions/            stop-sign occluder props
#   Content/WallAssets/                roadblocked wall props
#   Content/Carla/Blueprints/Walkers/  17 BP_<Animal>.{uasset,uexp} pairs
#
# Usage:
#   ./install_f2d_content.sh CARLA_ROOT [--zip PATH | --url URL]
#
# Examples:
#   ./install_f2d_content.sh /home/carla/carla-0-9-16
#   ./install_f2d_content.sh /home/carla/carla-0-9-16 --zip /tmp/f2d.zip
#   ./install_f2d_content.sh /home/carla/carla-0-9-16 --url https://.../f2d.zip

set -euo pipefail

# Edit this when you cut a new release on the ogbench-carla repo.
DEFAULT_URL="https://github.com/catglossop/ogbench-carla/releases/download/f2d-content-v1/f2d_content_pack.zip"
EXPECTED_SHA256="e966a2a98f1f99a2f6375b5c7d3907e6900c121c02187179109816b3d05276cf"

CARLA_ROOT=""
ZIP_PATH=""
URL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip)        ZIP_PATH="$2"; shift 2;;
        --url)        URL="$2"; shift 2;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
            exit 0;;
        -*) echo "unknown option: $1" >&2; exit 2;;
        *)  CARLA_ROOT="$1"; shift;;
    esac
done

if [[ -z "$CARLA_ROOT" ]]; then
    echo "usage: $(basename "$0") CARLA_ROOT [--zip PATH | --url URL]" >&2
    exit 2
fi

TARGET="$CARLA_ROOT/CarlaUE4"
if [[ ! -d "$TARGET" ]]; then
    echo "error: $TARGET does not exist; is CARLA_ROOT pointing at a packaged CARLA install?" >&2
    exit 1
fi

# Acquire the zip.
CLEANUP_ZIP=0
if [[ -z "$ZIP_PATH" ]]; then
    ZIP_PATH="$(mktemp --suffix=.zip)"
    CLEANUP_ZIP=1
    trap '[[ "$CLEANUP_ZIP" == 1 ]] && rm -f "$ZIP_PATH"' EXIT
    URL="${URL:-$DEFAULT_URL}"
    echo "Downloading $URL"
    curl -fL --progress-bar "$URL" -o "$ZIP_PATH"
fi

if [[ ! -s "$ZIP_PATH" ]]; then
    echo "error: zip $ZIP_PATH missing or empty" >&2
    exit 1
fi

# Verify checksum (skip if EXPECTED_SHA256 is empty).
if [[ -n "$EXPECTED_SHA256" ]] && command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$ZIP_PATH" | awk '{print $1}')"
    if [[ "$got" != "$EXPECTED_SHA256" ]]; then
        echo "warning: sha256 mismatch (got $got, expected $EXPECTED_SHA256). Continuing." >&2
    fi
fi

echo "Extracting $ZIP_PATH into $TARGET/"
unzip -o -q "$ZIP_PATH" -d "$TARGET"

echo "Done. Restart your CARLA server to pick up the new blueprints."
