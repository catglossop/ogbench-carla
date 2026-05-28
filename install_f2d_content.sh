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
#   ./install_f2d_content.sh CARLA_ROOT ZIP
#
# Example:
#   ./install_f2d_content.sh /home/carla/carla-0-9-16 /tmp/f2d_content.zip

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $(basename "$0") CARLA_ROOT ZIP" >&2
    exit 2
fi

CARLA_ROOT="$1"
ZIP="$2"
TARGET="$CARLA_ROOT/CarlaUE4"

if [[ ! -d "$TARGET" ]]; then
    echo "error: $TARGET does not exist; is CARLA_ROOT pointing at a packaged CARLA install?" >&2
    exit 1
fi

if [[ ! -s "$ZIP" ]]; then
    echo "error: zip $ZIP missing or empty" >&2
    exit 1
fi

echo "Extracting $ZIP into $TARGET/"
unzip -o -q "$ZIP" -d "$TARGET"

echo "Done. Restart your CARLA server to pick up the new blueprints."
