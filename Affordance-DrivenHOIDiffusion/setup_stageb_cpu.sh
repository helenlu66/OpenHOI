#!/usr/bin/env bash
# Wire Stage B's relative asset paths (from the hydra configs) to their real
# locations, so no config files need editing. Safe to re-run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_ROOT="${OPENHOI_DATA_ROOT:-$HOME/openhoi-data}"
GRAB_DATA="${OPENHOI_GRAB_STAGEB:-$HOME/PhysAwareHOI/dataset/grab/OpenHOI}"
MANO_DIR="${OPENHOI_MANO_DIR:-$HOME/PhysAwareHOI/mano}"

mkdir -p data/mano/mano_v1_2
[ -e data/grab ]                    || ln -s "$GRAB_DATA" data/grab
[ -e data/mano/mano_v1_2/models ]   || ln -s "$MANO_DIR" data/mano/mano_v1_2/models
[ -e checkpoints ]                  || ln -s "$DATA_ROOT/checkpoints" checkpoints

echo "Stage B assets linked:"
echo "  data/grab -> $GRAB_DATA"
echo "  data/mano/mano_v1_2/models -> $MANO_DIR"
echo "  checkpoints -> $DATA_ROOT/checkpoints"
