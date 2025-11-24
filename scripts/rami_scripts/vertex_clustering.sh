#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/vertex_clustering.py"
OUT_ROOT="$ROOT/data/my_new_meshes"
SRC_ROOT="/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed"
LOG_DIR="$ROOT/logs/vertex_clustering"

ISO_LEVEL=${ISO_LEVEL:-0.9}
TARGET_VERTS=${TARGET_VERTS:-2500}
TOLERANCE=${TOLERANCE:-0.02}
COMPONENT_THRESHOLD=${COMPONENT_THRESHOLD:-0.01}
SMOOTH_ITERS=${SMOOTH_ITERS:-5}
SAVE_HIST=${SAVE_HIST:-0}
MAX_PROCS=${MAX_PROCS:-8}
PYTHON_CMD=${PYTHON_CMD:-conda run -n mesh_autodecoder python}

mkdir -p "$OUT_ROOT" "$LOG_DIR"

PAIR_FILE=$(mktemp "$OUT_ROOT/pairs_XXXX.txt")

DEFAULT_GROUPS=$'cirrhotic/T1_masks/GT\ncirrhotic/T2_masks/GT\nhealthy/T1_masks/GT\nhealthy/T2_masks/GT'
DATA_GROUPS=${DATA_GROUPS:-$DEFAULT_GROUPS}

while IFS= read -r rel; do
  src="$SRC_ROOT/$rel"
  dst="$OUT_ROOT/$rel"
  [[ -d "$src" ]] || { echo "[WARN] missing $src"; continue; }
  mkdir -p "$dst"
  while IFS= read -r -d '' nii; do
    base=$(basename "${nii%.nii.gz}")
    out="$dst/$base.obj"
    [[ -f "$out" ]] || printf '%s|%s\n' "$nii" "$out" >> "$PAIR_FILE"
  done < <(find "$src" -name '*.nii.gz' -print0)
done <<< "$DATA_GROUPS"

tasks=$(wc -l < "$PAIR_FILE")
(( tasks > 0 )) || { echo "[INFO] nothing to do"; rm -f "$PAIR_FILE"; exit 0; }

echo "[INFO] processing $tasks meshes locally (max $MAX_PROCS parallel)"

read -r -a PY_ARR <<< "$PYTHON_CMD"

active=0
fail=0

while IFS='|' read -r input out; do
  log="$LOG_DIR/$(basename "${out%.*}").log"
  mkdir -p "$(dirname "$out")"

  (
    set -euo pipefail
    {
      echo "[START] $(date) :: $input"
      cmd=( "${PY_ARR[@]}" "$SCRIPT"
        --input "$input"
        --output "$out"
        --iso-level "$ISO_LEVEL"
        --target-verts "$TARGET_VERTS"
        --tolerance "$TOLERANCE"
        --component-threshold "$COMPONENT_THRESHOLD"
        --smooth-iters "$SMOOTH_ITERS"
      )
      [[ "$SAVE_HIST" == 1 ]] && cmd+=( --save-hist )
      echo "[CMD] ${cmd[*]}"
      "${cmd[@]}"
      echo "[DONE] $(date)"
    } &> "$log"
  ) &

  (( active+=1 ))
  if (( active >= MAX_PROCS )); then
    if ! wait -n; then
      fail=1
    fi
    (( active-=1 ))
  fi
done < "$PAIR_FILE"

while (( active > 0 )); do
  if ! wait -n; then
    fail=1
  fi
  (( active-=1 ))
done

rm -f "$PAIR_FILE"

if (( fail )); then
  echo "[WARN] one or more meshes failed (check $LOG_DIR)"
  exit 1
fi

echo "[INFO] finished all $tasks meshes"
echo "[INFO] logs dir: $LOG_DIR"