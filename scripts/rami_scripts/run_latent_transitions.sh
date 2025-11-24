#!/usr/bin/env bash
set -euo pipefail

# Location of latent_transition_video.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SCRIPT_PATH="$ROOT/latent_transition_video.py"

PYTHON_CMD=${PYTHON_CMD:-python}

# Sweep parameter lists (space-separated overrides supported)
LATENT_SOURCES_STR=${LATENT_SOURCES:-"healthy_mean cirrhotic_mean"}
read -r -a LATENT_SOURCES_ARR <<< "$LATENT_SOURCES_STR"

LATENT_INDICES_STR=${LATENT_INDICES:-"66 40 51 79 126"}
read -r -a LATENT_INDICES_ARR <<< "$LATENT_INDICES_STR"

# Common arguments (override via environment variables as needed)
CHECKPOINT=${CHECKPOINT:-"/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"}
LATENT_DIR=${LATENT_DIR:-"/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents"}
CLASSIFIER_MODEL=${CLASSIFIER_MODEL:-"/home/ralbe/DALS/mesh_autodecoder/inference_results/latent_classifier_outputs/best_model_seed_1343.pt"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/video_frames"}

MIN_VALUE=${MIN_VALUE:--0.5}
MAX_VALUE=${MAX_VALUE:-0.5}
STEPS=${STEPS:-200}
FPS=${FPS:-24}
RENDER_MODE=${RENDER_MODE:-distance_colormap}
DEVICE=${DEVICE:-}
KEEP_FRAMES=${KEEP_FRAMES:-0}

EXTRA_ARGS_STR=${EXTRA_ARGS:-}
read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS_STR"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERR] Cannot locate latent_transition_video.py at $SCRIPT_PATH" >&2
  exit 1
fi

for source in "${LATENT_SOURCES_ARR[@]}"; do
  for index in "${LATENT_INDICES_ARR[@]}"; do
    cmd=(
      "$PYTHON_CMD" "$SCRIPT_PATH"
      --latent-source "$source"
      --latent-index "$index"
      --min-value "$MIN_VALUE"
      --max-value "$MAX_VALUE"
      --steps "$STEPS"
      --fps "$FPS"
      --checkpoint "$CHECKPOINT"
      --latent-dir "$LATENT_DIR"
      --classifier-model "$CLASSIFIER_MODEL"
      --output-root "$OUTPUT_ROOT"
      --render-mode "$RENDER_MODE"
    )
    if [[ -n "$DEVICE" ]]; then
      cmd+=(--device "$DEVICE")
    fi
    if [[ "$KEEP_FRAMES" == 1 ]]; then
      cmd+=(--keep-frames)
    fi
    if [[ ${#EXTRA_ARGS_ARR[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS_ARR[@]}")
    fi

    echo "[INFO] Running ${cmd[*]}"
    "${cmd[@]}"
  done
done

