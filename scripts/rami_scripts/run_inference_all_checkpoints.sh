#!/bin/bash

# Paths
SCRIPT_PATH="/home/ralbe/DALS/mesh_autodecoder/inference_meshdecoder_single.py"
OUTPUT_DIR="/home/ralbe/DALS/mesh_autodecoder/inference_results"

# Checkpoints to process
CHECKPOINTS=(
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-15-42.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-15-44.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-15-45.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-15-50.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-17-54.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-18-18.ckpt"
  "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_02-18-19.ckpt"
)

mkdir -p "$OUTPUT_DIR"

job_count=0

for checkpoint in "${CHECKPOINTS[@]}"; do
  ckpt_name=$(basename "$checkpoint" .ckpt)
  job_name="inf_${ckpt_name}"
  csv_path="${OUTPUT_DIR}/${ckpt_name}.csv"

  echo "Submitting inference job for ${ckpt_name}..."

  sbatch --job-name="${job_name}" \
         --output="${job_name}.out" \
         --error="${job_name}.err" \
         --time=06:00:00 \
         --partition=titans \
         --gres=gpu:1 \
         --cpus-per-task=4 \
         --mem=16GB \
         --wrap="source /home/ralbe/miniconda3/etc/profile.d/conda.sh && \
                 conda activate mesh_autodecoder && \
                 python ${SCRIPT_PATH} \
                     --checkpoint_path '${checkpoint}' \
                     --output_csv '${csv_path}'"

  job_count=$((job_count + 1))
  sleep 3
done

echo "All ${job_count} inference jobs submitted! Check status with: squeue -u $USER"


