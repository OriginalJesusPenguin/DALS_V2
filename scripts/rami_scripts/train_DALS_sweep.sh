#!/bin/bash

# Configuration
BASE_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data"

# Define parameter combinations
SPLIT_TYPES=("separate" "mixed")
SCALING_TYPES=("individual" "global")
AUGMENTATION_TYPES=("noaug" "aug")
DECODER_MODES=("gcnn" "mlp")
LATENT_FEATURES=(128 256 512)

# Fixed parameters
NUM_VAL_SAMPLES=1
NUM_AUGMENT=0
DATA_RANDOM_SEED=1337

# Job counter
job_count=0

# Submit jobs for all parameter combinations
for split in "${SPLIT_TYPES[@]}"; do
  for scaling in "${SCALING_TYPES[@]}"; do
    for augmentation in "${AUGMENTATION_TYPES[@]}"; do
      for decoder in "${DECODER_MODES[@]}"; do
        for latent in "${LATENT_FEATURES[@]}"; do
          # Construct paths
          DATA_CONFIG="${split}_${scaling}_${augmentation}"
          TRAIN_DATA_PATH="${BASE_DATA_PATH}/${DATA_CONFIG}/train_meshes"
          VAL_DATA_PATH="${BASE_DATA_PATH}/${DATA_CONFIG}/val_meshes"
          
          # Create job name
          JOB_NAME="${DATA_CONFIG}_${decoder}_${latent}"
          
          # Submit job with current configuration
          echo "Submitting job: $JOB_NAME..."
          sbatch --job-name="${JOB_NAME}" \
                  --output="${JOB_NAME}.out" \
                  --error="${JOB_NAME}.err" \
                  --time=02:00:00 \
                  --partition=titans \
                  --gres=gpu:1 \
                  --cpus-per-task=4 \
                  --mem=4GB \
                  --wrap="source /home/ralbe/miniconda3/etc/profile.d/conda.sh && \
                          conda activate mesh_autodecoder && \
                          python /home/ralbe/DALS/mesh_autodecoder/train.py \
                              --device cuda \
                              --train_data_path '$TRAIN_DATA_PATH' \
                              --val_data_path '$VAL_DATA_PATH' \
                              --num_val_samples $NUM_VAL_SAMPLES \
                              --num_augment 0 \
                              --data_random_seed $DATA_RANDOM_SEED \
                              mesh_decoder \
                              --latent_features $latent \
                              --hidden_features 724 724 362 \
                              --decoder_mode $decoder \
                              --encoding none \
                              --normalization layer \
                              --rotate_template \
                              --num_epochs 9999 \
                              --learning_rate_net 2e-3 \
                              --learning_rate_lv 2e-3 \
                              --weight_norm_loss 1e-4 \
                              --weight_quality_loss 1e-3 \
                              --weight_laplacian_loss 0 \
                              --weight_edge_loss 1e-4 \
                              --template_subdiv 3 \
                              --num_mesh_samples 10000 \
                              --train_batch_size 8 \
                              --batch_size 8 \
                              --lr_reduce_factor 0.5 \
                              --lr_reduce_patience 100 \
                              --lr_reduce_min_lr 1e-5"
          
          job_count=$((job_count + 1))
          sleep 2
        done
      done
    done
  done
done

echo "All $job_count jobs submitted! Check status with: squeue -u $USER"