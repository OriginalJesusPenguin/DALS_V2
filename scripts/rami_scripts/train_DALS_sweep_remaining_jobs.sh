#!/bin/bash

# Resubmit only the 18 combinations that are still failing
BASE_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data"
SPLIT_TYPES=("separate" "mixed")
SCALING_TYPES=("individual" "global")
AUGMENTATION_TYPES=("noaug" "aug")
DECODER_MODES=("gcnn" "mlp")
LATENT_FEATURES=(128 256 512)
NUM_VAL_SAMPLES=1
NUM_AUGMENT=0
DATA_RANDOM_SEED=1337
job_count=0

declare -a MISSING_COMBOS=(
  # Separate dataset failures (12 combinations - all augmentation)
  "separate|global|aug|gcnn|128"
  "separate|global|aug|gcnn|256"
  "separate|global|aug|gcnn|512"
  "separate|global|aug|mlp|128"
  "separate|global|aug|mlp|256"
  "separate|global|aug|mlp|512"
  "separate|individual|aug|gcnn|128"
  "separate|individual|aug|gcnn|256"
  "separate|individual|aug|gcnn|512"
  "separate|individual|aug|mlp|128"
  "separate|individual|aug|mlp|256"
  "separate|individual|aug|mlp|512"
  
  # Mixed dataset failures (6 combinations)
  "mixed|global|noaug|mlp|128"
  "mixed|global|noaug|mlp|256"
  "mixed|global|noaug|mlp|512"
  "mixed|individual|aug|gcnn|128"
  "mixed|individual|aug|gcnn|256"
  "mixed|individual|aug|gcnn|512"
)

is_missing_combo() {
  local key="$1"
  for mc in "${MISSING_COMBOS[@]}"; do
    if [ "$mc" = "$key" ]; then
      return 0
    fi
  done
  return 1
}

# Submit jobs for all parameter combinations
for split in "${SPLIT_TYPES[@]}"; do
  for scaling in "${SCALING_TYPES[@]}"; do
    for augmentation in "${AUGMENTATION_TYPES[@]}"; do
      for decoder in "${DECODER_MODES[@]}"; do
        for latent in "${LATENT_FEATURES[@]}"; do
          combo_key="${split}|${scaling}|${augmentation}|${decoder}|${latent}"
          # Only submit if explicitly marked as missing
          if ! is_missing_combo "$combo_key"; then
            continue
          fi

          # Construct paths
          DATA_CONFIG="${split}_${scaling}_${augmentation}"
          TRAIN_DATA_PATH="${BASE_DATA_PATH}/${DATA_CONFIG}/train_meshes"
          VAL_DATA_PATH="${BASE_DATA_PATH}/${DATA_CONFIG}/val_meshes"
          
          # Create job name
          JOB_NAME="${DATA_CONFIG}_${decoder}_${latent}"
          
          # Submit job with current configuration
          echo "Submitting: $JOB_NAME"
          
          # Resource allocation based on combination type
          if [[ "$split" == "mixed" && "$scaling" == "global" ]]; then
            MEMORY="8GB"; TIME="04:00:00"; BATCH_SIZE="4"
          elif [[ "$augmentation" == "aug" ]]; then
            MEMORY="6GB"; TIME="03:00:00"; BATCH_SIZE="6"
          else
            MEMORY="4GB"; TIME="02:00:00"; BATCH_SIZE="8"
          fi
          
          sbatch --job-name="${JOB_NAME}" \
                  --output="${JOB_NAME}.out" \
                  --error="${JOB_NAME}.err" \
                  --time="$TIME" \
                  --partition=titans \
                  --gres=gpu:1 \
                  --cpus-per-task=4 \
                  --mem="$MEMORY" \
                  --wrap="source /home/ralbe/miniconda3/etc/profile.d/conda.sh && \
                          conda activate mesh_autodecoder && \
                          export CUDA_VISIBLE_DEVICES=0 && \
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
                              --train_batch_size $BATCH_SIZE \
                              --batch_size $BATCH_SIZE \
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

echo "Submitted $job_count jobs. Check status with: squeue -u $USER"