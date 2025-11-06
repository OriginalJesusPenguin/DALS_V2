#!/bin/bash

# Fixed paths
TRAIN_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data/train_meshes_aug50/"
VAL_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data/test_meshes"

# Define parameter combinations
DECODER_MODES=("gcnn")
LATENT_FEATURES=(128)
WEIGHT_NORMAL_LOSS=(1e-2 1.5e-2 2e-2 3e-2 5e-2 7.5e-2 1e-1)

# Fixed parameters
NUM_VAL_SAMPLES=1
NUM_AUGMENT=0
DATA_RANDOM_SEED=1337

# Job counter
job_count=0

# Submit jobs for all parameter combinations
for decoder in "${DECODER_MODES[@]}"; do
  for latent in "${LATENT_FEATURES[@]}"; do
    for normal_loss in "${WEIGHT_NORMAL_LOSS[@]}"; do
      # Create job name
      JOB_NAME="aug_${decoder}_${latent}_norm${normal_loss}"
      
      # Submit job with current configuration
      echo "Submitting job: $JOB_NAME..."
      sbatch --job-name="${JOB_NAME}" \
              --output="${JOB_NAME}.out" \
              --error="${JOB_NAME}.err" \
              --time=06:00:00 \
              --partition=titans \
              --gres=gpu:1 \
              --cpus-per-task=4 \
              --mem=16GB \
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
                          --weight_normal_loss $normal_loss \
                          --weight_quality_loss 1e-3 \
                          --weight_edge_loss 1e-2 \
                          --template_subdiv 3 \
                          --num_mesh_samples 2500 \
                          --train_batch_size 8 \
                          --batch_size 8 \
                          --lr_reduce_factor 0.5 \
                          --lr_reduce_patience 50 \
                          --lr_reduce_min_lr 1e-5"
      
      job_count=$((job_count + 1))
      sleep 3
    done
  done
done

echo "All $job_count jobs submitted! Check status with: squeue -u $USER"