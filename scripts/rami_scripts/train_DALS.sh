#!/bin/bash

# Configuration
# DATA_PATH="/home/ralbe/pyhppc_project/cirr_segm_clean/augmented_meshes"
TRAIN_DATA_PATH="/home/ralbe/pyhppc_project/cirr_segm_clean/healthy_T1_meshes"
VAL_DATA_PATH="/home/ralbe/pyhppc_project/cirr_segm_clean/healthy_T1_meshes"
# 
NUM_VAL_SAMPLES=1
NUM_AUGMENT=0
DATA_RANDOM_SEED=1337
MODEL="mesh_decoder"


# conda a

# Submit one job per decoder model
echo "Submitting job for $MODEL..."
sbatch --job-name="${MODEL}" \
        --output="${MODEL}.out" \
        --error="${MODEL}.err" \
        --time=24:00:00 \
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
                    --num_augment $NUM_AUGMENT \
                    --data_random_seed $DATA_RANDOM_SEED \
                    mesh_decoder \
                    --latent_features 512 \
                    --hidden_features 724 724 362 \
                    --decoder_mode gcnn \
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
echo "Job submitted for $MODEL"


echo "All jobs submitted! Check status with: squeue -u $USER"