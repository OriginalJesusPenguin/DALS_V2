#!/bin/bash

#SBATCH --job-name=mesh_decoder
#SBATCH --output=/home/ralbe/DALS/mesh_autodecoder/scripts/rami_scripts/please.out
#SBATCH --error=/home/ralbe/DALS/mesh_autodecoder/scripts/rami_scripts/please.err
#SBATCH --time=24:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# Simple configuration
TRAIN_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data/train_meshes"
VAL_DATA_PATH="/home/ralbe/DALS/mesh_autodecoder/data/test_meshes"
NUM_VAL_SAMPLES=1
NUM_AUGMENT=0
DATA_RANDOM_SEED=1337

source /home/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate mesh_autodecoder

python /home/ralbe/DALS/mesh_autodecoder/train.py \
    --device cuda \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --num_val_samples $NUM_VAL_SAMPLES \
    --num_augment $NUM_AUGMENT \
    --data_random_seed $DATA_RANDOM_SEED \
    mesh_decoder \
    --latent_features 128 \
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
    --lr_reduce_min_lr 1e-5