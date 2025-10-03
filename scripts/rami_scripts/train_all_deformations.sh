#!/bin/bash

# Configuration
DECODER_MODELS=('local_mesh_decoder' 'mesh_decoder' 'deep_sdf' 'siren' 'siren2' 'local_mod_siren')
DATA_PATH="/home/ralbe/pyhppc_project/cirr_segm_clean/augmented_meshes"
NUM_VAL_SAMPLES=5
NUM_AUGMENT=10
DATA_RANDOM_SEED=1337
PW_NUM_ANCHOR=4
PW_SAMPLE_TYPE="fps"
PW_SIGMA=0.5
PW_R_RANGE=10
PW_S_RANGE=2
PW_T_RANGE=0.25

# Submit one job per decoder model
for model in "${DECODER_MODELS[@]}"; do
    echo "Submitting job for $model..."
    sbatch --job-name="${model}" \
           --output="${model}.out" \
           --error="${model}.err" \
           --time=12:00:00 \
           --partition=titans \
           --gres=gpu:1 \
           --cpus-per-task=4 \
           --mem=16GB \
           --wrap="source /home/ralbe/miniconda3/etc/profile.d/conda.sh && \
                   conda activate mesh_autodecoder && \
                   python /home/ralbe/DALS/mesh_autodecoder/train.py \
                       --device cuda \
                       --data_path '$DATA_PATH' \
                       --num_val_samples $NUM_VAL_SAMPLES \
                       --num_augment $NUM_AUGMENT \
                       --data_random_seed $DATA_RANDOM_SEED \
                       --pw_num_anchor $PW_NUM_ANCHOR \
                       --pw_sample_type '$PW_SAMPLE_TYPE' \
                       --pw_sigma $PW_SIGMA \
                       --pw_r_range $PW_R_RANGE \
                       --pw_s_range $PW_S_RANGE \
                       --pw_t_range $PW_T_RANGE \
                       $model"
    echo "Job submitted for $model"
done

echo "All jobs submitted! Check status with: squeue -u $USER"