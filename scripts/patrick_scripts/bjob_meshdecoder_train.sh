#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J md_train
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /work1/patmjen/meshfit/experiments/mesh_decoder/batch_output/train_%J.out
#BSUB -e /work1/patmjen/meshfit/experiments/mesh_decoder/batch_output/train_%J.err
# -- end of LSF options --

source init.sh

nvidia-smi

git log -1 --no-color
git --no-pager diff -U1

export WANDB_API_KEY=$(cat ~/WANDB_api_key)

wandb online

EXP_POSTFIX=${LSB_JOBID:-NOID}

# Create experiment and trial directory
EXP_DIR=/work1/patmjen/meshfit/experiments/mesh_decoder/md_${EXP_POSTFIX}
mkdir --parents ${EXP_DIR}
TRIAL_ID=$(ls -1 ${EXP_DIR} | wc -l)
EXP_DIR=${EXP_DIR}/trial_${TRIAL_ID}
mkdir --parents ${EXP_DIR}

# Copy source code to experiment directory to know what was run
mkdir --parents ${EXP_DIR}/source
find . -type f -name "*.py" -o -name "*.sh" -o -name "*.yaml" | xargs -i cp --parents "{}" ${EXP_DIR}/source

# Ensure CUDA_VISIBLE_DEVICES is set to something
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

    # --data_path="/work1/patmjen/meshfit/datasets/shapes/organs/raw/" \

python -u train.py \
    --data_path="/work1/patmjen/meshfit/datasets/shapes/ShapeNetV2/cars" \
    --num_augment=5 \
    --num_val_samples=500 \
    --experiment_name="MD${EXP_POSTFIX}/${TRIAL_ID}" \
    mesh_decoder \
    --checkpoint_postfix=${EXP_POSTFIX} \
    --checkpoint_dir=${EXP_DIR} \
    --decoder_mode="mlp" \
    --encoding="none" \
    --num_epochs=99999 \
    --latent_features=256 \
    --learning_rate_net=2e-3 \
    --learning_rate_lv=2e-3 \
    --lr_reduce_factor=0.5 \
    --lr_reduce_min_lr=1e-5 \
    --weight_normal_loss=1e-4 \
    --weight_edge_loss=0 \
    --weight_laplacian_loss=0 \
    --weight_quality_loss=1e-1 \
    --weight_norm_loss=1e-3 \
    --template_subdiv=4 \
    --hidden_features 512 512 512 256 512 512 512 \
    --concat_latent_at 4 \
    --normalization="layer" \
    --rotate_template \

