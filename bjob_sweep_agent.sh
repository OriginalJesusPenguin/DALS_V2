#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J md_sweep
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu32gb]"
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
find . -type f -name "*.py" -o -name "*.sh" | xargs -i cp --parents "{}" ${EXP_DIR}/source

# Ensure CUDA_VISIBLE_DEVICES is set to something
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

wandb agent patmjen/mesh-decoder/cb8r4a8l

