#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J da_train
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
#BSUB -o /work1/patmjen/meshfit/experiments/dals_conv/batch_output/train_%J.out
#BSUB -e /work1/patmjen/meshfit/experiments/dals_conv/batch_output/train_%J.err
# -- end of LSF options --

source init.sh

export CUDA_VISIBLE_DEVICES=0

wandb offline

EXP_POSTFIX=${LSB_JOBID:-NOID}

# Create experiment and trial directory
EXP_DIR=/work1/patmjen/meshfit/experiments/dals_conv/dc_${EXP_POSTFIX}
mkdir --parents ${EXP_DIR}
TRIAL_ID=$(ls -1 ${EXP_DIR} | wc -l)
EXP_DIR=${EXP_DIR}/trial_${TRIAL_ID}
mkdir --parents ${EXP_DIR}

    #--train_data_path="/work1/patmjen/meshfit/datasets/CHAOS/liver_training_256.pt" \
    #--val_data_path="/work1/patmjen/meshfit/datasets/CHAOS/liver_testing_256.pt" \
    # --simulate_slice_annot \

python -u train_segment.py \
    --train_data_path="/work1/patmjen/meshfit/datasets/Task03_Liver/liver_data_64.pt" \
    --val_data_path="/work1/patmjen/meshfit/datasets/Task03_Liver/liver_data_64.pt" \
    --train_subset -40 -20 \
    --val_subset -20 -1 \
    --experiment_name="DC${EXP_POSTFIX}/${TRIAL_ID}" \
    --no_wandb \
    --simulate_slice_annot \
    conv_net \
    --model=UNETR \
    --checkpoint_postfix=${EXP_POSTFIX} \
    --checkpoint_dir=${EXP_DIR} \
    --num_epochs=300 \
    --learning_rate=1e-4 \
    --batch_size=8 \
    --data_size 64 64 64 \

