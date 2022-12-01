#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J lms_inf
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
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
#BSUB -o /work1/patmjen/meshfit/inference/local_mod_siren/batch_output/inf_%J.out
#BSUB -e /work1/patmjen/meshfit/inference/local_mod_siren/batch_output/inf_%J.err
# -- end of LSF options --

source init.sh

export CUDA_VISIBLE_DEVICES=0

THIS_JOBID=${LSB_JOBID:-NOID}

CHK_JOBID=13555131
CHK_TRIAL=0
CHK_PATH=/work1/patmjen/meshfit/experiments/siren/lms_${CHK_JOBID}/trial_${CHK_TRIAL}/

# Create experiment and trial directory
INF_DIR=/work1/patmjen/meshfit/inference/local_mod_siren/${CHK_JOBID}_${CHK_TRIAL}/${THIS_JOBID}/
mkdir --parents ${INF_DIR}

python -u inference_local_mod_siren.py \
    --data_path="/work1/patmjen/meshfit/datasets/shapes/liver/raw/" \
    --checkpoint_dir=${CHK_PATH} \
    --checkpoint_jobid=${CHK_JOBID} \
    --output_dir=${INF_DIR} \
    --train_test_split_idx=-40 \
    --sample_mode="uniform" \
    --num_point_samples=5000 \
    #--remesh_at_end \

