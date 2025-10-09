#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J md_inf
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
#BSUB -o /work1/patmjen/meshfit/inference/mesh_decoder/batch_output/inf_%J.out
#BSUB -e /work1/patmjen/meshfit/inference/mesh_decoder/batch_output/inf_%J.err
# -- end of LSF options --

source init.sh

export CUDA_VISIBLE_DEVICES=0

THIS_JOBID=${LSB_JOBID:-NOID}

CHK_JOBID=13190113 # Best liver model
#CHK_JOBID=13879213 # Low no. epoch liver model
#CHK_JOBID=13561522 # No BL loss
#CHK_JOBID=13603112 # Spleen
#CHK_JOBID=13879212 # Low no. epoch spleen model

#CHK_JOBID=13875561 # Combined liver and spleen
#CHK_JOBID=13879226 # Combined liver and spleen resumed training from 13875561

CHK_TRIAL=0
CHK_PATH=/work1/patmjen/meshfit/experiments/mesh_decoder/md_${CHK_JOBID}/trial_${CHK_TRIAL}/

# Create experiment and trial directory
INF_DIR=/work1/patmjen/meshfit/inference/mesh_decoder/${CHK_JOBID}_${CHK_TRIAL}/${THIS_JOBID}/
mkdir --parents ${INF_DIR}

    # --weight_edge_length_loss=2e-1 \  # Liver

python -u inference_meshdecoder.py \
    --data_path="/home/ralbe/pyhppc_project/cirr_segm_clean/unit_sphere_meshes_subsample" \
    --output_dir= '/home/ralbe/DALS/mesh_autodecoder/output/deep_sdf/global_local_decoder' \
    --latent_mode='local' \
    --train_test_split_idx=-1 \
    --weight_edge_length_loss=2e-1 \
    --remesh_at_end \
    --num_point_samples=400 \
    --point_sample_mode='fps' \
    --remesh_with_forward_at_end \
    #--lr=1e-3 \
    #--remesh_at 0.5 0.75 \

