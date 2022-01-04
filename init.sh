#!/bin/bash

module load python3/3.8.11
module load numpy/1.21.1-python-3.8.11-openblas-0.3.17
module load scipy/1.6.3-python-3.8.11
module load cuda/11.3

source /work1/patmjen/venvs/mdec/bin/activate

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

