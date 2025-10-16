#!/bin/bash

# ConvNet models
CONV_NET_MODELS=('ResUNet' 'VNet' 'DynUNet' 'UNETR' 'UNET' 'SYNVNET3D' 'SEGRESNET')
# CONV_NET_MODELS=('VNet' 'UNETR' 'DynUNet')
# Submit one job per ConvNet model
for MODEL in "${CONV_NET_MODELS[@]}"; do
    echo "Submitting job for $MODEL..."
    
    # Set batch size for 24GB GPUs (RTX A5000 with 24.5GB memory)
    if [[ "$MODEL" == "DynUNet" || "$MODEL" == "SYNVNET3D" ]]; then
        BATCH_SIZE=1  # Increased from 1 due to 24GB memory
    elif [[ "$MODEL" == "UNETR" ]]; then
        BATCH_SIZE=1  # UNETR is memory intensive
    else
        BATCH_SIZE=2  # Increased from 2 due to 24GB memory
    fi
    
    # Create a temporary script for this model
    TEMP_SCRIPT="/tmp/train_${MODEL}_$$.sh"
    
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name="${MODEL}"
#SBATCH --output="${MODEL}.out"
#SBATCH --error="${MODEL}.err"
#SBATCH --time=03:00:00
#SBATCH --partition=titans
#SBATCH --nodelist=comp-gpu14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB


# Activate conda environment
source /home/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate mesh_autodecoder

# Change to project directory
cd /home/ralbe/DALS/mesh_autodecoder

# Run training

# Memory optimization for 24GB GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH="/home/ralbe/DALS/mesh_autodecoder/util/bin:$LD_LIBRARY_PATH"

# Additional optimizations for large memory GPUs
export TORCH_CUDNN_V8_API_ENABLED=1

python train_segment.py \
    --train_data_path /scratch/ralbe/dals_data/train_data_mixed.pt \
    --val_data_path /scratch/ralbe/dals_data/val_data_mixed.pt \
    conv_net \
    --num_epochs 200 \
    --batch_size ${BATCH_SIZE} \
    --model ${MODEL} \
    --data_size 192 192 192 \
    --data_spacing 2.0 2.0 2.0 \


echo "Training completed for ${MODEL}"
EOF

    # Submit the job
    sbatch "$TEMP_SCRIPT"
    
    # Clean up temporary script
    rm "$TEMP_SCRIPT"
    
    echo "Job submitted for $MODEL"
    sleep 0.5  # Small delay between submissions
done

echo "All ConvNet training jobs submitted!"
echo "Check job status with: squeue -u ralbe" 