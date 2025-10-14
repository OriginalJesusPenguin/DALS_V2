#!/bin/bash

# ConvNet models
CONV_NET_MODELS=('ResUNet' 'VNet' 'DynUNet' 'UNETR')

# Submit one job per ConvNet model
for MODEL in "${CONV_NET_MODELS[@]}"; do
    echo "Submitting job for $MODEL..."
    
    # Create a temporary script for this model
    TEMP_SCRIPT="/tmp/train_${MODEL}_$$.sh"
    
    cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name="${MODEL}"
#SBATCH --output="${MODEL}.out"
#SBATCH --error="${MODEL}.err"
#SBATCH --time=00:05:00
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --nodelist=comp-gpu05


# Activate conda environment
source /home/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate mesh_autodecoder

# Change to project directory
cd /home/ralbe/DALS/mesh_autodecoder

# Run training
python train_segment.py \\
    --train_data_path /scratch/ralbe/dals_data/train_data_mixed.pt \\
    --val_data_path /scratch/ralbe/dals_data/val_data_mixed.pt \\
    conv_net \\
    --num_epochs 100 \\
    --batch_size 2 \\
    --model ${MODEL} \\
    --data_size 192 192 192 \\
    --data_spacing 2.0 2.0 2.0 \\


echo "Training completed for ${MODEL}"
EOF

    # Submit the job
    sbatch "$TEMP_SCRIPT"
    
    # Clean up temporary script
    rm "$TEMP_SCRIPT"
    
    echo "Job submitted for $MODEL"
    sleep 2  # Small delay between submissions
done

echo "All ConvNet training jobs submitted!"
echo "Check job status with: squeue -u ralbe" 