#!/bin/bash

# Job array submission script for parallel inference
# Uses SLURM job arrays for efficient parallel processing

# Create results directory
mkdir -p inference_results

# Check if model list exists
if [ ! -f "inference_models_list.txt" ]; then
    echo "Error: inference_models_list.txt not found. Run 'python list_models_for_inference.py' first."
    exit 1
fi

# Count total models
total_models=$(wc -l < inference_models_list.txt)
echo "Submitting job array for $total_models inference jobs..."

# Create a temporary script for the job array
cat > inference_job_array.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=inference_array
#SBATCH --output=inference_array_%A_%a.out
#SBATCH --error=inference_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB

# Get the checkpoint path for this array task
checkpoint_path=$(sed -n "${SLURM_ARRAY_TASK_ID}p" inference_models_list.txt)

if [ -z "$checkpoint_path" ]; then
    echo "No checkpoint path found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Extract model name for output naming
model_basename=$(basename "$checkpoint_path" .ckpt)
output_csv="inference_results/results_${model_basename}.csv"

echo "Processing array task $SLURM_ARRAY_TASK_ID: $model_basename"

# Run the inference
source /home/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate mesh_autodecoder
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

python /home/ralbe/DALS/mesh_autodecoder/inference_meshdecoder_single.py \
    --checkpoint_path "$checkpoint_path" \
    --output_csv "$output_csv"

echo "Completed array task $SLURM_ARRAY_TASK_ID: $model_basename"
EOF

# Make the script executable
chmod +x inference_job_array.sh

# Submit the job array
sbatch --array=1-${total_models} inference_job_array.sh

echo "Submitted job array with $total_models tasks!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress with: watch -n 5 'squeue -u $USER'"
echo ""
echo "To monitor completion:"
echo "  watch -n 10 'echo \"Running: \$(squeue -u \$USER | grep -c inference_array)\" && echo \"Completed: \$(ls inference_results/results_*.csv 2>/dev/null | wc -l)\"'"
