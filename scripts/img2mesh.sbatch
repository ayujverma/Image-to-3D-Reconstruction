#!/bin/bash
#SBATCH -J img2mesh                 # Job name
#SBATCH -o img2mesh/logs/slurm_%j.out         # Stdout output file
#SBATCH -e img2mesh/logs/slurm_%j.err         # Stderr error file
#SBATCH -p gpu-a100                   # GPU partition (A100 queue)
#SBATCH -N 1                          # Total number of nodes
#SBATCH -n 1                          # Total number of tasks
#SBATCH -t 06:00:00                   # Walltime (hh:mm:ss)
#SBATCH -A ASC25079                   # Project/Allocation name
#SBATCH --mail-type=ALL               # Send email at begin and end
#SBATCH --mail-user=ayuj@utexas.edu  # Email for notifications
# ------------------------------
# Load system modules in correct order
# ------------------------------
module purge
module load gcc/11.2.0
module load python3/3.9.7
module load cuda/12.8

source venv/bin/activate
# Restrict PyTorch to the first GPU
export CUDA_VISIBLE_DEVICES=0
# Print job info
echo "============================================================================"
echo "IMAGE-> MESH TRAINING JOB"
echo "============================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "CUDA Version: $(nvcc --version)"
echo "PyTorch sees GPU: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))')"
# Create output directories

# Run Autoencoder training script
./scripts/train_img2mesh.sh
