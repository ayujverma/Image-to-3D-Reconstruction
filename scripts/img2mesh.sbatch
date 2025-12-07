#!/bin/bash
#SBATCH -J img2mesh                                # Job name
#SBATCH -o img2mesh/logs/slurm_%j.out              # Stdout output file
#SBATCH -e img2mesh/logs/slurm_%j.err              # Stderr error file
#SBATCH -p gpu-h100                                 # GPU partition
#SBATCH -N 1                                        # Number of nodes
#SBATCH -n 1                                        # Number of tasks
#SBATCH -t 06:00:00                                 # Walltime
#SBATCH -A ASC25079                                 # Project allocation
#SBATCH --mail-type=ALL                             # Job notifications
#SBATCH --mail-user=ayuj@utexas.edu                 # Email

set -e  # Stop on error

# ------------------------------
# Prepare directories BEFORE running anything
# ------------------------------
mkdir -p img2mesh/logs

# ------------------------------
# Environment Setup
# ------------------------------
module purge
module load cuda/12.2

# Load conda
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate Image-To-3D

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

# ------------------------------
# Job Info
# ------------------------------
echo "==============================================================================="        
echo "IMAGE2MESH TRAINING JOB"
echo "Started: $(date)"
echo "Node: $(hostname)"

echo "CUDA version:"
nvcc --version

echo "==============================================================================="        

# ------------------------------
# Run Training
# ------------------------------
bash ./scripts/train_img2mesh.sh

echo "Completed: $(date)"