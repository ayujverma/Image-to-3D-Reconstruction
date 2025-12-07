#!/bin/bash
#SBATCH -J img2voxel                               # Job name
#SBATCH -o img2voxel/logs/slurm_%j.out             # Stdout output file
#SBATCH -e img2voxel/logs/slurm_%j.err             # Stderr error file
#SBATCH -p gpu-a100                                 # GPU partition
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
mkdir -p img2voxel/logs

# ------------------------------
# Environment Setup
# ------------------------------
module purge
module load cuda/12.2

# Load conda
source /work/10692/ayuj/miniconda3/etc/profile.d/conda.sh
conda activate Image-To-3D

export CUDA_VISIBLE_DEVICES=0,1,2

# ------------------------------
# Job Info
# ------------------------------
echo "==============================================================================="        
echo "IMAGE2VOXEL TRAINING JOB"
echo "Started: $(date)"
echo "Node: $(hostname)"

echo "CUDA version:"
nvcc --version

echo "==============================================================================="        

# ------------------------------
# Run Training
# ------------------------------
bash ./scripts/train_img2voxel.sh

echo "Completed: $(date)"
