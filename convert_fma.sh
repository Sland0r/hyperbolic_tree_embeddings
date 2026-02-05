#!/bin/bash
#SBATCH --job-name=convert_fma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/convert_fma_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate hyperbolic_embeddings

python convert_fma.py
