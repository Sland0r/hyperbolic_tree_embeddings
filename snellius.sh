#!/bin/bash
#SBATCH --job-name=create_embeddings
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/create_embeddings_%j.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate hyperbolic_embeddings

python create_embeddings.py --dataset fma_metadata --graph-name fma_metadata --root 0 --method constructive --embedding-dim 2 --tau 1.0 --terms 1 --dtype float64 --gen-type optim
