#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=genoa
#SBATCH --time=00:10:00
#SBATCH --output=logs/verify_h_mds_%j.out
#SBATCH --error=logs/verify_h_mds_%j.err

module load Anaconda3/2025.06-1
source activate hyperbolic_embeddings

python create_embeddings.py --dataset fma_metadata --graph-name fma_metadata --root 0 --method h_mds --embedding-dim 2 --tau 1.0 --terms 1 --dtype float64 --gen-type optim
