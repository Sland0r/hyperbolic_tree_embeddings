#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=logs/env_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

conda create -n hyperbolic_embeddings python=3.10
source activate hyperbolic_embeddings

pip install -r requirements.txt
