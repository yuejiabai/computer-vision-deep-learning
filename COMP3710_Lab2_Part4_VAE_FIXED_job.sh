#!/bin/bash
#SBATCH --job-name=p4-vae-fixed
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=p4_vae_fixed_%j.out
#SBATCH --error=p4_vae_fixed_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python COMP3710_Lab2_Part4_VAE_FIXED.py \
  --epochs 30 \
  --batch-size 64 \
  --workers 4 \
  --latent-dim 2 \
  --beta 0.0001 \
  --lr 0.0003
