#!/bin/bash

#SBATCH --job-name=VVP
#SBATCH --mail-user=achopra4@uw.edu
#SBATCH --mail-type=ALL

#SBATCH --gres=gpu:1
#SBATCH --account=kurtlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00

#SBATCH --chdir=.
#SBATCH --export=all
#SBATCH --output=/gscratch/kurtlab/vvp/log/output.txt
#SBATCH --error=/gscratch/kurtlab/vvp/log/error.txt

source ~/.bashrc
source activate agam_1

python3 main.py
