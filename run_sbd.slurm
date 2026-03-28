#!/bin/bash
#SBATCH --job-name=sbd_test
#SBATCH --output=sbd_out_%j.txt
#SBATCH --error=sbd_err_%j.txt
#SBATCH -A cis260009p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=GPU-shared

module load anaconda3
conda activate ./sbd_env

echo "Starting SBD script..."
python data/dataset.py