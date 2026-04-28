#!/bin/bash
#SBATCH --job-name=sbd_34k
#SBATCH --output=sbd_train_%j.log
#SBATCH --error=sbd_train_%j.err
#SBATCH -A cis260009p
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=GPU-shared

module load anaconda3
module load cuda/12.4.0
conda activate ./sbd_env

echo "Starting SBD 34k training..."
python training/train.py \
    --mode sbd \
    --steps 34000 \
    --seq_len 128 \
    --batch_size 8 \
    --lr 3e-4
echo "Done."