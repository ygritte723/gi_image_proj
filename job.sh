#!/bin/bash
# 
#SBATCH -p BatComputer
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -A bio170034p
#SBATCH -t 48:00:00 # HH:MM:SS
#SBATCH -e %j-errfile.out
source activate fewshot
# export WANDB_API_KEY=425c813e4ad3283798084d341b069aad7184735b
PYTHONPATH=/jet/home/lisun/work/xinliu/fewshot/Renet-pytorch
python train.py -batch 128 -dataset isic -gpu 0 -extra_dir sep_17_23 -temperature_attn 5.0 -lamb 0.25 -way 2 -shot 5 -query 15 -milestones 40 50 -max_epoch 60
