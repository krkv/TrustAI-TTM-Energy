#!/bin/bash

#SBATCH --job-name="TTM: Finetune T5 model"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000MB
#SBATCH --time=03:00:00

module load any/python/3.9.9
source ~/ttm/venv_energy/bin/activate

srun python ./parsing/t5/start_fine_tuning.py --gin ./parsing/t5/gin_configs/t5-small.gin --dataset "energy"
