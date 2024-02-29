#!/bin/bash

#SBATCH --job-name="TTM"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000MB
#SBATCH --time=06:00:00

module load any/python/3.9.9
source ~/ttm/venv_energy/bin/activate

srun python flask_app.py
