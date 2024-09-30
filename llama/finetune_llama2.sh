#!/bin/bash

#SBATCH --job-name="ttm-llama2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000MB
#SBATCH --time=08:00:00

module load any/python/3.9.9
source ~/ttm/llama/venv_llama/bin/activate

srun python ~/ttm/llama/llama-recipes/recipes/finetuning/finetuning.py --dataset "custom_dataset" --custom_dataset.file ~/ttm/llama/energy-dataset/energy_dataset.py --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-chat-hf --output_dir ~/ttm/llama/models/output --use_wandb


