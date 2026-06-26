#!/bin/bash
#SBATCH --job-name=finetune_droid_pi05_my_experiment
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=example@princeton.edu
#SBATCH --exclude=neu301,neu309,neu312

# if needed, include a line to cd into the appropriate directory, ex: cd home/path/openpi
source bash_scripts/setup.bash

# update the config name (if changed) and the experiment name 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_finetune --exp-name=pi05_fineune_my_experiment --overwrite
