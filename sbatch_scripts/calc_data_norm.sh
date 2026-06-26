#!/bin/bash
#SBATCH --job-name=calc_norm_droid_pi05_my_experiment
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sh1200@princeton.edu
#SBATCH --exclude=neu301,neu309,neu312

uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune
