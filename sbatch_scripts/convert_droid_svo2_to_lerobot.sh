#!/bin/bash
#SBATCH --job-name=convert_droid_svo2_to_lerobot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=example@princeton.edu
#SBATCH --exclude=neu301,neu309,neu312

# UPDATE: path to your DROID data directory
DATA_DIR=path/to/your/raw/data/directory
# UPDATE: path where the converted LeRobot dataset will be saved
OUTPUT_DIR=/path/to/your/processed/data/directory

# if needed, include a line to cd into the appropriate directory, ex: cd home/path/openpi
mkdir -p slurm_outputs
uv run examples/droid/convert_droid_svo2_to_lerobot.py --data_dir "$DATA_DIR" --output_path "$OUTPUT_DIR"
