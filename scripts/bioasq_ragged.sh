#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --job-name=bioasq_ragged
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/bioasq_ragged.out
#SBATCH --error=slurm_output/bioasq_ragged.err

source ~/.bashrc
conda activate bergen
python3 bergen.py generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=bioasq_ragged
