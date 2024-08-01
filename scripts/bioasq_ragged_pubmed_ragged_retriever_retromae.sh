#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --job-name=encode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="cuda12"
#SBATCH --output=slurm_output/bioasq_pubmed_ragged_retromae.out
#SBATCH --error=slurm_output/bioasq_pubmed_ragged_retromae.err

source ~/.bashrc
conda activate bergen
python3 bergen.py retriever='retromae' generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=bioasq_pubmed_ragged_retromae
