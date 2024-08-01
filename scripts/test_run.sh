#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --job-name=bm25_vllm_pubmed_bioasq_notitle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="cuda12"

# source ~/.bashrc
# conda activate bergen

# python3 bergen.py retriever="bm25" reranker="minilm6" 'generator='tinyllama-chat dataset='kilt_nq'
# python3 main.py retriever='repllama-7b' reranker='debertav3' generator='llama-2-7b-chat' generator.batch_size=16 dataset='kilt_nq'
# python3 bergen.py generator='vllm_SOLAR-107B' retriever='bm25' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=try_bioasq
# python3 bergen.py generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=try_bioasq
# python3 bergen.py generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=try_bioasq_ragged
# python3 bergen.py retriever='bm25' generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=try_bioasq_pubmed_ragged
python3 bergen.py retriever='bm25' generator='vllm_SOLAR-107B' dataset='pubmed_bioasq' ++experiments_folder=testbergen +run_name=try_bioasq_pubmed_ragged_notitle
