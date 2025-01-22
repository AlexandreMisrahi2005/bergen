#!/bin/bash
### add slurm parameters ###

source ~/.bashrc
conda activate bergen


###############################################
############ EVAL Llama EXPS ##################
###############################################

TOPK_DOCS=5 # generation_top_k
eval_dir="experiments/control_nq_llama/llama321binstruct_evaltop5"
mkdir -p "$eval_dir"
python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-32-1b-instruct' dataset='kilt_nq' prompt='basic' ++experiments_folder=$eval_dir +run_name=nq ++generator.init_args.batch_size=16 ++generation_top_k=5 ++generator.init_args.quantization=null; python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-32-1b-instruct' dataset='popqa' prompt='basic' ++experiments_folder=$eval_dir +run_name=popqa ++generator.init_args.batch_size=16 ++generation_top_k=5 ++generator.init_args.quantization=null
python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-32-1b-instruct' dataset='multidomain/RobustQA_Science' prompt='multidomain/RobustQA_Science' ++experiments_folder=$eval_dir +run_name=robustqa_science ++generator.init_args.batch_size=4 ++generator.init_args.quantization=null ++generation_top_k=5 ++generator.init_args.attn_implementation='eager'
# ... add more datasets specifying the dataset, prompt, run_name, and maybe batch size (some datasets like RobustQA_Science and RobustQA_Technology need smaller bs)

# WITH DISTRACTORS
N_DISTRACTORS=4
# just add ++train_with_k_distractors=$N_DISTRACTORS (bad naming here, train_with_k_distractors will also work for evaluation. If you train with distractors, it will also evaluate with distractors on the associated evaluation set)
eval_dir="experiments/control_nq_llama/llama321binstruct_evaltop5_4distractors"
mkdir -p "$eval_dir"
# for example:
python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-32-1b-instruct' dataset='kilt_nq' prompt='basic' ++experiments_folder=$eval_dir +run_name=nq ++generator.init_args.batch_size=16 ++generation_top_k=$TOPK_DOCS ++generator.init_args.quantization=null ++generator.init_args.attn_implementation='eager' ++train_with_k_distractors=$N_DISTRACTORS

# TO DO LLMEVAL
python3 eval.py --experiments_folder "${eval_dir}" --llm vllm_SOLAR-107B --llm_prompt default_multi_qa --llm_batch_size 512



###############################################
############### EVAL LoRA EXPS ################
###############################################
# specify list of experiment directories separated by spaces (each one is 1 training run)
dirs="experiments/llama32_1B_instruct/train_Lora_MultiQA_distillmistral7b_llama321b_instruct_spladeberta_top5_basicprompt experiments/llama32_1B_instruct/train_Lora_r512_MultiQA_distillmistral7b_llama321b_instruct_spladeberta_top5_basicprompt"
TOPK_DOCS=5
for dir in $dirs; do
    # Check if the current item is a directory and starts with "train_" or "IFT_" (you can remove these conditions if you want as long as you are sure of the directories you specified)
    if [[ -d "$dir" && ( $(basename "$dir") == train_* || $(basename "$dir") == IFT* ) ]] && [ -d "$dir/train" ] && [[ -d "$dir/train" ]]; then
        echo "Found train dir: $dir"
        # Take the latest checkpoint (ls -t sorts by time)
        latest_checkpoint=$(ls -t "$dir/train" 2>/dev/null | head -n 1)
        latest_checkpoint="${dir}/train/${latest_checkpoint}"
        echo "Latest checkpoint: $latest_checkpoint"
        eval_dir="${dir}/eval_top${TOPK_DOCS}docs"
        mkdir -p "$eval_dir"
        echo "Created eval dir: $eval_dir"

        # Run eval on llama FT with lora adapters on all domains
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='kilt_nq' prompt='basic' ++experiments_folder=$eval_dir +run_name=nq ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='popqa' prompt='basic' ++experiments_folder=$eval_dir +run_name=popqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/RobustQA_Science' prompt='multidomain/RobustQA_Science' ++experiments_folder=$eval_dir +run_name=robustqa_science ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=4 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/RobustQA_Technology' prompt='multidomain/RobustQA_Technology' ++experiments_folder=$eval_dir +run_name=robustqa_technology ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=8 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/RobustQA_Writing' prompt='multidomain/RobustQA_Writing' ++experiments_folder=$eval_dir +run_name=robustqa_writing ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/SearchQA' prompt='multidomain/SearchQA' ++experiments_folder=$eval_dir +run_name=searchqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/syllabusQA' prompt='multidomain/syllabusqa' ++experiments_folder=$eval_dir +run_name=syllabusqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/techQA' prompt='multidomain/techQA' ++experiments_folder=$eval_dir +run_name=techqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/bioasq12b' prompt='multidomain/bioasq' ++experiments_folder=$eval_dir +run_name=bioasq12b ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/covidQA' prompt='multidomain/covidQA' ++experiments_folder=$eval_dir +run_name=covidqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/FiQA' prompt='multidomain/FiQA' ++experiments_folder=$eval_dir +run_name=fiqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/paraphraseRC' prompt='multidomain/paraphraseRC' ++experiments_folder=$eval_dir +run_name=paraphraserc ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/RobustQA_Lifestyle' prompt='multidomain/RobustQA_Lifestyle' ++experiments_folder=$eval_dir +run_name=robustqa_lifestyle ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct' dataset='multidomain/RobustQA_Recreation' prompt='multidomain/RobustQA_Recreation' ++experiments_folder=$eval_dir +run_name=robustqa_recreation ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generator.init_args.quantization=null ++generation_top_k=$TOPK_DOCS
        
        # NIH tests + attention maps
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct' dataset='nih_hash' ++experiments_folder=$eval_dir +run_name=nih_hash ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.max_doc_len=200 ++generator.init_args.quantization=null
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct' dataset='nih_long_context' ++generator.init_args.model_name=$latest_checkpoint ++experiments_folder=$eval_dir +run_name="nih_long_context" ++generator.init_args.max_doc_len=-1 ++generator.init_args.quantization=null ++generator.init_args.batch_size=4
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct' dataset='nih_long_context_multi_needle' ++generator.init_args.model_name=$latest_checkpoint ++experiments_folder=$eval_dir +run_name="mnih_long_context" ++generator.init_args.max_doc_len=-1 ++generator.init_args.quantization=null ++generator.init_args.batch_size=4
        
        # attention map
        python3 eval.py --folder "${eval_dir}/nih_hash" --llm_att --sample 1

        # LLMeval
        python3 eval.py --experiments_folder "${eval_dir}" --llm vllm_SOLAR-107B --llm_prompt default_multi_qa --llm_batch_size 512
    fi
done
# === end controls === 



#########################################
######## EVAL DIFF ATTN MODELS ##########
#########################################
# the idea is the same but the config changes slightly
# example: one good run
dirs="experiments/llama32_1B_instruct/train_LoraDiffAtt_multiqa_distillmistral7b_spladeberta_top5_0.1_r32_noscalingpostattn_negativetermloraonly"
for dir in $dirs; do
    # Check if the current item is a directory and starts with "train_" or "IFT_" (you can remove these conditions if you want as long as you are sure of the directories you specified)
    if [[ -d "$dir" && ( $(basename "$dir") == train_* || $(basename "$dir") == IFT* ) ]] && [ -d "$dir/train" ]; then
        echo "Found train dir: $dir"
        # Take the latest checkpoint
        latest_checkpoint=$(ls -t "$dir/train" 2>/dev/null | head -n 1)
        latest_checkpoint="${dir}/train/${latest_checkpoint}"
        echo "Latest checkpoint: $latest_checkpoint"
        eval_dir="${dir}/eval"
        mkdir -p "$eval_dir"
        echo "Created eval dir: $eval_dir"

        # Run eval on llama FT with diff adapters

        # general domain
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='kilt_nq' prompt='basic' ++experiments_folder=$eval_dir +run_name=nq ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='popqa' prompt='basic' ++experiments_folder=$eval_dir +run_name=popqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        
        # specialized domains
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/RobustQA_Science' prompt='multidomain/RobustQA_Science' ++experiments_folder=$eval_dir +run_name=robustqa_science ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=4 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/RobustQA_Technology' prompt='multidomain/RobustQA_Technology' ++experiments_folder=$eval_dir +run_name=robustqa_technology ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=8 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/RobustQA_Writing' prompt='multidomain/RobustQA_Writing' ++experiments_folder=$eval_dir +run_name=robustqa_writing ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/SearchQA' prompt='multidomain/SearchQA' ++experiments_folder=$eval_dir +run_name=searchqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/syllabusQA' prompt='multidomain/syllabusqa' ++experiments_folder=$eval_dir +run_name=syllabusqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/techQA' prompt='multidomain/techQA' ++experiments_folder=$eval_dir +run_name=techqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/bioasq12b' prompt='multidomain/bioasq' ++experiments_folder=$eval_dir +run_name=bioasq12b ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/covidQA' prompt='multidomain/covidQA' ++experiments_folder=$eval_dir +run_name=covidqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/FiQA' prompt='multidomain/FiQA' ++experiments_folder=$eval_dir +run_name=fiqa ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/paraphraseRC' prompt='multidomain/paraphraseRC' ++experiments_folder=$eval_dir +run_name=paraphraserc ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/RobustQA_Lifestyle' prompt='multidomain/RobustQA_Lifestyle' ++experiments_folder=$eval_dir +run_name=robustqa_lifestyle ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='llama-3-8b-instruct-diff-transformer' dataset='multidomain/RobustQA_Recreation' prompt='multidomain/RobustQA_Recreation' ++experiments_folder=$eval_dir +run_name=robustqa_recreation ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.batch_size=16 ++generation_top_k=5
        
        # NIH (1st one only used for attention map as the test is maybe too easy, 2nd one is "long context" NIH, 3rd one is multiple-needles long context nih)
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct-diff-transformer' dataset='nih_hash' ++experiments_folder=$eval_dir +run_name=nih_hash ++generator.init_args.model_name=$latest_checkpoint ++generator.init_args.max_doc_len=200 ++generator.init_args.quantization=null ++generator.init_args.attn_implementation='eager'
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct-diff-transformer' dataset='nih_long_context' ++generator.init_args.model_name=$latest_checkpoint ++experiments_folder=$eval_dir +run_name="nih_long_context" ++generator.init_args.max_doc_len=-1 ++generator.init_args.quantization=null ++generator.init_args.attn_implementation='flash_attention_2' ++generator.init_args.batch_size=4
        python3 bergen.py retriever='oracle_provenance' generator='llama-3-8b-instruct-diff-transformer' dataset='nih_long_context_multi_needle' ++generator.init_args.model_name=$latest_checkpoint ++experiments_folder=$eval_dir +run_name="mnih_long_context" ++generator.init_args.max_doc_len=-1 ++generator.init_args.quantization=null ++generator.init_args.attn_implementation='flash_attention_2' ++generator.init_args.batch_size=4
        
        # attention map
        python3 eval.py --folder "${eval_dir}/nih_hash" --llm_att --sample 1
        
        # llmeval
        python3 eval.py --experiments_folder "${eval_dir}" --llm vllm_SOLAR-107B --llm_prompt default_multi_qa --llm_batch_size 512
    fi
done
# === end eval diff attn ===


#######################################
#########  WITH MANY DOCS  ############
#######################################
# same as the above for both lora and difflora, just change the topk_docs
# TOPK_DOCS=10
# and maybe adjust batch size




############################################
######### EVAL WITH DISTRACTORS ############
############################################
TOPK_DOCS=5
N_DISTRACTORS=4
# same as the above but:
# 1. add ++train_with_k_distractors=$N_DISTRACTORS to each command
# 2. maybe change eval_dir="${dir}/eval_top${TOPK_DOCS}docs_${N_DISTRACTORS}distractors" so it's distinguishable
