import os
import json

### Choose dataset directories to report

dataset_dirs = [
    "experiments/tune_diff_attn_lambda",
]

dataset_dirs = [
    # "experiments/tune_diff_attn_nq_lambda_spladeberta",
    "experiments/tune_diff_attn_nq_lambda_spladeberta_nogroupnorm",
    "experiments/tune_diff_attn_nqrfshort_lambda_spladeberta",
]

# dataset_dirs = [
#     "experiments/tune_diff_attn_learnlambda",
#     "experiments/tune_diff_attn_learnlambda_nogroupnorm",
# ]

dataset_dirs = [
    "experiments/tune_diff_attn_learnlambda",
    "experiments/tune_diff_attn_learnlambda_nogroupnorm",
]

LLMEVAL = False

# for each dir in dataset_dirs, take one subdir, add 'eval' and then find all subdirs that contain 'eval_dev_metrics.json'
subdirs_of_interest = []
for root_dir in dataset_dirs:
    if not os.path.isdir(root_dir):
        print(f"Skipping {root_dir}, as it's not a valid directory.")
        continue
    
    # Find the first subdirectory
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        print(f"No subdirectories found in {root_dir}.")
        continue
    
    # Find all sub-subdirectories containing 'eval_dev_metrics.json'
    for subdir in subdirs:
        for subsubdir in os.listdir(os.path.join(root_dir, subdir, 'eval')):
            subsubdir_path = os.path.join(root_dir, subdir, 'eval', subsubdir)
            if os.path.isdir(subsubdir_path) and not subsubdir_path.startswith('tmp_') and 'eval_dev_metrics.json' in os.listdir(subsubdir_path):
                subdirs_of_interest.append(subsubdir_path)

# subdirs_of_interest = [sorted(next(os.walk(dataset_dirs[i]))[1]) if os.path.exists(dataset_dirs[i]) else None for i in range(len(dataset_dirs)) ]

output_file = 'metrics_table.md'
text = "| Checkpoint | Match | Recall | LLMeval (Llama3.1-70b)\n" if LLMEVAL else "| Checkpoint | Match | Recall |\n" 
text += "|------------|-----------|--------|-------|\n" if LLMEVAL else "|------------|-----------|--------|\n"
def format_metric(value):
    try:
        return '%.3f' % float(value)
    except (ValueError, TypeError):
        return value 

for dir_ in sorted(subdirs_of_interest):
    json_file_path = os.path.join(dir_, 'eval_dev_metrics.json')
    name = dir_
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            metrics = json.load(json_file)
            match = format_metric(metrics.get('M', 'TBD'))
            recall = format_metric(metrics.get('Recall', 'TBD'))
            if LLMEVAL:
                llmeval = format_metric(metrics.get('LLMeval_llama3.1:70b', 'TBD'))
            text += f"| {name} | {match} | {recall} | {llmeval} |\n" if LLMEVAL else f"| {name} | {match} | {recall} |\n"
    else:
        print(f"File not found: {json_file_path}")
        text += f"| {name} | TBD | TBD | TBD |\n" if LLMEVAL else f"| {name} | TBD | TBD |\n"

with open(output_file, 'w') as file:
    file.write(text)

print(f"Markdown table written to {output_file}")
























# dataset_dirs = [
#     'experiments/FT/asqa',
#     'experiments/FT/hotpotqa',
#     'experiments/FT/bioasq',
#     'experiments/FT/syllabusqa'
# ]

# FT_NQ_experiments = [
#     'll3_8b_0shot', 
#     'll3_8b_bm25', 
#     'll3_8b_splade',
#     'll3_8b_FT_NQ_1epoch_0shot',
#     'll3_8b_FT_NQ_1epoch_bm25',
#     'll3_8b_FT_NQ_1epoch_splade',
#     'll3_8b_FT_NQ_2epochs_0shot',
#     'll3_8b_FT_NQ_2epochs_bm25',
#     'll3_8b_FT_NQ_2epochs_splade',
# ]

# FT_NQ_RF_labels_experiments = [
#     'll3_8b_0shot', 
#     'll3_8b_bm25', 
#     'll3_8b_splade',
#     'll3_8b_FT_NQ_RF_short_1epoch_0shot',
#     'll3_8b_FT_NQ_RF_short_1epoch_bm25',
#     'll3_8b_FT_NQ_RF_short_1epoch_splade',
#     'll3_8b_FT_NQ_RF_short_2epochs_0shot',
#     'll3_8b_FT_NQ_RF_short_2epochs_bm25',
#     'll3_8b_FT_NQ_RF_short_2epochs_splade',
# ]

# FT_MQA_experiments = [
#     'll3_8b_0shot', 
#     'll3_8b_bm25', 
#     'll3_8b_splade',
#     'll3_8b_FT_MQA_BM25_5K_0shot',
#     'll3_8b_FT_MQA_BM25_5K_bm25',
#     'll3_8b_FT_MQA_BM25_5K_splade',
#     'll3_8b_FT_MQA_splade_5K_0shot',
#     'll3_8b_FT_MQA_splade_5K_bm25',
#     'll3_8b_FT_MQA_splade_5K_splade',
#     'll3_8b_FT_MQA_BM25_11K_0shot',
#     'll3_8b_FT_MQA_BM25_11K_bm25',
#     'll3_8b_FT_MQA_BM25_11K_splade',
#     'll3_8b_FT_MQA_splade_11K_0shot',
#     'll3_8b_FT_MQA_splade_11K_bm25',
#     'll3_8b_FT_MQA_splade_11K_splade',
#     ]

# FT_MQA_with_distractors_experiments = [
#     'll3_8b_0shot', 
#     'll3_8b_bm25', 
#     'll3_8b_splade',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_06_0shot',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_09_0shot',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_06_0shot',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_09_0shot',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_06_bm25',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_09_bm25',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_06_bm25',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_09_bm25',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_06_splade',
#     'll3_8b_FT_MQA_5K_distractors_3_ret_5_P_09_splade',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_06_splade',
#     'll3_8b_FT_MQA_11K_distractors_3_ret_5_P_09_splade',
# ]

# FT_MQA_rf_short_rr = [
#     'll3_8b_splade',
#     'll3_8b_FT_MQA_rf_short_splade_deberta_11K_eval_splade',
#     'll3_8b_FT_MQA_rf_short_splade_deberta_11K_eval_splade_deberta',
# ]