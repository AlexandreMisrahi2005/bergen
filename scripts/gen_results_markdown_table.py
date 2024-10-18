import os
import json

### Choose dataset directories to report
dataset_dirs = [
    'experiments/bioasq12b',
    'experiments/paraphraserc',
    'experiments/techqa',
    'experiments/syllabusQA',
    'experiments/covidqa',
    'experiments/robustqa_lifestyle',
    'experiments/robustqa_recreation',
    'experiments/robustqa_science',
    'experiments/robustqa_technology',
    'experiments/robustqa_writing',
    'experiments/fiqa',
    'experiments/searchqa',
]

# dataset_dirs = [
#     'experiments/base_or_instruct/bioasq',
# ]

subdirs_of_interest = [sorted(next(os.walk(dataset_dirs[i]))[1]) if os.path.exists(dataset_dirs[i]) else None for i in range(len(dataset_dirs)) ]


output_file = 'tmp/metrics_table.md'
text = ""
def format_metric(value):
    try:
        return '%.3f' % float(value)
    except (ValueError, TypeError):
        return value 

for i,root_dir in enumerate(dataset_dirs):
    if not os.path.exists(root_dir):
        print("Did not find dir", root_dir)
        continue
    text += root_dir + "\n"
    # header
    markdown_table = "| Checkpoint | Match | Recall | LLMeval (Llama3.1-70b)\n"
    markdown_table += "|------------|-----------|--------|-------|\n"
    for subdir in subdirs_of_interest[i]:
        if subdir is None:
            print("Skipping" + subdir)
            continue
        if subdir.startswith('tmp_') or 'tinyllama' in subdir:
            continue
        subdir_path = os.path.join(root_dir, subdir)
        json_file_path = os.path.join(subdir_path, 'eval_dev_metrics.json')

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                metrics = json.load(json_file)
                match = format_metric(metrics.get('M', 'TBD'))
                recall = format_metric(metrics.get('Recall', 'TBD'))
                llmeval = format_metric(metrics.get('LLMeval_llama3.1:70b', 'TBD'))
                markdown_table += f"| {subdir} | {match} | {recall} | {llmeval} |\n"
        else:
            print(f"File not found: {json_file_path}")
            markdown_table += f"| {subdir} | TBD | TBD | TBD |\n"
    text += markdown_table + "\n\n"

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