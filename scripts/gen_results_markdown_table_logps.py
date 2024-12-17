import os
import pandas as pd

root_dir = 'experiments/eval_mcqa_logps'

csv_files = sorted(next(os.walk(root_dir))[2])
csv_files_llama = [file for file in csv_files if 'llama' in file.lower()]
csv_files_gemma = [file for file in csv_files if 'gemma' in file.lower()]
csv_files = [csv_files_gemma, csv_files_llama]

output_file = 'tmp/metrics_table_mcqa_logps.md'
text = ""

def format_metric(value: str):
    try:
        return '%.3f' % float(value)
    except (ValueError, TypeError):
        return value 

for i,subdir in enumerate(csv_files):
    # header
    markdown_table = "| Checkpoint | CommonSenseQA | HellaSwag | MMLU | PubMedQA | Mean |\n"
    markdown_table += "|-|-|-|-|-|-|\n"
    mean = []
    for j,file in enumerate(subdir):
        dataset = file.split('--')[1].split('.')[0]
        if j % 4 == 0:
            checkpoint = file.split('--')[0]
            markdown_table += f"| {checkpoint} |"
            assert dataset == 'commonsenseqa'
        if j % 4 == 1:
            assert dataset == 'hellaswag'
        if j % 4 == 2:
            assert dataset == 'mmlu'
        df = pd.read_csv(os.path.join(root_dir, file)).iloc[-1]
        assert df['id'] == 'final-accuracy' and df['correct'] == df['predicted']
        acc = format_metric(df['predicted'])
        markdown_table += f" {acc} |"
        mean.append(float(acc))
        if j % 4 == 3:
            assert dataset == 'pubmedqa'
            markdown_table += f" {format_metric(sum(mean)/len(mean))} |\n"
            mean = []
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