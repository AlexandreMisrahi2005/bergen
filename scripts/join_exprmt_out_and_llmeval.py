"""
Given folder name, join LLMeval scores onto eval_dev_out.json (from eval_dev_metrics_LLMeval_llama3.1:70b_out.json)
+ sort by q_id
does not overwrite eval_dev_out.json but writes a new file eval_dev_out_with_llmeval_score.json


Example command to run
```
python3 scripts/join_exprmt_out_and_llmeval.py experiments/FT/nq --subset
```

"""

import os
import json
import argparse
import pandas as pd
import random

def process_folders(args):
    master_folder = args.master_folder
    subset = args.subset
    # Loop through each subfolder in the master folder
    output_excel_file = os.path.join(master_folder, "before_after_FT_subset.xlsx") if subset else os.path.join(args.master_folder, "before_after_FT.xlsx")
    writer = pd.ExcelWriter(output_excel_file, engine='xlsxwriter')
    subset_q_ids = None
    for folder_name in os.listdir(master_folder):
        folder_path = os.path.join(master_folder, folder_name)
        if os.path.isdir(folder_path):
            eval_file = os.path.join(folder_path, 'eval_dev_out.json')
            scoring_file = os.path.join(folder_path, 'eval_dev_metrics_LLMeval_llama3.1:70b_out.json')

            # Check if both files exist
            if os.path.exists(eval_file) and os.path.exists(scoring_file):
                # Load eval_dev_out.json
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)

                # Load scoring file (in JSON lines format)
                with open(scoring_file, 'r') as f:
                    scoring_data = [json.loads(line) for line in f if line.strip()]

                # Create a dictionary mapping questions to scores from the scoring file
                score_map = {item['question']: item['score'] for item in scoring_data}

                # Add score to each item in eval_data
                for item in eval_data:
                    item['score'] = score_map.get(item['question'], 'No score')

                if subset:
                    if subset_q_ids is None:
                        random_subset = random.sample(eval_data, 100)
                        random_subset_sorted = sorted(random_subset, key=lambda x: x['q_id'])
                        subset_q_ids = [item['q_id'] for item in random_subset_sorted]
                    else:
                        random_subset_sorted = sorted([item for item in eval_data if item['q_id'] in subset_q_ids], key=lambda x: x['q_id'])
                    df = pd.json_normalize(random_subset_sorted)
                else:
                    eval_data_sorted = sorted(eval_data, key=lambda x: x['q_id'])
                    df = pd.json_normalize(eval_data_sorted)

                # Create a new sheet with the folder name as the sheet name
                df.to_excel(writer, sheet_name=folder_name, index=False)

                # Access the XlsxWriter workbook and worksheet objects
                workbook  = writer.book
                worksheet = writer.sheets[folder_name[:31]]

                # Define formats for coloring based on score
                green_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                yellow_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})

                # Auto-adjust column widths based on the maximum length of content in each column
                for i, col in enumerate(df.columns):
                    max_len = df[col].astype(str).map(len).max()
                    worksheet.set_column(i, i, max(max_len, len(col)) + 2)

                # Apply conditional formatting to each row based on the 'score' column
                score_column_index = df.columns.get_loc('score')  # Get the index of the score column
                for row_num in range(1, len(df) + 1):  # +1 because Excel rows are 1-indexed
                    score = df.iloc[row_num - 1, score_column_index]  # Get the score value for this row

                    # Apply color formatting based on score
                    if score == 1.0:
                        worksheet.set_row(row_num, cell_format=green_format)
                    elif score == 0.0:
                        worksheet.set_row(row_num, cell_format=red_format)
                    elif score == 0.5:
                        worksheet.set_row(row_num, cell_format=yellow_format)

                print(f"Added sheet for folder: {folder_name}")

            else:
                print(f'Skipping folder {folder_name}, files not found')
        
    # Save the Excel file
    writer.close()
    print(f"Excel file saved as: {output_excel_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process folders to merge eval_dev_out.json with scores.')
    parser.add_argument('master_folder', type=str, help='Path to the master folder containing subfolders with JSON files')
    parser.add_argument('--subset', default=False, action='store_true', help='Subset outputs (useful if eval set large)')
    
    args = parser.parse_args()
    
    process_folders(args)
