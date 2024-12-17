import os
import json
from modules.metrics import RAGMetrics

root_directory = "experiments/experiments_syllabusQA"

def process_json_file_and_recompute_metrics(file_path, metrics_file_path):
    """Modifies the label field in the eval_dev_out.json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Modify the label field to be a list of strings split at whitespace
    print("loaded data...", end='\r')
    for item in data:
        # if isinstance(item['label'], str):
        #     item['label'] = item['label'].split()
        #     assert isinstance(item['label'], list)
        # elif isinstance(item['label'], list):
        #     print(f"File {file_path} was already edited")
        #     return
        
        item['label'] = [" ".join(item['label'])]
        try:
            assert len(item['label']) == 1
        except AssertionError:
            print("Problem after converting to", item['label'], "in file", file_path)

    print("edited file...", end='\r')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("wrote back...", end='\r')

    metrics_out = RAGMetrics.compute(
        predictions=[d["response"] for d in data], 
        references=[d["label"] for d in data], 
        questions=[d["question"] for d in data]
        )
    
    print("recomputed metrics...", end='\r')
    
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=2)

    print("wrote back metrics.", end='\r')

    
def find_and_process_files(directory):
    """Searches for eval_dev_out.json files and processes them."""
    for root, dirs, files in os.walk(directory):
        if 'eval_dev_out.json' in files and 'eval_dev_metrics.json' in files:
            file_path = os.path.join(root, 'eval_dev_out.json')
            metrics_file_path = os.path.join(root, 'eval_dev_metrics.json')
            print(f"Processing file: {file_path}")
            process_json_file_and_recompute_metrics(file_path, metrics_file_path)
    print("Done.")

if __name__ == "__main__":
    find_and_process_files(root_directory)
