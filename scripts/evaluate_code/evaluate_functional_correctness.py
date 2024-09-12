"""
Modified from https://github.com/openai/human-eval/blob/master/human_eval/evaluate_functional_correctness.py
"""

import sys
import argparse

from data import HUMAN_EVAL
from evaluation import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    k: str = "1,5,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    labels_check: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, labels_check)
    print(results)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate LLM outputs"
    )
    parser.add_argument(
        'file_paths',
        type=str,
        nargs='+',
        help='Path(s) to the prediction JSON file(s)'
    )
    parser.add_argument(
        '--labels-check',
        action='store_true',
        help='call argument to run ACE on dataset labels for sanity check'
    )
    return parser.parse_args()

def main():
    """
    Main function to handle command-line arguments and invoke the entry point.
    """
    args = parse_args()
    for file_path in args.file_paths:
        print(file_path)
        entry_point(file_path, labels_check=args.labels_check)
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
python3 scripts/evaluate_code/evaluate_functional_correctness.py\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_commandr35b_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_commandr35b_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_commandr35b_splade_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_mixtral_moe_7b_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_mixtral_moe_7b_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_mixtral_moe_7b_splade_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_genphi3mini_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_genphi3mini_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_genphi3mini_splade_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_solar_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_solar_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_solar_splade_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama2_7b_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama2_7b_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama2_7b_splade_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama3_8b_bge_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama3_8b_retromae_top5/eval_dev_out.json\
    experiments/experiments_coderagbench/humaneval/independent_datastore_gen_vllmllama3_8b_splade_top5/eval_dev_out.json\
    

"""