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
        'file_path',
        type=str,
        help='Path to the prediction JSON file'
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
    entry_point(args.file_path, labels_check=args.labels_check)
    return 0

if __name__ == "__main__":
    sys.exit(main())