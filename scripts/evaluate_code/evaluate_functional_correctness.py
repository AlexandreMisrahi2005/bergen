"""
Modified from https://github.com/openai/human-eval/blob/master/human_eval/evaluate_functional_correctness.py
"""

import fire
import sys

from data import HUMAN_EVAL
from evaluation import evaluate_functional_correctness
from typing import Callable
from data import stream_problem_labels

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    prediction_streamer: Callable = stream_problem_labels,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, prediction_streamer)
    print(results)


def main():
    entry_point("experiments/experiments_coderagbench/test_humaneval/eval_dev_out_sample.json")


sys.exit(main())