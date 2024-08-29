"""
Modified from https://github.com/openai/human-eval/blob/master/human_eval/data.py
"""

from typing import Iterable, Dict
import gzip
import json
import os
import datasets


HUMAN_EVAL = os.path.join("datasets", "CodeRAGBench_HumanEval_train")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {row['id']: row for row in datasets.load_from_disk(evalset_file) if row['id'] in ['HumanEval/23', 'HumanEval/83']}

    # return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_problem_labels(evalset_file: str = HUMAN_EVAL) -> Iterable[Dict]:
    """
    Parses each row of the Dataset at evalset_file path
    """
    dataset = datasets.load_from_disk(evalset_file)
    for row in dataset:
        if row['id'] in ['HumanEval/23', 'HumanEval/83']:
            yield row['label']


def stream_json(filename: str) -> Iterable[Dict]:
    """
    Parses each json element and yields it as a dictionary
    """
    with open(filename, 'r') as f:
        all_predictions = json.load(f)
    for pred in all_predictions:
        yield pred

    # if filename.endswith(".gz"):
    #     with open(filename, "rb") as gzfp:
    #         with gzip.open(gzfp, 'rt') as fp:
    #             for line in fp:
    #                 if any(not x.isspace() for x in line):
    #                     yield json.loads(line)
    # else:
    #     with open(filename, "r") as fp:
    #         for line in fp:
    #             if any(not x.isspace() for x in line):
    #                 yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))