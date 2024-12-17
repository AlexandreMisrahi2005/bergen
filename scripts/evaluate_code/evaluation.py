"""
Modified from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
"""

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Dict
import itertools

import numpy as np
import tqdm

from data import HUMAN_EVAL, read_problems, stream_problem_labels, stream_json, write_jsonl
from execution import check_correctness

import re

def print_results(regexes_passed, results):

    # Initialize a dictionary to store statistics
    regex_stats = {i: {"passed_count": 0, "total_count": 0} for i in range(20)}
    regex_stats_syntaxerrors = {i: 0 for i in range(20)}
    exec_output_stats = {"syntax": 0, "unit tests": 0, "other": 0, "passed": 0}

    # Combine dictionaries and calculate statistics
    for key in regexes_passed:
        regex_passed = regexes_passed[key][0]
        result = results[key][0][1]["result"]
        
        # Update the statistics
        regex_stats[regex_passed]["total_count"] += 1

        if result.startswith("failed: SyntaxError"):
            exec_output_stats["syntax"] += 1
            regex_stats_syntaxerrors[regex_passed] += 1
        elif result.startswith("failed: AssertionError"):
            exec_output_stats["unit tests"] += 1
        elif result == "passed":
            exec_output_stats["passed"] += 1
            regex_stats[regex_passed]["passed_count"] += 1
        else:
            exec_output_stats["other"] += 1

    # Print statistics
    print()
    for value, counts in regex_stats.items():
        total = counts["total_count"]
        true_count = counts["passed_count"]
        if total > 0:  # Only print if there is at least one boolean
            print(f"Regex pattern {value}:      {true_count} / {total} generations are correct       {regex_stats_syntaxerrors[value]} syntax errors occured")
    print()
    print("Error types:")
    print(exec_output_stats)
    print()


def match_code(text: str, problem: Dict):
    """
    Given text (LLM answer), find the code in the LLM answer. 
    The returned code should be self-sufficient and runnable by itself.

    :param problem: used to get the entry point (i.e., the signature of the target function)
    """

    ### Add as many regex as needed and add name in regexs variable

    # regex1 = re.compile(rf"```python\n"
    #                     rf"def {problem['entry_point']}(?:.*?):\n"
    #                     rf"    \"\"\"(?:.*?)\"\"\"\n"
    #                     rf"(.*?)", re.DOTALL
    #                         )

    # # regex2 = re.compile(rf"...")

    # regexs = [regex1]

    # for i,regex in enumerate(regexs):
    #     match_completion = regex.findall(text)
    #     if len(match_completion) >= 1:
    #         found_completion = match_completion[0]
    #         print(problem['entry_point'])
    #         print(regex.pattern)
    #         print("found completion", match_completion)
    #         print("from text =", repr(text))
    #         return i+1,found_completion
    # found_completion = '    pass\n'   # if nothing is found treat the prediction as an undefined function
    # return 0,found_completion



    # function_name = problem['entry_point']

    # pattern = rf"((?:from\s+\S+\s+import\s+\S+|import\s+\S+)[\s\S]*?)\n+def {function_name}\(.*\):\s*(\n(    .*)*)"

    # # Search for the pattern in the text
    # match = re.search(pattern, text)

    # if match:
    #     # Return the matched imports and the full function definition
    #     return 0, match.group(0)
    # else:
    #     if text.strip().startswith("return"):
    #         return 2, problem['content'] + '\n    ' + text.strip()
    #     return 1, text


    if "```python\n" in text:
        if "```" in text.split("```python\n")[1]:
            # "```python def ret_None():\n    return None```"
            assert isinstance(text, str), text
            return 0, text.split("```python\n")[1].split("```")[0]
        # "```python def ret_None():\n    return None"
        assert isinstance(text, str), text
        return 2, text.split("```python\n")[1]
    
    elif "```Python\n" in text and "```" in text.split("```Python\n")[1]:
        # "```Python def ret_None():\n    return None```"
        assert isinstance(text, str), text
        return 5, text.split("```Python\n")[1].split("```")[0]
    if "```" in text:
        if len(text.split("```")) >= 3:
            # "```code```"
            text = text.split("```")[1]
            if text.strip().startswith("return"):
                # "```\n    return None```"
                text = problem['content'] + '\n    ' + text.strip()
            assert isinstance(text, str), text
            return 4, text
        else:
            assert isinstance(text, str), text
            return 9, text
    else:
        # "    return None"
        if text.startswith("    ") or text.startswith("\n    ") or text.startswith("\n\n    "):
            assert isinstance(text, str), text
            return 12, problem["content"] + text
        if text.strip().startswith("def") or text.strip().startswith("from"):
            if len(text.split("\n\n")) > 2:
                if len(text.split('"""')) >= 3:
                    if "\n\n" in text.split('"""')[2]:
                        text = '"""'.join(text.split('"""')[:2] + ['\n    ' + text.split('"""')[2].strip().split("\n\n")[0]])
                        assert isinstance(text, str), text
                        return 8, text
                    else:
                        text = text
                        assert isinstance(text, str), text
                        return 10, text
                elif "\n\n" in text:
                    # text = text.split("\n\n")[0]
                    # cut the text after the last occurence of "return"
                    # last_return_pos = text.rfind("return")
                    # if last_return_pos != -1:
                    #     text_after_return = text[last_return_pos:]
                    #     if "\n\n" in text_after_return:
                    #         text = text[:last_return_pos] + text_after_return.split("\n\n")[0]
                    #         return 12, text
                    assert isinstance(text, str), text
                    return 11, text
            else:
                text = text.split("\n\n")[0]
            assert isinstance(text, str), text
            return 3, text
        if "from" in text:
            assert isinstance(text, str), text
            return 6, text[text.index("from"):]
        if "def" in text:
            assert isinstance(text, str), text
            return 7, text[text.index("def"):]
        print("[Looking for code in LLM answer...] No code found for task id =", problem["id"])
        assert isinstance(text, str), text
        return 1, text
    
    

    # try:
    #     return 0, text.split("```python\n")[1].split("```")[0]
    # except IndexError:
    #     try:
    #         return 5, text.split("```Python\n")[1].split("```")[0]
    #     except IndexError:
    #         try:
    #             return 2, text.split("```python\n")[1]
    #         except IndexError:
    #             try:
    #                 text = text.split("```\n")[1].split("```")[0]
    #                 if text.strip().startswith("return"):
    #                     text = problem['content'] + '\n    ' + text.strip()
    #                 return 4, text
    #             except IndexError:
    #                 if text.strip().startswith("def") or text.strip().startswith("from"):
    #                     if "\n\n" in text:
    #                         text = text.split
    #                     return 3, text
    #                 if "from" in text:
    #                     return 6, text[text.index("from"):]
    #                 if "def" in text:
    #                     return 7, text[text.index("def"):]
    #                 if text.startswith("    "):
    #                     return 8, problem['content'] + text
    #                 print("[Looking for code in LLM answer...] No code found for task id =", problem["id"])
    #                 return 1, text


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    labels_check: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        prediction_streamer = stream_problem_labels(HUMAN_EVAL) if labels_check else stream_json(sample_file)

        regexes_passed = defaultdict(list)
        matched_completions = defaultdict(list)

        print("Reading samples...")

        for sample in tqdm.tqdm(prediction_streamer):
            task_id = sample["q_id"]
            completion = sample["response"]

            if labels_check:
                regex_passed = 0
                found_completion = completion
            else:
                # Find python code in the LLM output.
                regex_passed,found_completion = match_code(completion, problems[task_id])
            # remember regex passed and code found for debug later on
            regexes_passed[task_id].append(regex_passed)
            matched_completions[task_id].append(found_completion)

            args = (problems[task_id], found_completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."
        assert len(set(completion_id.values())) <= 1, "Some problems have an unequal number of samples."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["q_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    
    # print results per regex passed
    print_results(regexes_passed, results)

    # Finally, save the results in one file:
    prediction_streamer = stream_problem_labels(HUMAN_EVAL) if labels_check else stream_json(sample_file)
    def combine_results():
        for sample in prediction_streamer:
            task_id = sample["q_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            sample["regex_passed"] = regexes_passed[task_id].pop(0)
            sample["found_completion"] = matched_completions[task_id].pop(0)
            yield sample

    out_file = sample_file.rstrip('.json') + "_evalresults_labels.jsonl" if labels_check else sample_file.rstrip('.json') + "_evalresults.jsonl"
    import os
    out_metrics_file = os.path.join(os.path.dirname(sample_file), "eval_dev_metrics.json")
    print(f"Writing results to {out_file} and {out_metrics_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    import json
    with open(out_metrics_file, 'r') as file:
        metrics = json.load(file)
    if labels_check:
        for metric,score in pass_at_k.items():
            metrics[f"{metric}_labels"] = score
    else:
        for metric,score in pass_at_k.items():
            metrics[f"{metric}"] = score
    with open(out_metrics_file, 'w') as file:
        json.dump(metrics, file, indent=2)

    return pass_at_k