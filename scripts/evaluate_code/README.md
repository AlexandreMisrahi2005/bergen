# Automatic Code Evaluation

- Example to run ACE:

```
cd bergen
python3 scripts/evaluate_code/evaluate_functional_correctness.py <path_to_LLM_output>
```

- If you want to run an ACE sanity check on your dataset labels:

IMPORTANT: first, modify path to dataset in file ```scripts/evaluate_code/data.py``` > ```HUMAN_EVAL = ...```
and modify the ```read_problems()``` method so that it returns the dataset in the required format ({row['id']: row for row in dataset}). Then run the script with ```labels-check``` argument.
```
cd bergen
python3 scripts/evaluate_code/evaluate_functional_correctness.py <path_to_LLM_output> --labels-check
```
 If the dataset labels and the parsing of the labels into a code snippet are correct, then pass@1 should be 1. 

 - Output: 

 The matching patterns to find the code in LLM output is pretty basic, it is done in ```evaluation.py > match_code()```
 Don't hesitate to modify it to match your cases. If no code is found, the default behavior is to return the entire LLM output and print the id of the problematic rows. Then, a summary of the number of correct generations will be printed (grouped by matched code pattern). Finally, a summary of automatic code evaluations will be printed; for example ```{'syntax': 31, 'unit tests': 53, 'other': 22, 'passed': 58}``` means for HumanEval dataset there were 31 syntax errors (either due to error by generator or max generation length was reached), 53 unit tests errors (the function was syntactically correct by semantically incorrect), 22 other errors (like ValueError, IndexError, etc...), and 58 generations were correct. 

 If the code is ran with $n > k$ generations per query, the behavior should be that each generations is counted as a dataset row (so for example there could be 2 queries, 4 generations with 2 per query, but pass@1=1 because for each query there is 1 correct and 1 incorrect generation). But evaluating with multiple generations was not extensively tested.  