import datasets
import re

if __name__ == "__main__":
    """
    Sanity check to make sure labels are runnable code (and pass unit tests if there are any)
    """
    dataset = datasets.load_from_disk("datasets/CodeRAGBench_HumanEval_train")
    for row in dataset:
        if row['id'] in ['HumanEval/23', 'HumanEval/83']:
            print(row)
    print(dataset)
    