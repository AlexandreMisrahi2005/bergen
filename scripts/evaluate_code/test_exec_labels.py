import datasets
import re

if __name__ == "__main__":
    """
    Sanity check to make sure labels are runnable code (and pass unit tests if there are any)
    """
    dataset = datasets.load_from_disk("datasets/NarrativeQA_docs_validation")
    print(dataset)
    print(dataset[0])
    print(dataset[1])
    