# import datasets
import random
import string
import re
import json

# ds_simple = datasets.load_dataset('naver/bergen_nih_v1', 'simple')['train']
# # print(ds_simple[0])

# ds_number = datasets.load_dataset('naver/bergen_nih_v1', 'number')['train']
# # print(ds_number[0])

# ds_multihop = datasets.load_dataset('naver/bergen_nih_v1', 'multihop')['train']
# # print(ds_multihop[0])

# def generate_hash(length=32):
#     return ''.join(random.choices(string.ascii_letters + string.digits + '-./?,^%', k=length))

# def get_magic_number_for_qid(qid, hash_lookup):
#     if qid not in hash_lookup:
#         hash_lookup[qid] = generate_hash()
#     return hash_lookup[qid]


# def replace_magic_number_pattern(text, qid, hash_lookup):
#     pattern = r'The magic number is \d{2}'
#     matches = re.findall(pattern, text)
    
#     if matches:
#         for match in matches:
#             text = text.replace(match, f"The magic number is {hash_lookup[qid]}")
#     else:
#         print(f"Warning: Pattern not found in text for row ID '{qid}'")
#     return text

# dataset = datasets.load_dataset('naver/bergen_nih_v1', 'number')

# hash_lookup = {}
# processed_splits = {}
# for split, data in dataset.items():
#     for example in data:
#         if example["qid"] == "nih_dataset_v1_nih_number_q_442":
#             print(example)
    
#     processed_data = data.map(
#         lambda example: {
#             "qid": example["qid"],
#             "query": example["query"],
#             "label": get_magic_number_for_qid(example["qid"], hash_lookup),
#             "did": example["did"],
#             "doc": replace_magic_number_pattern(example["doc"], example["qid"], hash_lookup)
#         }
#     )
#     for example in processed_data:
#         assert example["label"] in example["doc"], f"Label not found in document for row ID '{example['qid']}' = '{example['label']}' with doc '{example['doc']}'"
#         if example["qid"] == "nih_dataset_v1_nih_number_q_442":
#             print(example)
#     processed_splits[split] = processed_data

# new_dataset = datasets.DatasetDict(processed_splits)
# print(new_dataset["train"][0])

# new_dataset.save_to_disk("nih_hash")

# ds = datasets.load_from_disk("nih_hash")["train"]
# for i,sample in enumerate(ds):
#     nih_res = re.search(r"\(The magic number is (.*)\)", sample['doc'])
#     # if i == 0 and nih_res:
#     #     print("first example", nih_res.group(0))
#     if not nih_res:
#         print(f"No match found for row ID '{sample['qid']}'")
#         print(sample['doc'])
#         break


### check context lengths of multiqa with 5 documents

from modules.retrieve import Retrieve
from modules.rerank import Rerank
from modules.dataset_processor import ProcessDatasets
from modules.dataset import Tokenized_Sorted_Dataset
from utils import prepare_dataset_from_ids, load_trec, get_index_path, get_ranking_filename, get_reranking_filename


### LOAD DATASET

dataset_config = {"train": {"doc": {"init_args": {"_target_": "modules.dataset_processor.MergedDocDataset",
                                                    "in_dataset_names": ["ms-marco", "kilt-100w"],
                                                    "in_dataset_splits": ["full", "full"],
                                                    "out_dataset_name": "multi_qa_merged_docs",
                                                    "split": "_ms-marco__kilt-100w"}},
                            "query": {"init_args": {"_target_": "modules.processors.multidomain_dataset_processor.MultiQA",
                                                    "split": "train"}}},
                  "dev": {"doc": {"init_args": {"_target_": "modules.dataset_processor.MergedDocDataset",
                                                "in_dataset_names": ["ms-marco", "kilt-100w"],
                                                "in_dataset_splits": ["full", "full"],
                                                "out_dataset_name": "multi_qa_merged_docs",
                                                "split": "_ms-marco__kilt-100w"}},
                          "query": {"init_args": {"_target_": "modules.processors.multidomain_dataset_processor.MultiQA",
                                                  "split": "dev"}}},
                  "test": {"doc": None, "query": None}}

retriever_config = {
    "init_args": {
        "_target_": "models.retrievers.splade.Splade",
        "model_name": "naver/splade-v3",
        "max_len": 512,
    },
    "batch_size": 256,
    "batch_size_sim": 256,
}

reranker_config = {
    "init_args": {
        "_target_": "models.rerankers.crossencoder.CrossEncoder",
        "model_name": "naver/trecdl22-crossencoder-debertav3",
        "max_len": 1024,
    },
    "batch_size": 64,
}

datasets = ProcessDatasets.process(
        dataset_config, 
        out_folder="datasets/", 
        num_proc=1,
        overwrite=False,
        debug=False
        )

retriever = Retrieve(
                    **retriever_config,
                    pyserini_num_threads=1,
                    continue_batch=None,
                    )
# init reranker
reranker = Rerank(
    **reranker_config,
    )

generation_top_k = 5
### TRAIN
dataset_split = 'train'
dataset = datasets[dataset_split] 
query_dataset_name = dataset['query'].name
doc_dataset_name = dataset['doc'].name
        
query_ids, doc_ids = None, None
ranking_file = get_ranking_filename(
            'runs/',
            query_dataset_name,
            doc_dataset_name,
            retriever.get_clean_model_name(),
            dataset_split, 
            25, # retrieve topk
            query_generator_name="copy",
            generation_top_k=generation_top_k,       # used only if train_with_k_distractors > 0
        )
index_folder = "indexes/"
doc_embeds_path = get_index_path(index_folder, doc_dataset_name, retriever.get_clean_model_name(), 'doc')
query_embeds_path = get_index_path(index_folder, query_dataset_name, retriever.get_clean_model_name(), 'query', dataset_split=dataset_split)

query_ids, doc_ids, scores = load_trec(ranking_file)

doc_ids = [doc_ids_q[:25] for doc_ids_q in doc_ids]

reranking_file = get_reranking_filename(
    'runs/',
    query_dataset_name,
    doc_dataset_name,
    dataset_split,
    retriever.get_clean_model_name(),
    25,
    reranker.get_clean_model_name(),
    25,
    "copy",
)
query_ids, doc_ids, scores = load_trec(reranking_file)

doc_ids = [doc_ids_q[:generation_top_k] for doc_ids_q in doc_ids]

# prepare dataset
gen_dataset = prepare_dataset_from_ids(
    dataset, 
    query_ids, 
    doc_ids, 
    multi_doc=True, 
    )

from tqdm import tqdm
context_lengths = [len(doc) for docs in tqdm(gen_dataset['doc']) for doc in docs]

import pandas as pd
import matplotlib.pyplot as plt

df = pd.Series(context_lengths)
print(df.describe())
plt.hist(context_lengths, bins=100)
plt.savefig('figs/multiqa_train_doc_lengths.png')


train_test_datasets = gen_dataset.train_test_split(0.01, seed=42)

train_counts = {
    "squad": 0, 
    "adversarial_qa": 0, 
    "nq_open": 0,
    "hotpotqa": 0,
    "msmarco": 0,
    "triviaqa": 0,
    "freebase_qa": 0,
    "sciq": 0,
    "asqa": 0,
    "wikiqa": 0,
}
test_counts = {
    "squad": 0, 
    "adversarial_qa": 0, 
    "nq_open": 0,
    "hotpotqa": 0,
    "msmarco": 0,
    "triviaqa": 0,
    "freebase_qa": 0,
    "sciq": 0,
    "asqa": 0,
    "wikiqa": 0,
}

for i in tqdm(range(len(train_test_datasets['train']))):
    # get provenance from q_id until digits
    digit_index = re.search(r"\d", train_test_datasets['train'][i]['q_id']).start()
    provenance = train_test_datasets['train'][i]['q_id'][:digit_index]
    train_counts[provenance] += 1

for i in tqdm(range(len(train_test_datasets['test']))):
    # get provenance from q_id until digits
    digit_index = re.search(r"\d", train_test_datasets['test'][i]['q_id']).start()
    provenance = train_test_datasets['test'][i]['q_id'][:digit_index]
    test_counts[provenance] += 1

print(train_counts)
print(test_counts)