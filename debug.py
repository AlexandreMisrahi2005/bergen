import datasets
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

ds = datasets.load_from_disk("nih_hash")["train"]
for i,sample in enumerate(ds):
    nih_res = re.search(r"\(The magic number is (.*)\)", sample['doc'])
    # if i == 0 and nih_res:
    #     print("first example", nih_res.group(0))
    if not nih_res:
        print(f"No match found for row ID '{sample['qid']}'")
        print(sample['doc'])
        break