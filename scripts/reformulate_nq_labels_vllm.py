"""
script is not well maintained / does not save checkpoints
"""

from tqdm import tqdm
import random
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
import nltk

import sys
sys.path.append("./")
from modules.retrieve import Retrieve
from modules.dataset_processor import ProcessDatasets
from utils import load_trec, get_ranking_filename, prepare_dataset_from_ids
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download

top_k = 5
batch_size = 128
gguf_file = None
tokenizer_id = None

# Llama-3-8b instruct
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Llama-3.1-70b instruct
# model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# # Llama-3.1-70b instruct, quantized with gguf (also need gguf_file)
model_id = "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF"
tokenizer_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
gguf_file = "Meta-Llama-3.1-70B-Instruct-Q4_K_S.gguf"

tokenizer_id = model_id if tokenizer_id is None else tokenizer_id

system_prompt = "You will be given a question, along with some retrieved documents that may or may not be relevant, \
and the gold answers to this question. Your task is to reformulate the gold answer into a well-formed sentence with \
reasoning if necessary. Important point: if the retrieved documents cannot fully help determine the golden label, \
you should ignore the retrieved documents and answer as if you already knew the gold answer without the retrieved documents.\n\n\
Importantly, after I give you the documents, question, and gold labels, immediately give the reformulated label, and do not add anything else to the response!"

system_prompt_without_docs = "You will be given a question and the gold answers to this question. Your task is to reformulate the gold answer into a well-formed sentence with \
reasoning if necessary. Importantly, after I give you the question and gold labels, immediately give the reformulated label, and do not add anything else to the response!"

# reformulate train labels
dataset_split = "train"

# to reformulate labels without docs, just comment out the docs from the dataset config
dataset_config = {
    "train": {
        # "doc": {
        #     "init_args": {
        #         "_target_": "modules.dataset_processor.KILT100w",
        #         "split": "full",
        #     }
        # },
        "query": {
            "init_args": {
                "_target_": "modules.processors.kilt_dataset_processor.KILTNQ",
                "split": "train",
            }
        }
    },
    "dev": {
        "doc": {
            "init_args": {
                "_target_": "modules.dataset_processor.KILT100w",
                "split": "full",
            }
        },
        "query": {
            "init_args": {
                "_target_": "modules.processors.kilt_dataset_processor.KILTNQ",
                "split": "validation",
            }
        }
    },
    "test": {
        "doc": None,
        "query": None,
    }
}

# only used to get the trec file name
retriever_config = {
    "init_args": {
        "_target_": "models.retrievers.bm25.BM25",
        "model_name": "bm25",
    },
    "batch_size": 512, 
    "batch_size_sim": 2048,
}

generation_options = {"temperature": 0}


def format_question(sample):
    if 'doc' in sample:
        docs = ''
        for i, doc in enumerate(sample['doc']):
            doc = ' '.join(doc.split())    # truncate to max_doc_len ???
            docs += f"Document {i+1}: {doc}\n"
        compiled_prompt = f"### Background: \n{docs}\n\n\n### Question: {sample['query']}\n\n\n### Gold answer(s): {', '.join(sample['label'])}"
    else:
        compiled_prompt = f"### Question: {sample['query']}\n\n\n### Gold answer(s): {', '.join(sample['label'])}"
    return compiled_prompt

def save_reformulations(gen_dataset, all_outputs, doc):
    response_dict = {q_id: (reformulation)
                     for q_id, reformulation in all_outputs}

    def add_response(row):
        q_id = row['q_id']
        row['RF_label'] = response_dict[q_id][1] if q_id in response_dict else None
        return row

    gen_dataset = gen_dataset.map(add_response)

    remove_cols = ["doc", "d_id", "d_idx", "label", "ranking_label"] if doc else ["label", "ranking_label"]
    gen_dataset = gen_dataset.rename_column("q_id", "id").rename_column("query", "content").remove_columns(remove_cols).rename_column("RF_label", "label")

    save_name = "datasets/kilt_nq_RF_train"
    gen_dataset.save_to_disk(save_name)

def main():

    datasets = ProcessDatasets.process(
            dataset_config, 
            num_proc=40,
            )

    query_dataset_name = datasets[dataset_split]['query'].name
    if 'doc' in dataset_config[dataset_split]:
        doc_dataset_name = datasets[dataset_split]['doc'].name
    else:
        doc_dataset_name = None

    retriever = Retrieve(
            **retriever_config,
            pyserini_num_threads=20,
            continue_batch=None,
            )

    # assume trec file already exists
    if doc_dataset_name is not None:
        ranking_file = get_ranking_filename(
                "runs/",
                query_dataset_name,
                doc_dataset_name,
                retriever.get_clean_model_name(),
                dataset_split,
                25,  # retrieve top k
                "copy" # query generator get clean model name
            )

        query_ids, doc_ids, _ = load_trec(ranking_file)
        doc_ids = [doc_ids_q[:top_k] for doc_ids_q in doc_ids]
        prompt = system_prompt

    else:
        query_ids, doc_ids = None, None
        prompt = system_prompt_without_docs

    dataset = datasets[dataset_split]

    gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids,
            multi_doc=True, 
            query_field="content",
            )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if gguf_file is not None:
        gguf_model_path = hf_hub_download(repo_id=model_id, filename=gguf_file)
        model = LLM(model=gguf_model_path,tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.9,quantization="gguf",max_model_len=80000)
        # model = LLM(model=model_id,tensor_parallel_size=torch.cuda.device_count(),
        #             gpu_memory_utilization=0.9,gguf_file=gguf_file)
    else:
        model = LLM(model=model_id, tensor_parallel_size=torch.cuda.device_count(), dtype=torch.float16, 
                    gpu_memory_utilization=0.9, max_model_len=2048, enforce_eager=True, kv_cache_dtype="fp8_e5m2")
    

    sampling_params = SamplingParams(
                        max_tokens=256,
                        temperature=0.0,  # Temperature set to 0 for greedy decoding
                        #top_p=1.0,        # Consider all tokens
                        #top_k=-1,         # Consider all tokens
                        #do_sample=False   # Disable sampling
                    )

    prompts = []
    ids = []
    for sample in gen_dataset:
        compiled_prompt = format_question(sample)
        sys_prompt = system_prompt if doc_dataset_name is not None else system_prompt_without_docs
        messages = [
            {"role": "system", "content": sys_prompt}, 
            {"role": "user", "content": compiled_prompt}]
        prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)
        prompts.append(prompt)
        ids.append(sample["q_id"])

    dataset = Dataset.from_dict({"prompts": prompts, "id": ids})
    # debug
    dataset = dataset.select(range(150))
    
    def mycollate(batch):
        return {key:[item[key] for item in batch] for key in batch[0]}
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=mycollate)

    all_outputs = []
    for i_batch, batch in enumerate(tqdm(dataloader, desc='Reformulating...', total=len(dataloader))):
        outputs = model.generate(batch["prompts"], sampling_params)
        for i,output in enumerate(outputs):
            all_outputs.append((batch["id"][i],output.outputs[0].text))

        if i_batch < 1:
            print("Printing one example for this batch...")
            print(f"id = {batch['id'][-1]}  ||    query + gold answer ={batch['prompts'][-1].split('### Question:')[1]}     || reformulation = {outputs[-1].outputs[0].text}")

        if i_batch % 10000 == 0:
            save_reformulations(gen_dataset, all_outputs, doc=True if doc_dataset_name is not None else False)

    save_reformulations(gen_dataset, all_outputs, doc=True if doc_dataset_name is not None else False)

    print("Reformulating done.")

if __name__ == "__main__":
    main()
