import os
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
import asyncio
import random
import time
from ollama import AsyncClient
import datasets

import sys
sys.path.append("./")
from modules.retrieve import Retrieve
from modules.dataset_processor import ProcessDatasets
from utils import load_trec, get_ranking_filename, prepare_dataset_from_ids



# model_name = "llama3.1:70b"
model_name = "llama3:latest"
ollama_url = "http://10.57.16.172:11434" # insert your ollama url
default_client = AsyncClient(host=ollama_url)
top_k = 5
dataset_split = "train"
new_dataset_name = f"kilt_nq_RF_oLlama3_8b_{dataset_split}"

# determine nb of parallel requests by using test_multiprocessing_entry_point() and look at optimal n
# + uncomment all "# debug" lines
N_PARALLEL_REQ = 45  # optimal for Llama3.1:70B, NQ without docs
# N_PARALLEL_REQ = 9   # optimal for Llama3.1:70B, NQ with docs

system_prompt = "You will be given a question, along with some retrieved documents that may or may not be relevant, \
and the gold answers to this question. Your task is to reformulate the gold answer into a well-formed sentence with \
reasoning if necessary. Important point: if the retrieved documents cannot fully help determine the golden label, \
you should ignore the retrieved documents and answer as if you already knew the gold answer without the retrieved documents.\n\n\
Importantly, after I give you the documents, question, and gold labels, immediately give the reformulated label, and do not add anything else to the response!"

system_prompt_without_docs = "You will be given a question and the gold answers to this question. Your task is to reformulate the gold answer into a well-formed sentence with \
reasoning if necessary. Importantly, after I give you the question and gold labels, immediately give the reformulated label, and do not add anything else to the response!"


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
        # "doc": None,
        "query": {
            "init_args": {
                "_target_": "modules.processors.kilt_dataset_processor.KILTNQ",
                "split": "validation",
            }
        },
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
    return (sample["q_id"], compiled_prompt)

async def achat(message_tuple, client=default_client, model=model_name, system=system_prompt, verbose=False):
    q_id, message = message_tuple
    if verbose:
        print(message)
    try:
        response = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                options=generation_options,
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        print(f"1 generation timed out for q_id {q_id}")
        response = (q_id, message, None, 0)
    return (q_id, message, response['message']['content'], response['eval_count'])

async def chat_in_parallel(messages, system, client=default_client, model=model_name, debug=False):
    start = time.time()
    tasks = []
    for message in messages:
        tasks.append(achat(message, client, model, system))
    responses = await asyncio.gather(*tasks)
    end = time.time()
    ntokens = sum([nts for (_,_,_,nts) in responses])
    seconds = end - start
    if debug:
        print(f"{ntokens} tokens generated in {seconds:.03f}s → {(ntokens/seconds):.03f} tokens/second")
    return responses, (seconds, ntokens)

async def test_multiprocessing_entry_point(questions, sys):
    if len(questions) > 100:
        print("WARNING: will soon launch more than 100 requests asynchronously...")
    for n in tqdm(range(3, len(questions)+1, 3)):
        random.shuffle(questions)
        print(f"n={n}: ", end='')
        out, _ = await chat_in_parallel(questions[:n], sys, debug=True)
    return out

async def entry_point(gen_dataset, ids, questions, sys, doc, tmp_save_file=f"tmp/{new_dataset_name}", save_file=f"datasets/{new_dataset_name}", save_interval=100):
    responses = []
    total_ntokens, total_seconds = 0, 0

    if os.path.exists(tmp_save_file):
        print(f"Resuming from checkpoint: {tmp_save_file}")
        checkpoint_data = datasets.load_from_disk(tmp_save_file)
        processed_ids = set(checkpoint_data.filter(lambda x: x['label'] is not None)['id'])
        print(f"{len(processed_ids)} / {len(ids)} already processed")
        questions_todo = [q for i,q in enumerate(questions) if ids[i] not in processed_ids]
        responses += [(row['id'],None,row['label'],None) for row in checkpoint_data if row['label'] is not None]
        if len(processed_ids) == len(ids):
            print("All ids processed. Exiting.")
            return None
    else:
        print("no checkpoint found. reformulating for all questions")
        questions_todo = questions

    for n in tqdm_async(range(0, len(questions_todo) + 1, N_PARALLEL_REQ)):
        # print("n = ",n)
        # debug
        # if n == N_PARALLEL_REQ:
        #     raise NotImplementedError("debug")
        response, (ntokens, seconds) = await chat_in_parallel(questions_todo[n:n+N_PARALLEL_REQ], sys)
        total_ntokens += ntokens
        total_seconds += seconds
        responses += response

        if (n // N_PARALLEL_REQ + 1) % save_interval == 0:
            print(f"n = {n}   ||   temp save...")
            save_reformulations(gen_dataset, responses, doc, save_name=tmp_save_file)
            # yield checkpoint_data, responses, False

    print(f"{total_ntokens} tokens generated in {total_seconds:.03f}ms → {((total_ntokens/1000)/total_seconds):.03f} tokens/second")
    print(f"Saving final checkpoint to {save_file}")
    save_reformulations(gen_dataset, responses, doc, save_name=save_file)
    print("Reformulating done.")
    # yield checkpoint_data, responses, True

def save_reformulations(gen_dataset, all_outputs, doc, save_name):
    response_dict = {q_id: reformulation
                     for q_id,_,reformulation,_ in all_outputs}
    print(f"total of {sum([e is None for e in list(response_dict.values())])} generations timed out after 30s.")
    def add_response(row):
        q_id = row['q_id']
        row['RF_label'] = response_dict[q_id] if row['q_id'] in response_dict else None
        return row
    gen_dataset = gen_dataset.map(add_response)
    remove_cols = ["doc", "d_id", "d_idx", "label", "ranking_label"] if doc else ["label", "ranking_label"]
    gen_dataset = gen_dataset.rename_column("q_id", "id").rename_column("query", "content").remove_columns(remove_cols).rename_column("RF_label", "label")
    gen_dataset.save_to_disk(save_name)
    print(f"Saved {len(response_dict)} responses successfully to {save_name}")


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
        
    ids = []
    questions = []
    for i,item in enumerate(tqdm(gen_dataset)):
        ids.append(item['q_id'])
        questions.append(format_question(item))

        # debug
        # if i > 100:
        #     break

    print(f"Starting to reformulate {len(questions)} labels.")

    # debug
    # out = asyncio.run(test_multiprocessing_entry_point(questions, prompt))
    # ids, out = asyncio.run(entry_point(ids, questions, prompt))
    asyncio.run(entry_point(gen_dataset, ids, questions, prompt, doc=True if doc_dataset_name is not None else False, save_interval=100))
    # for tmp_gen_dataset, out, done in asyncio.run(entry_point(gen_dataset, ids, questions, prompt, doc=True if doc_dataset_name is not None else False, save_interval=1)): # debug (save-interval=100)
    #     if done:
    #         print("Reformulating done.")
    #         save_reformulations(gen_dataset, out, doc=True if doc_dataset_name is not None else False, save_name="datasets/kilt_nq_oRF_train")
    #     else:
    #         print("saving checkpoint")
    #         save_reformulations(tmp_gen_dataset, out, doc=True if doc_dataset_name is not None else False, save_name="tmp/kilt_nq_oRF_train")

    # response_dict = {q_id: (message, response_content, eval_count)
    #                  for q_id, message, response_content, eval_count in out}


    # def add_response(row):
    #     q_id = row['q_id']
    #     row['RF_label'] = response_dict[q_id][1] if q_id in response_dict else None
    #     return row

    # gen_dataset_RF = gen_dataset.map(add_response)

    # remove_cols = ["doc", "d_id", "d_idx", "label", "ranking_label"] if doc_dataset_name is not None else ["label", "ranking_label"]
    # gen_dataset_RF = gen_dataset_RF.rename_column("q_id", "id").rename_column("query", "content").remove_columns(remove_cols).rename_column("RF_label", "label")

    # save_name = "datasets/kilt_nq_oRF_train"
    # gen_dataset_RF.save_to_disk(save_name)
    # print(f"Saved new dataset at {save_name} successfully.")

if __name__ == "__main__":
    main()
