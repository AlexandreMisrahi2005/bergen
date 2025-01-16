'''
Modified from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/needlehaystack/llm_needle_haystack_tester.py
'''

import os
import glob
import random
import string
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map='auto', quantization_config=None)

system_prompt = "You are a helpful AI bot. Give a short answer to the trailing user question by using the given context."
# needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
needle = "\nThe magic number is "
# retrieval_question = "\n\n\nThe question is: what is the best thing to do in San Francisco?"
retrieval_question = "\nWhat is the magic number?\n"
# context_lengths = [10, 100] + list(range(1000, 29001, 2000))
context_lengths = [100, 200, 300, 400, 500]
depth_percents = [0, 20, 40, 60, 80, 100]
haystack_dir = "PaulGrahamEssays"

hash_lookup = {}

def generate_hash(length=32):
    # return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return ''.join(random.choices(string.digits, k=length))

def get_magic_number_for_qid(qid, hash_lookup):
    if qid not in hash_lookup:
        hash_lookup[qid] = needle + generate_hash() + "\n"
    return hash_lookup[qid]

def generate_context(context_length, depth_percent, qid):

    # Get your haystack dir files loaded into a string
    context = read_context_files()

    # Truncate the haystack dir essays to the context length you desire
    context = encode_and_trim(context, context_length)

    # Insert your random statement according to your depth percent
    context = insert_needle(context, depth_percent, context_length, qid)

    return context

def insert_needle(context, depth_percent, context_length, qid):
    needle_with_hash = get_magic_number_for_qid(qid, hash_lookup)
    tokens_needle = tokenizer.encode(needle_with_hash, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 150

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = tokenizer.encode(".", add_special_tokens=False)
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context)
    return new_context

def get_context_length_in_tokens(context):
    return len(tokenizer.encode(context, add_special_tokens=False))

def read_context_files():
    context = ""
    max_context_length = max(context_lengths)
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

    while get_context_length_in_tokens(context) < max_context_length:
        for file in glob.glob(os.path.join(base_dir, haystack_dir, "*.txt")):
            with open(file, 'r') as f:
                context += f.read()
    return context

def encode_and_trim(context, context_length):
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) > context_length:
        context = tokenizer.decode(tokens[:context_length])
    return context

if __name__ == "__main__":
    # generate contexts
    datarows = []
    for depth_percent in tqdm(depth_percents):
        for context_length in context_lengths:
            qid = f"nih_long_context_cl{context_length}_dp{depth_percent}"
            context = generate_context(context_length, depth_percent, qid)
            
            datarow = {
                    "qid": qid,
                    "query": "What is the magic number ?",
                    "label": get_magic_number_for_qid(qid, hash_lookup).replace("\nThe magic number is ", "").strip(),
                    "did": qid,
                    "doc": context,
                }
            datarows.append(datarow)

            # inputs = tokenizer.apply_chat_template(
            #     [
            #         {
            #             "role": "system",
            #             "content": system_prompt,
            #         },
            #         {
            #             "role": "user",
            #             "content": context + retrieval_question,
            #         },
            #     ],
            #     tokenize=False,
            #     add_generation_prompt=True,
            # )
            # inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
            # final_context_length = inputs['input_ids'].shape[1]
            # print(f"Depth percent = {depth_percent} || Context length = {context_length} || Final context length =", final_context_length)


            # outputs = model.generate(**inputs, num_return_sequences=1, do_sample=False, max_new_tokens=80, top_p=None, temperature=None)
            # print("Model output:")
            # print("\t",tokenizer.decode(outputs[0,final_context_length:], skip_special_tokens=False))
            # print("Label: ")
            # print("\t",datarow["label"])
            # print()

    split = {"train": datasets.Dataset.from_list(datarows)}
    new_dataset = datasets.DatasetDict(split)

    new_dataset.save_to_disk("nih_long_context")

    ds = datasets.load_from_disk("nih_long_context")["train"]
    print(ds["qid"])