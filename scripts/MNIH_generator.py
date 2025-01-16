'''
Modified from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/needlehaystack/llm_multi_needle_haystack_tester.py
'''

import os
import glob
import random
import string
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

N_NEEDLES = 5

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map='auto', quantization_config=None)

system_prompt = "You are a helpful AI bot. Give a short answer to the trailing user question by using the given context."
needle = "\nOne of the magic numbers is "
retrieval_question = "\nWhat are all the magic numbers?\n"
# context_lengths = [100] + list(range(1000, 29001, 2000))
context_lengths = [100, 200, 300, 400, 500]
depth_percents = [0, 20, 40, 60, 80, 100] # 0 means evenly distributed, then 50 means evenly distributed starting at the context half point. So increasing depth percent means concentrating the needles together towards the end of the context.
# depth_percents = [0]
haystack_dir = "PaulGrahamEssays"

hash_lookup = {}

def generate_hash(length=8):
    # return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return [''.join(random.choices(string.digits, k=length)) for _ in range(N_NEEDLES)]

def get_magic_number_for_qid(qid, hash_lookup):
    if qid not in hash_lookup:
        needles = generate_hash()
        hash_lookup[qid] = ([h for h in needles], [needle + h + "\n" for h in needles])
    return hash_lookup[qid]

def generate_context(context_length, depth_percent, qid):

    # Get your haystack dir files loaded into a string
    context = read_context_files()

    # Truncate the haystack dir essays to the context length you desire
    context = encode_and_trim(context, context_length)

    # Insert your random statement according to your depth percent
    context = insert_needles(context, depth_percent, context_length, qid)

    return context

def insert_needles(context, depth_percent, context_length, qid):
    """
    Inserts multiple needles (specific facts or pieces of information) into the original context string at 
    designated depth percentages, effectively distributing these needles throughout the context. This method 
    is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
    (haystack) based on the placement depth of these needles.

    The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
    It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
    that the total token count (context plus needles) does not exceed the maximum allowable context length, 
    which might otherwise lead to information being truncated.

    This approach calculates the initial insertion point for the first needle as before but then calculates even 
    spacing for the remaining needles based on the remaining context length. It ensures that needles are 
    distributed as evenly as possible throughout the context after the first insertion. 
    
    Args:
        context (str): The original context string.
        depth_percent (float): The depth percent at which to insert the needles.
        context_length (int): The total length of the context in tokens, adjusted for final buffer.
    
    Returns:
        str: The new context with needles inserted.
    """
    needles_with_hash = get_magic_number_for_qid(qid, hash_lookup)[1]
    tokens_context = tokenizer.encode(context, add_special_tokens=False)
    context_length -= 250

    # Calculate the total length of all needles in tokens
    total_needles_length = sum(len(tokenizer.encode(needle, add_special_tokens=False)) for needle in needles_with_hash)

    # Ensure context length accounts for needles
    if len(tokens_context) + total_needles_length > context_length:
        tokens_context = tokens_context[:context_length - total_needles_length]
    
    # To evenly distribute the needles, we calculate the intervals they need to be inserted.
    depth_percent_interval = (100 - depth_percent) / N_NEEDLES
    
    # Reset the insertion percentages list for the current context
    insertion_percentages = []

    # Insert needles at calculated points
    for needle in needles_with_hash:

        tokens_needle = tokenizer.encode(needle, add_special_tokens=False)

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = tokenizer.encode('.', add_special_tokens=False)
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
                
            # Insert the needle into the context at the found position
            tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

            # Log 
            insertion_percentage = (insertion_point / len(tokens_context)) * 100
            insertion_percentages.append(insertion_percentage)
            # print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
            
            # Adjust depth for next needle
            depth_percent += depth_percent_interval  

    new_context = tokenizer.decode(tokens_context)
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
            qid = f"mnih_long_context_cl{context_length}_dp{depth_percent}"
            context = generate_context(context_length, depth_percent, qid)
            print(f"Depth percent = {depth_percent} || Context length = {context_length}")
            # print(context)
            
            datarow = {
                    "qid": qid,
                    "query": "What are all the magic numbers ?",
                    "label": " ".join(get_magic_number_for_qid(qid, hash_lookup)[0]),
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

    new_dataset.save_to_disk("mnih_long_context")

    ds = datasets.load_from_disk("mnih_long_context")["train"]
    print(ds["qid"])
    print(ds["label"])