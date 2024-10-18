import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from peft import PeftModel
import csv
import os

"""
For example:
python3 scripts/evaluate_logps.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --base_model_name "meta-llama/Meta-Llama-3-8B-Instruct" --datasets all
"""

DEBUG_EVALUATE = False
DEBUG = False

def evaluate_example(inp, options, model, tokenizer, device):
    """Returns the index of the most likely MCQ answer as predicted by the LLM based on log-probs."""

    texts = []
    for option in options:
        chat = [
            {"role": "user", "content": inp},
            {"role": "assistant", "content": option},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        texts.append(text)

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # get the logits for each sequence in the batch (before softmax = output of the LM head)
        logits = model(**inputs).logits   # shape (batch_size, sequence_length, config.vocab_size)    = scores before softmax

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    if DEBUG_EVALUATE:
        print("log_probs.shape: ", log_probs.shape)     # (batch_size, seq_length, vocab_size)

    target_ids = inputs.input_ids[:, 1:]
    if DEBUG_EVALUATE:
        print("target_ids.shape: ", target_ids.shape)    # (batch_size, seq_length - 1)

    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1) # gets the log probs of the actual sequence tokens
    if DEBUG_EVALUATE:
        print("token_log_probs.shape: ", token_log_probs.shape) # (batch_size, seq_length)

    padding_mask = target_ids != tokenizer.pad_token_id
    token_log_probs_masked = token_log_probs * padding_mask

    sentence_log_probs = token_log_probs_masked.sum(dim=1)  # sum the log probs to get sequence log prob
    if DEBUG_EVALUATE:
        print("sentence_log_probs.shape: ", sentence_log_probs.shape)   # (batch_size)

    valid_token_count = padding_mask.sum(dim=1)
    normalized_log_probs = sentence_log_probs / valid_token_count.float()

    highest_probable_seq = torch.argmax(normalized_log_probs).item()  # int, index of highest log prob in the batch
    return highest_probable_seq


def eval_pubmedqa(dataset, model, tokenizer, device):
    correct = 0
    results = []
    for example in tqdm(dataset):
        inp = ""
        context = example["data"]["Context"]
        for i,ctx in enumerate(context): 
            inp += f"Context {i+1}: {ctx}\n"
        inp += f"\n{example['data']['Question']}\n"
        options = [k+'. '+v for k,v in example['data']['Options'].items()]
        join_options = '\n'.join(options)
        inp += f"\n{join_options}\n"
        predicted_option = evaluate_example(inp, options, model, tokenizer, device)
        correct_option = f"{example['data']['Correct Option']}. {example['data']['Correct Answer']}"
        if options[predicted_option] == correct_option:
            correct += 1
        results.append({"id": example['id'], "predicted": options[predicted_option], "correct": correct_option})
    accuracy = correct / len(dataset)
    print("PUBMED_QA")
    print(f"Accuracy: {accuracy:.4f}")
    return results, accuracy


def eval_hellaswag(dataset, model, tokenizer, device):
    correct = 0
    results = []
    for example in tqdm(dataset):
        context = example["ctx"]
        endings = example["endings"]
        true_label = example["label"]
        predicted_label = evaluate_example(context, endings, model, tokenizer, device) # faster, 1 inference with batched context+endings        
        if predicted_label == int(true_label):
            correct += 1
        results.append({"id": example["ind"], "predicted": endings[predicted_label], "correct": endings[int(true_label)]})
    accuracy = correct / len(dataset)
    print("HELLASWAG")
    print(f"Accuracy: {accuracy:.4f}")
    return results, accuracy


def eval_commonsenseqa(dataset, model, tokenizer, device):
    key2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    correct = 0
    results = []
    for example in tqdm(dataset):
        assert (example["choices"]["label"] == ['A', 'B', 'C', 'D', 'E']), example["id"]
        inp = ""
        inp += f"{example['question']}\n"
        options = [k+'. '+v for k,v in zip(example['choices']['label'], example['choices']['text'])]
        join_options = '\n'.join(options)
        inp += f"\n{join_options}\n"
        predicted_option = evaluate_example(inp, options, model, tokenizer, device)
        correct_option = f"{example['answerKey']}. {example['choices']['text'][key2idx[example['answerKey']]]}"
        if options[predicted_option] == correct_option:
            correct += 1
        results.append({"id": example['id'], "predicted": options[predicted_option], "correct": correct_option})
    accuracy = correct / len(dataset)
    print("COMMONSENSE_QA")
    print(f"Accuracy: {accuracy:.4f}")
    return results, accuracy


def eval_mmlu(dataset, model, tokenizer, device):
    idx2key = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct = 0
    results = []
    correct_per_subject = {subject: 0 for subject in set([example['subject'] for example in dataset])}
    total_per_subject = {subject: 0 for subject in set([example['subject'] for example in dataset])}
    for example in tqdm(dataset):
        inp = ""
        inp += f"{example['question']}\n"
        options = [k+'. '+v for k,v in zip(['A', 'B', 'C', 'D'], example['choices'])]
        join_options = '\n'.join(options)
        inp += f"\n{join_options}\n"
        predicted_option = evaluate_example(inp, options, model, tokenizer, device)
        correct_option = f"{idx2key[example['answer']]}. {example['choices'][example['answer']]}"
        if options[predicted_option] == correct_option:
            correct += 1
            correct_per_subject[example['subject']] += 1
        total_per_subject[example['subject']] += 1
        results.append({"id": example['question'], "predicted": options[predicted_option], "correct": correct_option})
    meta_accuracy = correct / len(dataset)
    print("MMLU")
    print(f"Accuracy: {meta_accuracy:.4f}")
    print("Per subject:")
    for subject in correct_per_subject:
        accuracy = correct_per_subject[subject] / total_per_subject[subject]
        print(f"{subject}: acc={accuracy} || {correct_per_subject[subject]} / {total_per_subject[subject]}")
    return results, meta_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Load and evaluate a LoRA-finetuned model.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the LoRA-finetuned model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base model (e.g., llama3-8b-instruct). Can be the same as model_name")
    parser.add_argument("--local_path", action="store_true", help="Flag indicating if the model is stored locally.")
    parser.add_argument('--datasets', nargs='+', default=[], help="List of datasets to include from [pubmedqa hellaswag commonsenseqa] or all")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    datasets_mappings = {
        "commonsenseqa": {"dataset":None, "evaluator":eval_commonsenseqa},
        "hellaswag": {"dataset":None, "evaluator":eval_hellaswag},
        "pubmedqa": {"dataset":None, "evaluator":eval_pubmedqa},
        "mmlu": {"dataset":None, "evaluator":eval_mmlu},
    }
    if args.datasets == ['all'] or "commonsenseqa" in args.datasets:
        datasets_mappings["commonsenseqa"]["dataset"] = load_dataset("tau/commonsense_qa", split="validation").select(range(10)) if DEBUG else load_dataset("tau/commonsense_qa", split="validation")
    if args.datasets == ['all'] or "hellaswag" in args.datasets:
        datasets_mappings["hellaswag"]["dataset"] = load_dataset("rowan/hellaswag", split="validation", trust_remote_code=True).select(range(10)) if DEBUG else load_dataset("rowan/hellaswag", split="validation", trust_remote_code=True)
    if args.datasets == ['all'] or "pubmedqa" in args.datasets:
        datasets_mappings["pubmedqa"]["dataset"] = load_dataset("openlifescienceai/pubmedqa", split="test").select(range(10)) if DEBUG else load_dataset("openlifescienceai/pubmedqa", split="test")
    if args.datasets == ['all'] or "mmlu" in args.datasets:
        datasets_mappings["mmlu"]["dataset"] = load_dataset("cais/mmlu", "all")["test"].select(range(10)) if DEBUG else load_dataset("cais/mmlu", "all")["test"]

    print("datasets loaded")

    base_model_name = args.base_model_name
    model_name = args.model_name
    
    if args.local_path:
        print(f"Loading model from local path: {model_name}")
    else:
        print(f"Loading model from Hugging Face Hub: {model_name}")

    # int 4 quantization
    quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
    attn_implementation = "flash_attention_2"
    if "A100" not in torch.cuda.get_device_name(torch.cuda.current_device):
        attn_implementation="sdpa"

    if args.local_path:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config, 
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        lora_model_path = model_name
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.eval()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quant_config, 
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model loaded\n")
    for name,ds in datasets_mappings.items():
        if ds["dataset"] is not None:
            print(f"Evaluating {name}...")
            dataset = ds["dataset"]
            evaluator = ds["evaluator"]
            results, acc = evaluator(dataset, model, tokenizer, device)
            os.makedirs("experiments/eval_mcqa_logps/lora_where", exist_ok=True)
            filename = f"experiments/eval_mcqa_logps/lora_where/{model_name.split('/')[2]}--{name}.csv" if args.local_path else f"experiments/eval_mcqa_logps/{model_name.split('/')[1]}--{name}.csv"
            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, results[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(results)
                dict_writer.writerow({"id": "final-accuracy", "predicted": acc, "correct": acc})
            print()
    print("Done.")