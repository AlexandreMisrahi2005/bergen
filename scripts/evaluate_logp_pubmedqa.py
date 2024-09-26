import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from peft import PeftModel

"""
python3 scripts/evaluate_logp_pubmedqa.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --base_model_name "meta-llama/Meta-Llama-3-8B-Instruct"
"""

DEBUG = False

def evaluate_example(inp, options, model, tokenizer, device):
    """Returns the index of the most likely MCQ answer as predicted by the LLM based on log-probs."""

    # Tokenize each question+option_i (3 options each time for pubmedqa)
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
    if DEBUG:
        print("log_probs.shape: ", log_probs.shape)     # (batch_size, seq_length, vocab_size)

    target_ids = inputs.input_ids[:, 1:]
    if DEBUG:
        print("target_ids.shape: ", target_ids.shape)    # (batch_size, seq_length - 1)

    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1) # gets the log probs of the actual sequence tokens
    if DEBUG:
        print("token_log_probs.shape: ", token_log_probs.shape) # (batch_size, seq_length)

    padding_mask = target_ids != tokenizer.pad_token_id
    token_log_probs_masked = token_log_probs * padding_mask

    sentence_log_probs = token_log_probs_masked.sum(dim=1)  # sum the log probs to get sequence log prob
    if DEBUG:
        print("sentence_log_probs.shape: ", sentence_log_probs.shape)   # (batch_size)

    valid_token_count = padding_mask.sum(dim=1)
    normalized_log_probs = sentence_log_probs / valid_token_count.float()

    highest_probable_seq = torch.argmax(normalized_log_probs).item()  # int, index of highest log prob in the batch
    return highest_probable_seq


def main(dataset, model_name, model, tokenizer, device):
    correct = 0
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

    print("inference done")

    accuracy = correct / len(dataset)
    print(model_name)
    print(f"Accuracy: {accuracy:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Load and evaluate a LoRA-finetuned model.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the LoRA-finetuned model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base model (e.g., llama3-8b-instruct). Can be the same as model_name")
    parser.add_argument("--local_path", action="store_true", help="Flag indicating if the model is stored locally.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    dataset = load_dataset("openlifescienceai/pubmedqa", split="test").select(range(10)) if DEBUG else load_dataset("openlifescienceai/pubmedqa", split="test")
    # assert all([dataset[i]["data"]["Options"] == { "A": "yes", "B": "no", "C": "maybe" } for i in range(len(dataset))]) is true
    print("dataset loaded")

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

    print("model loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(dataset, model_name, model, tokenizer, device)
