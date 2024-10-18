import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

DEBUG = True

def main(dataset, model, tokenizer, device):
    unanswerable_ids = []
    unanswerable_questions = []
    ambig_ids = []
    ambig_questions = []
    for example in tqdm(dataset):
        inp = f"Question: {example['question']}\n\nIs this question clear on its own, or should the user provide additional context? Answer directly by [yes] or [no]."
        inputs = tokenizer(inp, return_tensors="pt").to(device)
        length = inputs["input_ids"].shape[1]
        generation_output = model.generate(**inputs, max_new_tokens=5, num_return_sequences=1, do_sample=False)[0][length:]
        output = tokenizer.decode(generation_output, skip_special_tokens=True).lower()
        print(inp)
        print(output)
        print()
        if "[yes]" in output or "yes" in output:
            continue
        elif "[no]" in output or "no" in output:
            unanswerable_ids.append(example["id"])
            unanswerable_questions.append(example["question"])
        else:
            ambig_ids.append(example["id"])
            ambig_questions.append(example["question"])

    print("inference done")
    print(len(unanswerable_ids), "unanswerable questions found")
    print(len(ambig_ids), "ambig")
    print(len(dataset), "total questions")
    for i in range(min(5, len(unanswerable_questions))):
        print(unanswerable_ids[i])
        print(unanswerable_questions[i])
        print()

    # write unanswerable ids to some file
    with open("tmp/covidqa_unanswerable_ids.txt", "w") as f:
        for id in unanswerable_ids:
            f.write(str(id) + ",")

if __name__ == "__main__":

    dataset = load_dataset("deepset/covid_qa_deepset")['train'].filter(lambda x: x['document_id'] in [1557,630]) if DEBUG else load_dataset("deepset/covid_qa_deepset")['train']
    print(dataset)

    # int 4 quantization
    quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
    attn_implementation = "flash_attention_2"
    if "A100" not in torch.cuda.get_device_name(torch.cuda.current_device):
        attn_implementation="sdpa"

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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

    main(dataset, model, tokenizer, device)
