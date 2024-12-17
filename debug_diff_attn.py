import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from peft import LoraConfig
# from torchviz import make_dot
from modules.dataset import Tokenized_Sorted_Dataset
from models.generators.DiffTransformer.LlamaDiffAttention import LlamaLoraDiffAttention
from modules.dataset_processor import ProcessDatasets
from modules.generators.llm_diff_transformer import LLM
from utils import prepare_dataset_from_ids

### LOAD DATASET
dataset_config = {"train": {"doc": {"init_args": {"_target_": "modules.dataset_processor.KILT100w", "split": "full"}},
                            "query": {"init_args": {"_target_": "modules.processors.kilt_dataset_processor.KILTNQ_Reformulated",
                                                    "path": "datasets/kilt_nq_RF_short_oLlama3_8b_train",
                                                    "split": "train"}}},
                  "dev": {"doc": {"init_args": {"_target_": "modules.dataset_processor.KILT100w", "split": "full"}},
                          "query": {"init_args": {"_target_": "modules.processors.kilt_dataset_processor.KILTNQ_Reformulated",
                                                  "path": "datasets/kilt_nq_RF_short_oLlama3_8b_dev",
                                                  "split": "validation"}}},
                  "test": {"doc": None, "query": None}}
datasets = ProcessDatasets.process(
        dataset_config, 
        out_folder="datasets/", 
        num_proc=40,
        overwrite=False,
        debug=True
        )

### LOAD MODELS
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# if "A100" not in torch.cuda.get_device_name(torch.cuda.current_device):
#     attn_implementation="sdpa"
# else:
#     attn_implementation="flash_attention_2"
attn_implementation = "eager"
diff_attn_init_with_base_weights = True
diff_attn_lambda = 0
# path = "experiments/test_diff_attn/train_LoraDiffAtt_NQ_llama3_8b_instruct_basicprompt_lambda00_base_init_alllayers_lora_mlp/train/checkpoint-2704"
path = None

tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
tokenizer.padding_side = "left"
if tokenizer.bos_token is not None:
    tokenizer.pad_token = tokenizer.bos_token
elif tokenizer.pad_token is not None:
    tokenizer.pad_token = tokenizer.pad_token
else:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='bfloat16',
)

### Load standard llama model in 4bit
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # quantization_config=quant_config,
#     attn_implementation=attn_implementation,
#     torch_dtype=torch.bfloat16,
#     device_map='auto',
# )

### Load standard llama model in 4bit and modify first attention layer

  _target_: models.generators.llm_diff_transformer.LLM
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  max_new_tokens: 128
  max_length: 2048
  batch_size: 32
  quantization: null
  path: null
  diff_attn_init_with_base_weights: True
  diff_attn_lambda: 0.001
  layers_to_transform: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
  verbose: True

generator = LLM( 
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    batch_size=32, 
    max_new_tokens=128, 
    max_length=2048,
    prompt: str = None,
    quantization: str = None,
    attn_implementation: str = "eager",
    path: str = None, # path to a local checkpoint
    diff_attn_init_with_base_weights: bool = True,
    diff_attn_lambda: float = 0.0,
    layers_to_transform: List = list(range(0,32)),
    lora_negative_term_only: bool = False,
    verbose: bool = False,
)

lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj"],
        lora_alpha=64,
        r=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(0,32)),
        )

diff_attn_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quant_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# freeze all model parameters (including embedding and LM head)
for param in diff_attn_model.parameters():
    param.requires_grad = False

for i,layer in enumerate(diff_attn_model.model.layers):
    if isinstance(layer.self_attn, LlamaAttention) and i in lora_config.layers_to_transform:
        if diff_attn_init_with_base_weights:
            print("Copying base weights for layer ", i)
            q_proj = layer.self_attn.q_proj.weight.data.clone()
            k_proj = layer.self_attn.k_proj.weight.data.clone()
            v_proj = layer.self_attn.v_proj.weight.data.clone()
            o_proj = layer.self_attn.o_proj.weight.data.clone()
        layer.self_attn = LlamaLoraDiffAttention(
            diff_attn_model.config, 
            layer_idx=i, 
            lora_config=lora_config, 
            lambda_init_fn=lambda _: diff_attn_lambda
        ).to(diff_attn_model.device)
        if diff_attn_init_with_base_weights:
            layer.self_attn.q_proj.weight.data.copy_(q_proj)
            layer.self_attn.k_proj.weight.data.copy_(k_proj)
            layer.self_attn.v_proj.weight.data.copy_(v_proj)
            layer.self_attn.o_proj.weight.data.copy_(o_proj)
        if path: # load weights from a checkpoint
            print("Loading weights for layer ", i, " from ", path)
            layer.self_attn.load_weights(diff_attn_model.model.layers[i])
    else: # freeze model parameters
        for param in layer.parameters():
            param.requires_grad = False

# base_model = base_model.bfloat16()
# base_model.eval()
# base_model.config.pretraining_tp = 1


### TRAIN
dataset_split = 'train'
dataset = datasets[dataset_split] 
query_dataset_name = dataset['query'].name
doc_dataset_name = dataset['doc'].name
        
query_ids, doc_ids = None, None

# prepare dataset
gen_dataset = prepare_dataset_from_ids(
    dataset, 
    query_ids, 
    doc_ids, 
    multi_doc=True, 
    )
        
# split train into train and test
test_size = min(len(gen_dataset)//2, 64)
    
train_test_datasets = gen_dataset.train_test_split(test_size, seed=42)

print("Preprocessing data...")
train_test_datasets['train'] = Tokenized_Sorted_Dataset(train_test_datasets['train'], self.generator, training=True)
train_test_datasets['test'] = Tokenized_Sorted_Dataset(train_test_datasets['test'], self.generator, training=True)

# Switch back the model to 'train' mode:
diff_attn_model.train()
gradient_ckpt_enabled = False
if getattr(self.training_config, 'gradient_checkpointing', None):            
    print('Enabling checkpointing')
    try:
        # Attempt to enable gradient checkpointing
        self.generator.model.gradient_checkpointing_enable()
        gradient_ckpt_enabled = True
        print("Gradient checkpointing enabled.")
    except AttributeError:
        # If gradient checkpointing is not supported, catch the AttributeError
        print("Warning: Model does not support gradient checkpointing. Continuing without it.")
    except Exception as e:
        # Catch any other unexpected exceptions and print the error
        print(f"Warning: An error occurred while enabling gradient checkpointing: {e}")
                
        print("Data preprocessed")
        # if lora in train config
        if 'lora' in self.training_config:
            self.generator.model = prepare_model_for_kbit_training(self.generator.model)
            print("using lora training")
            # lora config
            target_modules = list(self.training_config.lora.target_modules) if isinstance(self.training_config.lora.target_modules, ListConfig) else self.training_config.lora.target_modules
            self.training_config.lora.__delattr__('target_modules')
            lora_config = LoraConfig(
                target_modules=target_modules,
                **self.training_config.lora,
                )
            # get adapter
            self.generator.model = get_peft_model(self.generator.model, lora_config)
            print(self.generator.model)
            self.generator.model.print_trainable_parameters()
            self.generator.model = self.generator.model.bfloat16()

        total_batch_size = self.training_config.trainer.per_device_train_batch_size * torch.cuda.device_count()
        total_steps = self.training_config.trainer.num_train_epochs * (len(train_test_datasets['train']) // total_batch_size) // self.training_config.trainer.gradient_accumulation_steps
        num_saving_steps = self.training_config.num_saving_steps
        eval_steps =  max(total_steps// num_saving_steps, 1)
        save_steps = max(total_steps  // num_saving_steps, 1)
        logging_steps = max(total_steps // num_saving_steps, 1)
        print(f"Total steps: {total_steps}, eval steps: {eval_steps}, save steps: {save_steps}, logging steps: {logging_steps}")

        args = TrainingArguments(
            run_name=self.run_name,
            output_dir=f'{self.experiment_folder}/train/',
            **self.training_config.trainer,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            remove_unused_columns=False,
        )
        
        self.generator.model = self.generator.model.bfloat16()

        trainer = Trainer(
            model=self.generator.model,
            args=args,
            data_collator=self.generator.collate_fn,
            train_dataset=train_test_datasets['train'],
            eval_dataset=train_test_datasets['test'],
        )
        print("trainer evaluate =")
        print(trainer.evaluate())
        trainer.train(resume_from_checkpoint=self.training_config.resume_from_checkpoint)
        self.generator.model = trainer.model


### EVAL

diff_attn_model = diff_attn_model.bfloat16()
diff_attn_model.eval()
diff_attn_model.config.pretraining_tp = 1

## make a test example
chat1 = [
    # {"role": "system", "content": "You are an AI assistant. Answer short questions with short answers."},
    {"role": "user", "content": "France capital?"}
]
# chat2 = [
#     # {"role": "system", "content": "You are an AI assistant. Answer short questions with short answers."},
#     {"role": "user", "content": "Who are you?"}
# ]
text1 = tokenizer.apply_chat_template(chat1, tokenize=False, add_generation_prompt=True)
# text2 = tokenizer.apply_chat_template(chat2, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text1], return_tensors="pt", padding=True).to(diff_attn_model.device)

# print("BASE MODEL")
# print(base_model.model.layers[0].self_attn.q_proj.weight.min(), base_model.model.layers[0].self_attn.q_proj.weight.max())
# print(base_model.model.layers[0].self_attn.k_proj.weight.min(), base_model.model.layers[0].self_attn.k_proj.weight.max())
# output1 = base_model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=False)
# print(tokenizer.decode(output1[0]))
print("\nDIFF ATTN MODEL")
print(diff_attn_model)
# print(diff_attn_model.model.layers[0].self_attn.q_proj.weight.min(), diff_attn_model.model.layers[0].self_attn.q_proj.weight.max())
# print(diff_attn_model.model.layers[0].self_attn.k_proj.weight.min(), diff_attn_model.model.layers[0].self_attn.k_proj.weight.max())
output2 = diff_attn_model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=False, temperature=0, top_p=1)
print(tokenizer.decode(output2[0]))
