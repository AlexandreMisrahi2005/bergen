from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffAttention, LlamaLoraDiffTransformerModel, LlamaLoraDiffTransformerForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
import torch
import gc

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
diff_transformer_config_path = "config/generator/llama-3-8b-instruct-diff-transformer.yaml"
attn_implementation = "eager"

# read config with OmegaConf
model_config = OmegaConf.load(diff_transformer_config_path)
diff_transformer_config = OmegaConf.to_container(model_config)
base_config = AutoConfig.from_pretrained(base_model_name)
concat_config = {**base_config.to_dict(), **diff_transformer_config}

# base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_name, 
#         attn_implementation=attn_implementation,
#         torch_dtype=torch.bfloat16,
#         device_map='auto',
#     )

config = LlamaLoraDiffTransformerConfig(**concat_config)
model = LlamaLoraDiffTransformerForCausalLM(config).to("cuda")

# del base_model
# gc.collect()
# torch.cuda.empty_cache()

print(model)

# # print trainable parameters names
print("Layer 0 self-attn trainable params")
for name, param in model.model.layers[0].self_attn.named_parameters():
    if param.requires_grad:
        print(name, param.numel())

# print number of trainable params
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", num_params)

# test inference
input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(model.device)
output = model(input_ids)


