from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
diff_transformer_config_path = "config/generator/llama-3-8b-instruct-diff-transformer.yaml"

model_config = OmegaConf.load(diff_transformer_config_path)
diff_transformer_config = OmegaConf.to_container(model_config)
base_config = AutoConfig.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
concat_config = {**base_config.to_dict(), **diff_transformer_config['init_args']['model_config']}

base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    ).eval()

config = LlamaLoraDiffTransformerConfig(**concat_config)
config.diff_attn_implementation = 'eager'
config.learn_lambda = False
config.diff_attn_lambda = 0.5
config.layers_to_transform = list(range(0, 16))
config.diff_attn_init_with_base_weights = True
config.lora_negative_term_only = True
config.negative_term_lora_only = False
config.negative_term_full_dim = False
config.attention_lora_alpha = 4
config.attention_lora_r = 2
config.attention_lora_dropout = 0.1
config.groupnorm = False
config.relu_on_differential = False
config.verbose = True


# print(config._attn_implementation)
model_eager = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to("cuda").bfloat16().eval()
config.diff_attn_implementation = 'flash_attention_2'
model_FA = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to("cuda").bfloat16().eval()

print(model_eager)
print(model_FA)


# test inference
with torch.no_grad():
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model.device)

    output_model_eager = model_eager(input_ids).logits
    output_model_FA = model_FA(input_ids).logits

    # Final outputs for reference
    print("\n### EAGER MODEL OUTPUT ###")
    print(output_model_eager)

    print("\n### FA MODEL OUTPUT ###")
    print(output_model_FA)

    ##########################################################################################################################################################
