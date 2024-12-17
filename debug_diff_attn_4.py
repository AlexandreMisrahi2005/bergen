from importlib import reload

from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
diff_transformer_config_path = "config/generator/llama-3-8b-instruct-diff-transformer.yaml"
attn_implementation = "eager"
# attn_implementation = "flash_attention_2"

# read config with OmegaConf
model_config = OmegaConf.load(diff_transformer_config_path)
# print("model_config", model_config)
model_config.init_args.model_config.attn_implementation = attn_implementation
# print("model_config", model_config)
diff_transformer_config = OmegaConf.to_container(model_config)
# print("diff_transformer_config", diff_transformer_config)
base_config = AutoConfig.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
# print("base_config", base_config)
concat_config = {**base_config.to_dict(), **diff_transformer_config['init_args']['model_config']}
# print("concat_config", concat_config)

base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    ).eval()

config = LlamaLoraDiffTransformerConfig(**concat_config)
config.learn_lambda = False
config.diff_attn_lambda = 0.5
config.layers_to_transform = list(range(0, 0))
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
config.dev = True


# print(config._attn_implementation)
model = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to("cuda").bfloat16().eval()

# del base_model
# gc.collect()
# torch.cuda.empty_cache()

print(base_model)
print(model)

# # print trainable parameters names
# print("Layer 0 self-attn trainable params")
# for name, param in model.model.layers[0].self_attn.named_parameters():
#     if param.requires_grad:
#         print(name, param.numel())

# print number of trainable params
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of trainable parameters: ", num_params)

# test inference
with torch.no_grad():

    ##########################################################################################################################################################
    # input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(model.device)
    # inputs_embeds = base_model.model.embed_tokens(input_ids)
    # past_key_values = DynamicCache()
    # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    # cache_position = torch.arange(
    #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    # )
    # position_ids = cache_position.unsqueeze(0)
    # print(inputs_embeds.shape)
    # output_base = base_model.model.layers[0].self_attn(inputs_embeds, position_ids=position_ids, past_key_value=past_key_values, cache_position=cache_position)[0]

    # # output_base = base_model(input_ids).logits
    # print("### BASE ###")
    # print(output_base)
    # print(output_base.shape)


    # # output = model(input_ids).logits
    # inputs_embeds = model.model.embed_tokens(input_ids)
    # print(inputs_embeds.shape)
    # past_key_values = DynamicCache()
    # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    # cache_position = torch.arange(
    #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    # )
    # position_ids = cache_position.unsqueeze(0)
    # output = model.model.layers[0].self_attn(inputs_embeds, position_ids=position_ids, past_key_value=past_key_values, cache_position=cache_position)[0]
    # print("\n\n### DIFF FLASH ATTN ###")
    # print(output)
    # print(output.shape)
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    # input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(model.device)

    # output_base = base_model(input_ids).logits
    # print("### BASE ###")
    # print(output_base)

    # output = model(input_ids).logits
    # print("\n\n### DIFF FLASH ATTN ###")
    # print(output)
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    # Store intermediate outputs with module names as keys
    base_activations = {}
    model_activations = {}

    # Hook function to capture outputs with module names
    def hook_fn_base(name):
        def hook(module, input, output):
            base_activations[name] = output
        return hook

    def hook_fn_model(name):
        def hook(module, input, output):
            model_activations[name] = output
        return hook

    # Register hooks on both models
    def register_hooks_with_names(model, hook_fn, prefix=""):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
                full_name = f"{prefix}{name}"
                hooks.append(module.register_forward_hook(hook_fn(full_name)))
                print(f"Registered hook on {full_name}")
        return hooks

    # Register hooks for both models
    hooks_base = register_hooks_with_names(base_model, hook_fn_base, prefix="base_")
    hooks_model = register_hooks_with_names(model, hook_fn_model, prefix="model_")

    # Input tensor
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model.device)

    # Forward pass for both models
    output_model = model(input_ids).logits
    output_base = base_model(input_ids).logits

    # print(base_activations.keys())
    # print(model_activations.keys())

    # Compare outputs layer by layer using module names
    for name in base_activations.keys():
        name_model = name.replace('base_', 'model_')
        if name_model in model_activations:
            base_out = base_activations[name]
            model_out = model_activations[name_model]
            if not torch.allclose(base_out, model_out, atol=1e-8):
                print(f"Difference detected in module: {name_model}")
                print(f"Base output: {base_out}")
                print(f"Model output: {model_out}")
                break
        else:
            print(f"Module {name_model} not found in model_activations")

    # Cleanup: Remove hooks
    for hook in hooks_base + hooks_model:
        hook.remove()

    # Final outputs for reference
    print("\n### BASE MODEL OUTPUT ###")
    print(output_base)

    print("\n### DIFF MODEL OUTPUT ###")
    print(output_model)

    ##########################################################################################################################################################
