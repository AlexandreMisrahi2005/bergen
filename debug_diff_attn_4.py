from importlib import reload

from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM

INSPECT_FORWARD_PASS = True
INSPECT_BACKWARD_PASS = False
INSPECT_FA = True

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
diff_transformer_config_path = "config/generator/llama-3-8b-instruct-diff-transformer.yaml"
attn_implementation = "eager"

model_config = OmegaConf.load(diff_transformer_config_path)
model_config.init_args.model_config.attn_implementation = attn_implementation
diff_transformer_config = OmegaConf.to_container(model_config)
base_config = AutoConfig.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
concat_config = {**base_config.to_dict(), **diff_transformer_config['init_args']['model_config']}

base_model_eager = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    ).eval()

base_model_FA = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    ).eval() if INSPECT_FA else None

config = LlamaLoraDiffTransformerConfig(**concat_config)
config.diff_attn_implementation = attn_implementation
config.learn_lambda = False
config.diff_attn_lambda = 0.5
config.layers_to_transform = list(range(0, 16))
config.diff_attn_init_with_base_weights = True
config.lora_negative_term_only = False
config.negative_term_lora_only = False
config.negative_term_full_dim = False
config.attention_lora_alpha = 4
config.attention_lora_r = 2
config.attention_lora_dropout = 0.1
config.lora_v = False
config.lora_o = False
config.lora_mlp = False
config.groupnorm = False
config.relu_on_differential = False
config.verbose = True


# print(config._attn_implementation)
model = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model_eager).to("cuda").bfloat16().eval()
config.diff_attn_implementation = 'flash_attention_2' if INSPECT_FA else 'eager'
model_FA = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model_eager).to("cuda").bfloat16().eval() if INSPECT_FA else None

# del base_model
# gc.collect()
# torch.cuda.empty_cache()

print(base_model_eager)
print(base_model_FA)
print(model)
print(model_FA)

# # print trainable parameters names
# print("Layer 0 self-attn trainable params")
# for name, param in model.model.layers[0].self_attn.named_parameters():
#     if param.requires_grad:
#         print(name, param.numel())

# print number of trainable params
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of trainable parameters: ", num_params)


##########################################################################################################################################################
if INSPECT_FORWARD_PASS:
    # test inference
    with torch.no_grad():
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
        hooks_base = register_hooks_with_names(base_model_eager, hook_fn_base, prefix="base_")
        hooks_model = register_hooks_with_names(model, hook_fn_model, prefix="model_")

        # Input tensor
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model_eager.device)

        # Forward pass for both models
        output_model = model(input_ids).logits
        output_model_FA = model_FA(input_ids).logits if INSPECT_FA else None
        output_base = base_model_eager(input_ids).logits
        output_base_FA = base_model_FA(input_ids).logits if INSPECT_FA else None

        # print(base_activations.keys())
        # print(model_activations.keys())

        # Compare outputs layer by layer using module names
        for name in base_activations.keys():
            name_model = name.replace('base_', 'model_')
            if name_model in model_activations:
                base_out = base_activations[name]
                model_out = model_activations[name_model]
                if not torch.allclose(base_out, model_out, atol=1e-4):
                    print(f"Difference detected in module: {name_model}")
                    print(f"Base output: \n{base_out}")
                    print(f"diff-model output: \n{model_out}")
                    break
            else:
                print(f"Module {name_model} not found in model_activations")

        # Cleanup: Remove hooks
        for hook in hooks_base + hooks_model:
            hook.remove()

        # Final outputs for reference
        print("\n### BASE MODEL OUTPUT ###")
        print(output_base)
        print("max-diff with base eager = ", torch.max(torch.abs(output_base - output_base)).cpu().float())
        print("avg-diff with base eager = ", torch.mean(torch.abs(output_base - output_base)).cpu().float())

        if INSPECT_FA:
            print("\n### BASE MODEL FA OUTPUT ###")
            print(output_base_FA)
            print("max-diff with base eager = ", torch.max(torch.abs(output_base_FA - output_base)).cpu().float())
            print("avg-diff with base eager = ", torch.mean(torch.abs(output_base_FA - output_base)).cpu().float())

        print("\n### DIFF MODEL OUTPUT ###")
        print(output_model)
        print("max-diff with base eager = ", torch.max(torch.abs(output_base - output_model)).cpu().float())
        print("avg-diff with base eager = ", torch.mean(torch.abs(output_base - output_model)).cpu().float())

        if INSPECT_FA:
            print("\n### DIFF MODEL FA OUTPUT ###")
            print(output_model_FA)
            print("max-diff with base FA = ", torch.max(torch.abs(output_base - output_model_FA)).cpu().float())
            print("avg-diff with base FA = ", torch.mean(torch.abs(output_base - output_model_FA)).cpu().float())

if INSPECT_BACKWARD_PASS:
    # test the backward pass

    # activate all gradients (full finetuning)
    for param in base_model_eager.parameters():
        param.requires_grad = True
    
    if INSPECT_FA:
        for param in base_model_FA.parameters():
            param.requires_grad = True
        
    for param in model.parameters():
        param.requires_grad = True

    if INSPECT_FA:
        for param in model_FA.parameters():
            param.requires_grad = True

    # print trainable params
    print("Trainable params")
    print("base eager    = ", sum(p.numel() for p in base_model_eager.parameters() if p.requires_grad))
    print("base FA       = ", sum(p.numel() for p in base_model_FA.parameters() if p.requires_grad) if INSPECT_FA else None)
    print("diff-model    = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("diff-model FA = ", sum(p.numel() for p in model_FA.parameters() if p.requires_grad) if INSPECT_FA else None)

    # test with loss = mse
    loss = torch.nn.MSELoss()
    
    # input tensor
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model_eager.device)

    # forward pass
    output_base_eager = base_model_eager(input_ids).logits
    output_base_FA = base_model_FA(input_ids).logits if INSPECT_FA else None
    output_model = model(input_ids).logits
    output_model_FA = model_FA(input_ids).logits if INSPECT_FA else None

    # loss
    loss_base_eager = loss(output_base_eager, torch.zeros_like(output_base_eager))
    loss_base_FA = loss(output_base_FA, torch.zeros_like(output_base_FA)) if INSPECT_FA else None
    loss_model = loss(output_model, torch.zeros_like(output_model))
    loss_model_FA = loss(output_model_FA, torch.zeros_like(output_model_FA)) if INSPECT_FA else None

    # backward pass
    base_model_eager.zero_grad()
    base_model_FA.zero_grad() if INSPECT_FA else None
    model.zero_grad()
    model_FA.zero_grad() if INSPECT_FA else None

    loss_base_eager.backward()
    base_model_eager = base_model_eager.cpu()
    if INSPECT_FA:
        loss_base_FA.backward()
        base_model_FA = base_model_FA.cpu()
    loss_model.backward()
    model = model.cpu()
    if INSPECT_FA:
        loss_model_FA.backward()
        model_FA = model_FA.cpu()

    # check gradients
    print("Gradients")
    names = ["- BASE (EAGER)", "- BASE (FA)", "- DIFF-MODEL (EAGER)", "- DIFF-MODEL (FA)"] if INSPECT_FA else ["- BASE (EAGER)", "- DIFF-MODEL (EAGER)"]
    for i,m in enumerate([base_model_eager, base_model_FA, model, model_FA] if INSPECT_FA else [base_model_eager, model]):
        for layer in [0,15]:
            print(f"{names[i]} layer {layer} q_proj = \n", m.model.layers[layer].self_attn.q_proj.weight.grad)
            print("max diff with base eager = ", torch.max(torch.abs(base_model_eager.model.layers[layer].self_attn.q_proj.weight.grad - m.model.layers[layer].self_attn.q_proj.weight.grad)).cpu().float())
            print("avg diff with base eager = ", torch.mean(torch.abs(base_model_eager.model.layers[layer].self_attn.q_proj.weight.grad - m.model.layers[layer].self_attn.q_proj.weight.grad)).cpu().float())
            print("---")
        print("------")


##########################################################################################################################################################
