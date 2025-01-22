from omegaconf import OmegaConf
from transformers import AutoConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffAttention, LlamaLoraDiffTransformerModel, LlamaLoraDiffTransformerForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
import torch

# model_config = OmegaConf.load("config/generator/llama-3-8b-instruct-diff-transformer.yaml")
# diff_transformer_config = OmegaConf.to_container(model_config)["init_args"]["model_config"]

# base_config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# concat_config = {**base_config.to_dict(), **diff_transformer_config}
# concat_config = LlamaLoraDiffTransformerConfig(**concat_config)

# model = LlamaLoraDiffTransformerForCausalLM.from_pretrained("experiments/debug/train_eval_lambda00_debug/train/checkpoint-1", config=concat_config, attn_implementation="eager", torch_dtype=torch.bfloat16)

# # print trainable parameters names
# for name, param in model.named_parameters():
#     if param.requires_grad and 'lora' not in name:
#         print(name)

# # print number of trainable params
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of trainable parameters: ", num_params)

# print("before load, layer 0, wq_lora_A1", model.model.layers[0].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_A1.weight.clone().detach()))
# print("before load, layer 0, wq_lora_A2", model.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_A2.weight.clone().detach()))
# print("before load, layer 0, wq_lora_B1", model.model.layers[0].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_B1.weight.clone().detach()))
# print("before load, layer 0, wq_lora_B2", model.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_B2.weight.clone().detach()))
# print("before load, layer 0, wk_lora_A1", model.model.layers[0].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_A1.weight.clone().detach()))
# print("before load, layer 0, wk_lora_A2", model.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_A2.weight.clone().detach()))
# print("before load, layer 0, wk_lora_B1", model.model.layers[0].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_B1.weight.clone().detach()))
# print("before load, layer 0, wk_lora_B2", model.model.layers[0].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_B2.weight.clone().detach()))
# print("before load, layer 31, wq_lora_A1", model.model.layers[31].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_A1.weight.clone().detach()))
# print("before load, layer 31, wq_lora_A2", model.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_A2.weight.clone().detach()))
# print("before load, layer 31, wq_lora_B1", model.model.layers[31].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_B1.weight.clone().detach()))
# print("before load, layer 31, wq_lora_B2", model.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_B2.weight.clone().detach()))
# print("before load, layer 31, wk_lora_A1", model.model.layers[31].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_A1.weight.clone().detach()))
# print("before load, layer 31, wk_lora_A2", model.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_A2.weight.clone().detach()))
# print("before load, layer 31, wk_lora_B1", model.model.layers[31].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_B1.weight.clone().detach()))
# print("before load, layer 31, wk_lora_B2", model.model.layers[31].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_B2.weight.clone().detach()))

# trained_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
# # artificially add 1 to every trained param
# with torch.no_grad():
#     for name, param in trained_params.items():
#         trained_params[name] = param + 1
# save_path = "experiments/debug/test_save_adapters/diff_attn_lora.pth"
# torch.save(trained_params, save_path)
# trained_params = torch.load(save_path)

# model.load_state_dict(trained_params, strict=False)

# print("checkpoint-1, layer 0, wq_lora_A1", model.model.layers[0].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_A1.weight.clone().detach()), model.model.layers[0].self_attn.wq_lora_A1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wq_lora_A2", model.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_A2.weight.clone().detach()), model.model.layers[0].self_attn.wq_lora_A2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wq_lora_B1", model.model.layers[0].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_B1.weight.clone().detach()), model.model.layers[0].self_attn.wq_lora_B1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wq_lora_B2", model.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wq_lora_B2.weight.clone().detach()), model.model.layers[0].self_attn.wq_lora_B2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wk_lora_A1", model.model.layers[0].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_A1.weight.clone().detach()), model.model.layers[0].self_attn.wk_lora_A1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wk_lora_A2", model.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_A2.weight.clone().detach()), model.model.layers[0].self_attn.wk_lora_A2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wk_lora_B1", model.model.layers[0].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_B1.weight.clone().detach()), model.model.layers[0].self_attn.wk_lora_B1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 0, wk_lora_B2", model.model.layers[0].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[0].self_attn.wk_lora_B2.weight.clone().detach()), model.model.layers[0].self_attn.wk_lora_B2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wq_lora_A1", model.model.layers[31].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_A1.weight.clone().detach()), model.model.layers[31].self_attn.wq_lora_A1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wq_lora_A2", model.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_A2.weight.clone().detach()), model.model.layers[31].self_attn.wq_lora_A2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wq_lora_B1", model.model.layers[31].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_B1.weight.clone().detach()), model.model.layers[31].self_attn.wq_lora_B1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wq_lora_B2", model.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wq_lora_B2.weight.clone().detach()), model.model.layers[31].self_attn.wq_lora_B2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wk_lora_A1", model.model.layers[31].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_A1.weight.clone().detach()), model.model.layers[31].self_attn.wk_lora_A1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wk_lora_A2", model.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_A2.weight.clone().detach()), model.model.layers[31].self_attn.wk_lora_A2.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wk_lora_B1", model.model.layers[31].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_B1.weight.clone().detach()), model.model.layers[31].self_attn.wk_lora_B1.weight.clone().detach()[0,:3])
# print("checkpoint-1, layer 31, wk_lora_B2", model.model.layers[31].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(model.model.layers[31].self_attn.wk_lora_B2.weight.clone().detach()), model.model.layers[31].self_attn.wk_lora_B2.weight.clone().detach()[0,:3])

# model = LlamaLoraDiffTransformerForCausalLM.from_pretrained("experiments/debug/train_eval_lambda00_debug/train/checkpoint-2", config=concat_config, attn_implementation="eager", torch_dtype=torch.bfloat16)

# load this adapter: experiments/debug/debug_train_eval_lambda05_loramlp_debug/train/checkpoint-1/adapter_model.safetensors
from safetensors.torch import load_file
# adapters_state_dict = load_file("experiments/tune_diff_attn_nq_lambda_spladeberta/tmp_train_LoraDiffAtt_NQ_llama38b_instruct_spladeberta_top3_basicprompt_lambda02_base_init_alllayers_rightloraonly/train/checkpoint-4324/model.safetensors")
adapters_state_dict = load_file("experiments/debug/debug_train_eval_rightloraonly_lambda_learnable/train/checkpoint-2/model.safetensors")
print(adapters_state_dict.keys())