# from omegaconf import OmegaConf
# from transformers import AutoConfig
# from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
# from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffAttention, LlamaLoraDiffTransformerModel, LlamaLoraDiffTransformerForCausalLM
# from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
# import torch

# model_config = OmegaConf.load("config/generator/llama-3-8b-instruct-diff-transformer.yaml")
# # print("model_config: ", model_config)
# diff_transformer_config = OmegaConf.to_container(model_config)["init_args"]["model_config"]

# base_config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# # print(base_config)
# # concatenate configs
# concat_config = {**base_config.to_dict(), **diff_transformer_config}
# # print("model_config: ", model_config)


# # diff_transformer_config = LlamaLoraDiffTransformerConfig(**diff_transformer_config)
# concat_config = LlamaLoraDiffTransformerConfig(**concat_config)

# # print("base_config: \n", base_config)
# # print("diff_transformer_config: \n", diff_transformer_config)
# # print("concat_config: \n", concat_config)

# # model_base_config = LlamaLoraDiffTransformerModel(base_config)
# # print("model_base_config: \n", model_base_config)
# # model_diff_transformer_config = LlamaLoraDiffTransformerModel(diff_transformer_config)
# # print("model_diff_transformer_config: \n", model_diff_transformer_config)
# # model_concat_config = LlamaLoraDiffTransformerModel(concat_config)
# # model_concat_config = LlamaLoraDiffTransformerForCausalLM(concat_config)
# # print("model_concat_config: \n", model_concat_config)
# # print("model device: ", model_concat_config.device)
# # model_concat_config = model_concat_config.to("cuda")
# # print("model device: ", model_concat_config.device)

# # LlamaLoraDiffTransformerConfig.register_for_auto_class()
# # LlamaLoraDiffTransformerModel.register_for_auto_class("AutoModelForCausalLM")
# # print("registered")
# # model.save_pretrained("tests/llama_diff_transformer")
# # print("saved model")
# # model_loaded = model.from_pretrained("tests/llama_diff_transformer")
# # print("loaded model")
# # print(model_loaded)



# ### CHECK A2 / B2 PARAMS DECAY TO 0
# init_model = LlamaLoraDiffTransformerForCausalLM.from_pretrained("experiments/debug/train_eval_lambda00_debug/train/checkpoint-1", config=concat_config)
# print("checkpoint 1, layer 0, wq_lora_A1", init_model.model.layers[0].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_A1.weight.clone().detach()))
# print("checkpoint 1, layer 0, wq_lora_A2", init_model.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_A2.weight.clone().detach()))
# print("checkpoint 1, layer 0, wq_lora_B1", init_model.model.layers[0].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_B1.weight.clone().detach()))
# print("checkpoint 1, layer 0, wq_lora_B2", init_model.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_B2.weight.clone().detach()))
# print("checkpoint 1, layer 0, wk_lora_A1", init_model.model.layers[0].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_A1.weight.clone().detach()))
# print("checkpoint 1, layer 0, wk_lora_A2", init_model.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_A2.weight.clone().detach()))
# print("checkpoint 1, layer 0, wk_lora_B1", init_model.model.layers[0].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_B1.weight.clone().detach()))
# print("checkpoint 1, layer 0, wk_lora_B2", init_model.model.layers[0].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_B2.weight.clone().detach()))
# print("checkpoint 1, layer 31, wq_lora_A1", init_model.model.layers[31].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_A1.weight.clone().detach()))
# print("checkpoint 1, layer 31, wq_lora_A2", init_model.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_A2.weight.clone().detach()))
# print("checkpoint 1, layer 31, wq_lora_B1", init_model.model.layers[31].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_B1.weight.clone().detach()))
# print("checkpoint 1, layer 31, wq_lora_B2", init_model.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_B2.weight.clone().detach()))
# print("checkpoint 1, layer 31, wk_lora_A1", init_model.model.layers[31].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_A1.weight.clone().detach()))
# print("checkpoint 1, layer 31, wk_lora_A2", init_model.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_A2.weight.clone().detach()))
# print("checkpoint 1, layer 31, wk_lora_B1", init_model.model.layers[31].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_B1.weight.clone().detach()))
# print("checkpoint 1, layer 31, wk_lora_B2", init_model.model.layers[31].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_B2.weight.clone().detach()))
# init_model = LlamaLoraDiffTransformerForCausalLM.from_pretrained("experiments/debug/train_eval_lambda00_debug/train/checkpoint-2", config=concat_config)
# print("checkpoint 2, layer 0, wq_lora_A1", init_model.model.layers[0].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_A1.weight.clone().detach()))
# print("checkpoint 2, layer 0, wq_lora_A2", init_model.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_A2.weight.clone().detach()))
# print("checkpoint 2, layer 0, wq_lora_B1", init_model.model.layers[0].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_B1.weight.clone().detach()))
# print("checkpoint 2, layer 0, wq_lora_B2", init_model.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wq_lora_B2.weight.clone().detach()))
# print("checkpoint 2, layer 0, wk_lora_A1", init_model.model.layers[0].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_A1.weight.clone().detach()))
# print("checkpoint 2, layer 0, wk_lora_A2", init_model.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_A2.weight.clone().detach()))
# print("checkpoint 2, layer 0, wk_lora_B1", init_model.model.layers[0].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_B1.weight.clone().detach()))
# print("checkpoint 2, layer 0, wk_lora_B2", init_model.model.layers[0].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[0].self_attn.wk_lora_B2.weight.clone().detach()))
# print("checkpoint 2, layer 31, wq_lora_A1", init_model.model.layers[31].self_attn.wq_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_A1.weight.clone().detach()))
# print("checkpoint 2, layer 31, wq_lora_A2", init_model.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_A2.weight.clone().detach()))
# print("checkpoint 2, layer 31, wq_lora_B1", init_model.model.layers[31].self_attn.wq_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_B1.weight.clone().detach()))
# print("checkpoint 2, layer 31, wq_lora_B2", init_model.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wq_lora_B2.weight.clone().detach()))
# print("checkpoint 2, layer 31, wk_lora_A1", init_model.model.layers[31].self_attn.wk_lora_A1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_A1.weight.clone().detach()))
# print("checkpoint 2, layer 31, wk_lora_A2", init_model.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_A2.weight.clone().detach()))
# print("checkpoint 2, layer 31, wk_lora_B1", init_model.model.layers[31].self_attn.wk_lora_B1.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_B1.weight.clone().detach()))
# print("checkpoint 2, layer 31, wk_lora_B2", init_model.model.layers[31].self_attn.wk_lora_B2.weight.shape, torch.linalg.matrix_norm(init_model.model.layers[31].self_attn.wk_lora_B2.weight.clone().detach()))

# model_name = "experiments/tune_diff_attn_lambda/lambda00/train/checkpoint-2162"
# config = LlamaLoraDiffTransformerConfig.from_pretrained(model_name)
# model_epoch_01 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(model_name, config=config, attn_implementation="eager", torch_dtype=torch.bfloat16)
# print("model_epoch_01 wq_lora_A2", model_epoch_01.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[0].self_attn.wq_lora_A2.weight))
# print("model_epoch_01 wk_lora_A2", model_epoch_01.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[0].self_attn.wk_lora_A2.weight))
# print("model_epoch_01 wq_lora_A2 layer 31", model_epoch_01.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[31].self_attn.wq_lora_A2.weight))
# print("model_epoch_01 wk_lora_A2 layer 31", model_epoch_01.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[31].self_attn.wk_lora_A2.weight))
# print("model_epoch_01 wq_lora_B2 layer 0", model_epoch_01.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[0].self_attn.wq_lora_B2.weight))
# print("model_epoch_01 wq_lora_B2 layer 31", model_epoch_01.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[31].self_attn.wq_lora_B2.weight))
# print("model_epoch_01 wq_lora_B2 layer 0", model_epoch_01.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[0].self_attn.wq_lora_B2.weight))
# print("model_epoch_01 wq_lora_B2 layer 31", model_epoch_01.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_01.model.layers[31].self_attn.wq_lora_B2.weight))

# model_name = "experiments/tune_diff_attn_lambda/lambda00/train/checkpoint-21625"
# config = LlamaLoraDiffTransformerConfig.from_pretrained(model_name)
# model_epoch_10 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(model_name, config=config, attn_implementation="eager", torch_dtype=torch.bfloat16)
# print("model_epoch_10 wq_lora_A2", model_epoch_10.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[0].self_attn.wq_lora_A2.weight))
# print("model_epoch_10 wk_lora_A2", model_epoch_10.model.layers[0].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[0].self_attn.wk_lora_A2.weight))
# print("model_epoch_10 wq_lora_A2 layer 31", model_epoch_10.model.layers[31].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[31].self_attn.wq_lora_A2.weight))
# print("model_epoch_10 wk_lora_A2 layer 31", model_epoch_10.model.layers[31].self_attn.wk_lora_A2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[31].self_attn.wk_lora_A2.weight))
# print("model_epoch_10 wq_lora_B2 layer 0", model_epoch_10.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[0].self_attn.wq_lora_B2.weight))
# print("model_epoch_10 wq_lora_B2 layer 31", model_epoch_10.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[31].self_attn.wq_lora_B2.weight))
# print("model_epoch_10 wq_lora_B2 layer 0", model_epoch_10.model.layers[0].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[0].self_attn.wq_lora_B2.weight))
# print("model_epoch_10 wq_lora_B2 layer 31", model_epoch_10.model.layers[31].self_attn.wq_lora_B2.weight.shape, torch.linalg.matrix_norm(model_epoch_10.model.layers[31].self_attn.wq_lora_B2.weight))

# print("init_model wq_lora_A2 layer 0", init_model.model.layers[0].self_attn.wq_lora_A2.weight)
# print("model_epoch_10 wq_lora_A2 layer 0", model_epoch_10.model.layers[0].self_attn.wq_lora_A2.weight)

# model_name = "experiments/debug/train_eval_lambda00_debug/train/checkpoint-1"
# config = LlamaLoraDiffTransformerConfig.from_pretrained(model_name)
# other_model = LlamaLoraDiffTransformerForCausalLM.from_pretrained(model_name, config=config, attn_implementation="eager", torch_dtype=torch.bfloat16)
# print("other_model wq_lora_A2 layer 0", other_model.model.layers[0].self_attn.wq_lora_A2.weight.shape, torch.linalg.matrix_norm(other_model.model.layers[0].self_attn.wq_lora_A2.weight))



# ### KILT NQ RF SHORT ### - check long labels
# import datasets
# import matplotlib.pyplot as plt
# ds = datasets.load_from_disk("datasets/kilt_nq_rf_short_train")
# print(ds[0])
# content_lengths = [len(x["content"]) for x in ds]
# label_lengths = [len(x["label"][0]) for x in ds]
# plt.hist(content_lengths, bins=50)
# plt.savefig("content_lengths_hist.png")
# plt.hist(label_lengths, bins=50)
# plt.savefig("label_lengths_hist.png")
# print(max(content_lengths))
# print(max(label_lengths))

# ### KILT NQ ### - check long labels
# import datasets
# import matplotlib.pyplot as plt
# ds = datasets.load_from_disk("datasets/kilt_nq_train")
# print(ds[0])
# content_lengths = [len(x["content"]) for x in ds]
# label_lengths = [len(x["label"][0]) for x in ds]
# plt.hist(content_lengths, bins=50)
# plt.savefig("content_lengths_hist.png")
# plt.hist(label_lengths, bins=50)
# plt.savefig("label_lengths_hist.png")
# print(max(content_lengths))
# print(max(label_lengths))

import numpy as np
att = np.load("attentions_0_last.npy")
print(att)
print(np.allclose(att, np.tril(att)))