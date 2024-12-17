from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM
import torch

lambda2checkpoint = {
    "00": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda00_base_init_alllayers/train/checkpoint-2702",
    "0001": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda0001_base_init_alllayers/train/checkpoint-2702",
    "001": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda001_base_init_alllayers/train/checkpoint-2702",
    "01": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda01_base_init_alllayers/train/checkpoint-2702",
    "04": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda04_base_init_alllayers/train/checkpoint-2702",
    "05": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda05_base_init_alllayers/train/checkpoint-3240",
    "06": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda06_base_init_alllayers/train/checkpoint-3240",
    "07": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda07_base_init_alllayers/train/checkpoint-3240",
    "08": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda08_base_init_alllayers/train/checkpoint-3240",
    "09": "experiments/tune_diff_attn_lambda/train_LoraDiffAtt_NQrfshort_llama38b_instruct_basicprompt_lambda09_base_init_alllayers/train/checkpoint-3240",
}

norms = []
Q_norms = []
K_norms = []
A_norms = []
B_norms = []
left_term_norms = []
right_term_norms = []

for lambda_ in lambda2checkpoint.keys():
    model = LlamaLoraDiffTransformerForCausalLM.from_pretrained(lambda2checkpoint[lambda_])
    norms_per_layer = []
    Q_norms_per_layer = []
    K_norms_per_layer = []
    A_norms_per_layer = []
    B_norms_per_layer = []
    left_term_norms_per_layer = []
    right_term_norms_per_layer = []
    for layer in model.model.layers:
        norms_layer_per_module = {
            "wq_lora_A1": torch.linalg.matrix_norm(layer.self_attn.wq_lora_A1.weight.clone().detach()).item(),
            "wq_lora_A2": torch.linalg.matrix_norm(layer.self_attn.wq_lora_A2.weight.clone().detach()).item(),
            "wq_lora_B1": torch.linalg.matrix_norm(layer.self_attn.wq_lora_B1.weight.clone().detach()).item(),
            "wq_lora_B2": torch.linalg.matrix_norm(layer.self_attn.wq_lora_B2.weight.clone().detach()).item(),
            "wk_lora_A1": torch.linalg.matrix_norm(layer.self_attn.wk_lora_A1.weight.clone().detach()).item(),
            "wk_lora_A2": torch.linalg.matrix_norm(layer.self_attn.wk_lora_A2.weight.clone().detach()).item(),
            "wk_lora_B1": torch.linalg.matrix_norm(layer.self_attn.wk_lora_B1.weight.clone().detach()).item(),
            "wk_lora_B2": torch.linalg.matrix_norm(layer.self_attn.wk_lora_B2.weight.clone().detach()).item(),
        }
        norms_per_layer.append(sum(norms_layer_per_module.values()) / len(norms_layer_per_module))
        Q_norms_per_layer.append((norms_layer_per_module["wq_lora_A1"] + norms_layer_per_module["wq_lora_A2"] + norms_layer_per_module["wq_lora_B1"] + norms_layer_per_module["wq_lora_B2"]) / 4)
        K_norms_per_layer.append((norms_layer_per_module["wk_lora_A1"] + norms_layer_per_module["wk_lora_A2"] + norms_layer_per_module["wk_lora_B1"] + norms_layer_per_module["wk_lora_B2"]) / 4)
        A_norms_per_layer.append((norms_layer_per_module["wq_lora_A1"] + norms_layer_per_module["wq_lora_A2"] + norms_layer_per_module["wk_lora_A1"] + norms_layer_per_module["wk_lora_A2"]) / 4)
        B_norms_per_layer.append((norms_layer_per_module["wq_lora_B1"] + norms_layer_per_module["wq_lora_B2"] + norms_layer_per_module["wk_lora_B1"] + norms_layer_per_module["wk_lora_B2"]) / 4)
        left_term_norms_per_layer.append((norms_layer_per_module["wq_lora_A1"] + norms_layer_per_module["wq_lora_B1"] + norms_layer_per_module["wk_lora_A1"] + norms_layer_per_module["wk_lora_B1"]) / 4)
        right_term_norms_per_layer.append((norms_layer_per_module["wq_lora_A2"] + norms_layer_per_module["wq_lora_B2"] + norms_layer_per_module["wk_lora_A2"] + norms_layer_per_module["wk_lora_B2"]) / 4)
    norms.append(norms_per_layer)
    Q_norms.append(Q_norms_per_layer)
    K_norms.append(K_norms_per_layer)
    A_norms.append(A_norms_per_layer)
    B_norms.append(B_norms_per_layer)
    left_term_norms.append(left_term_norms_per_layer)
    right_term_norms.append(right_term_norms_per_layer)

# plot norms along layers for each lambda
import matplotlib.pyplot as plt

for name,norm in zip(["norms", "Q_norms", "K_norms", "A_norms", "B_norms", "left_term_norms", "right_term_norms"], [norms, Q_norms, K_norms, A_norms, B_norms, left_term_norms, right_term_norms]):
    fig, ax = plt.subplots()
    x = range(32)
    for i, norms_per_layer in enumerate(norm):
        ax.plot(x, norms_per_layer, label=f"lambda={['0.0', '0.001', '0.01', '0.1', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'][i]}")
    ax.legend()
    ax.set_title(f"{name.replace('_', ' ')} per layer")
    ax.set_xlabel("Layer i")
    ax.set_ylabel("Average norm")
    plt.savefig(f"lora_{name}_per_layer_xlambdas.png")
