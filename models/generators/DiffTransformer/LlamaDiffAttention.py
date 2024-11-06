'''
Implementation of Differential Transformer mechanism for transformers' LlamaAttention class
'''

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from transformers.utils import logging

logger = logging.get_logger(__name__)

def lambda_init_fn(layer_idx: int = None) -> float:
    return 1e-3 # try small init
    # return 1.0 - math.exp(-0.003 * layer_idx) # try small init
    # return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)

class LlamaLoraDiffAttention(LlamaAttention):
    """Multi-headed differential attention from 'Differential Transformer' paper: https://arxiv.org/abs/2410.05258"""

    def __init__(self, config: LlamaConfig, layer_idx: int, lora_config: dict):
        super().__init__(config, layer_idx)

        self.lora_config = lora_config

        self.lambda_init = lambda_init_fn()
        # self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
        # self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
        # self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
        # self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)

        self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, lora_config.r, bias=False)
        self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, lora_config.r, bias=False)
        self.wq_lora_B1 = nn.Linear(lora_config.r, self.num_heads * self.head_dim, bias=False)
        self.wq_lora_B2 = nn.Linear(lora_config.r, self.num_heads * self.head_dim, bias=False)
        self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, lora_config.r, bias=False)
        self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, lora_config.r, bias=False)
        self.wk_lora_B1 = nn.Linear(lora_config.r, self.num_key_value_heads * self.head_dim, bias=False)
        self.wk_lora_B2 = nn.Linear(lora_config.r, self.num_key_value_heads * self.head_dim, bias=False)
        self.lora_dropout = nn.Dropout(p=lora_config.lora_dropout)
        self.lora_scaling = lora_config.lora_alpha / lora_config.r
        self.init_weights()

        # activate LoRA parameters, deactivate the rest
        for name, param in self.named_parameters():
            # param.requires_grad = True
            # print(f"[inside module __init__] Activating parameter {name}")
            if 'lambda' in name or 'lora' in name:
                # print(f"[inside module __init__] Activating parameter {name}")
                param.requires_grad = True
            else:
                # print(f"[inside module __init__] Deactivating parameter {name}")
                param.requires_grad = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # X = (batch, q_len, hidden_dim) where hidden_dim = num_heads * head_dim; note in QA task q_len > 1 in first pass and q_len=1 in next passes (context length);

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaDiffAttention")
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states) # X @ W_q = (b, q_len, hidden_dim) @ (hidden_dim, self.num_heads * self.head_dim) = (b, q_len, self.num_heads * self.head_dim)  where self.num_heads * self.head_dim = hidden_dim = 4096
            key_states = self.k_proj(hidden_states)   # X @ W_k = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            value_states = self.v_proj(hidden_states) # X @ W_v = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            assert all((
                query_states.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                key_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                value_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            )), f"query_states.size() = {query_states.size()}, key_states.size() = {key_states.size()}, value_states.size() = {value_states.size()}"

            lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
            lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
            lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1
            assert all((
                lora_query_states_1.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                lora_query_states_2.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                lora_key_states_1.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                lora_key_states_2.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            )), f"lora_query_states_1.size() = {lora_query_states_1.size()}, lora_query_states_2.size() = {lora_query_states_2.size()}, lora_key_states_1.size() = {lora_key_states_1.size()}, lora_key_states_2.size() = {lora_key_states_2.size()}"

            query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
            query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
            key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
            key_states_2 = key_states + lora_key_states_2 # (b, q_len, self.num_key_value_heads * self.head_dim)

        query_states_1 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        key_states_1 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states_1, key_states_1 = apply_rotary_pos_emb(query_states_1, key_states_1, cos, sin)
        query_states_2, key_states_2 = apply_rotary_pos_emb(query_states_2, key_states_2, cos, sin) 

        # TODO: check if this is correct
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # concatenate key_states along the head_dim dimension for easier cache management 
            # this way we 'pretend' the head dim is twice is truly is, but we can store both key_1 and key_2 in the cache without modifying how cache works
            key_states_cache = torch.cat([key_states_1, key_states_2], dim=-1) # concat to shape (b, num_key_value_heads, q_len, 2 * head_dim)
            key_states_cache, value_states = past_key_value.update(key_states_cache, value_states, self.layer_idx, cache_kwargs)
            total_q_len = key_states_cache.size(-2)
            key_states_1, key_states_2 = key_states_cache.split(self.head_dim, dim=-1) # split each key back to shape (b, num_key_value_heads, q_len, head_dim)
            assert all((
                key_states_1.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
                key_states_2.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
                value_states.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            )), f"key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"

        key_states_1 = repeat_kv(key_states_1, self.num_key_value_groups) # (b, num_key_value_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        key_states_2 = repeat_kv(key_states_2, self.num_key_value_groups) # same
        value_states = repeat_kv(value_states, self.num_key_value_groups) # (b, num_key_value_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        assert all((
            query_states_1.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]),
            query_states_2.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]),
            key_states_1.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
            key_states_2.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
            value_states.size() == torch.Size([bsz, self.num_heads, total_q_len, self.head_dim]),
        )), f"query_states_1.size() = {query_states_1.size()}, query_states_2.size() = {query_states_2.size()}, key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"

        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim) # (b, num_heads, q_len, head_dim) @ (b, num_heads, q_len, head_dim).T(2,3) -> (b, num_heads, q_len, q_len)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim) # same
        assert all((
            attn_weights_1.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
            attn_weights_2.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
        )), f"attn_weights_1.size() = {attn_weights_1.size()}, attn_weights_2.size() = {attn_weights_2.size()}"

        if attention_mask is not None:  # no matter the length, we just slice it
            if attention_mask.dim() == 2: # depending on gpu type and inference setup (training or not, flash-attention, etc) attention mask can be 2D or 4D
                # Expand attention mask to 4D
                attention_mask = attention_mask[:, None, None, :].expand(-1, 1, hidden_states.size(1), -1) # TODO: check this is correct (+ implement flash attention)
            causal_mask = attention_mask[:, :, :, : key_states_1.shape[-2]] # (b, 1, q_len, q_len)
            
            attn_weights_1 = attn_weights_1 + causal_mask
            attn_weights_2 = attn_weights_2 + causal_mask

        # upcast attention to fp32
        attn_weights_1 = nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_2 = nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_1 = nn.functional.dropout(attn_weights_1, p=self.attention_dropout, training=self.training)
        attn_weights_2 = nn.functional.dropout(attn_weights_2, p=self.attention_dropout, training=self.training)

        # lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        # lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        # lambda_full = lambda_1 - lambda_2 + self.lambda_init
        lambda_full = self.lambda_init

        attn_weights = attn_weights_1 - lambda_full * attn_weights_2 # diff attn
        attn_output = torch.matmul(attn_weights, value_states) # (b, num_heads, q_len, q_len) @ (b, num_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        assert attn_output.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]), f"attn_output.size() = {attn_output.size()}"

        attn_output = attn_output * (1 - self.lambda_init)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaDiffAttention")
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def load_weights(self, layer):
        """
        Load the weights of the model with the layer loaded from a checkpoint
        """
        # self.lambda_q1.data = layer.self_attn.lambda_q1.data
        # self.lambda_k1.data = layer.self_attn.lambda_k1.data
        # self.lambda_q2.data = layer.self_attn.lambda_q2.data
        # self.lambda_k2.data = layer.self_attn.lambda_k2.data

        self.wq_lora_A1.weight.data = layer.self_attn.wq_lora_A1.weight.data
        self.wq_lora_A2.weight.data = layer.self_attn.wq_lora_A2.weight.data
        self.wq_lora_B1.weight.data = layer.self_attn.wq_lora_B1.weight.data
        self.wq_lora_B2.weight.data = layer.self_attn.wq_lora_B2.weight.data
        self.wk_lora_A1.weight.data = layer.self_attn.wk_lora_A1.weight.data
        self.wk_lora_A2.weight.data = layer.self_attn.wk_lora_A2.weight.data
        self.wk_lora_B1.weight.data = layer.self_attn.wk_lora_B1.weight.data
        self.wk_lora_B2.weight.data = layer.self_attn.wk_lora_B2.weight.data

    def init_weights(self):
        """
        Init LoRA Bs to 0
        Init lambdas close to 0 with normal distrib
        """
        # same init as https://github.com/huggingface/peft/blob/a4f35971cda2bace54b297ad797ebc98a8f50292/src/peft/tuners/lora/layer.py#L158
        nn.init.kaiming_uniform_(self.wq_lora_A1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wq_lora_A2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wk_lora_A1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wk_lora_A2.weight, a=math.sqrt(5))

        nn.init.zeros_(self.wq_lora_B1.weight)
        nn.init.zeros_(self.wq_lora_B2.weight)
        nn.init.zeros_(self.wk_lora_B1.weight)
        nn.init.zeros_(self.wk_lora_B2.weight)

        # nn.init.normal_(self.lambda_q1, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_k1, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_q2, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_k2, mean=0, std=0.01)
