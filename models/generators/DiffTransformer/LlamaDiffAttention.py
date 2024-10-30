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

def lambda_init_fn(layer_idx : int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)

class LlamaLoraDiffAttention(LlamaAttention):
    """Multi-headed differential attention from 'Differential Transformer' paper: https://arxiv.org/abs/2410.05258"""

    def __init__(self, config: LlamaConfig, layer_idx: int, lora_config: dict):
        super().__init__(config, layer_idx)

        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

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

        # freeze non-LoRA parameters
        for name, param in self.named_parameters():
            if 'lambda' not in name and 'lora' not in name:
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
        bsz, q_len, _ = hidden_states.size()
        # import time
        # start = time.time()
        # print(f"hidden_states.size() = bsz,q_len,_ = {hidden_states.size()}")

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
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling
            lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling
            lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling
            lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling
            query_states_1 = query_states + lora_query_states_1
            query_states_2 = query_states + lora_query_states_2
            key_states_1 = key_states + lora_key_states_1
            key_states_2 = key_states + lora_key_states_2

        # print("time 1 == ", time.time() - start)

        query_states_1 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_1 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # print(f"query_states.size() = {query_states.size()}")
        # print(f"key_states.size() = {key_states.size()}")
        # print(f"value_states.size() = {value_states.size()}")
        # print("time 2 == ", time.time() - start)

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
        # print("after rotary emb:")
        # print(f"query_states_1.size() = {query_states_1.size()}")
        # print(f"query_states_2.size() = {query_states_2.size()}")
        # print(f"key_states_1.size() = {key_states_1.size()}")
        # print(f"key_states_2.size() = {key_states_2.size()}")
        # print("time 3 == ", time.time() - start)

        # print("past_key_value = ", past_key_value)

        # TODO: check if this is correct
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # concatenate key_states along the head_dim dimension for easier cache management 
            # this way we 'pretend' the head dim is twice is truly is, but we can store both key_1 and key_2 in the cache without modifying how cache works
            key_states_cache = torch.cat([key_states_1, key_states_2], dim=-1)
            key_states_cache, value_states = past_key_value.update(key_states_cache, value_states, self.layer_idx, cache_kwargs)
            key_states_1, key_states_2 = key_states_cache.split(self.head_dim, dim=-1)
            # print("after past_key_value:")
            # print(f"key_states_1.size() = {key_states_1.size()}")
            # print(f"key_states_2.size() = {key_states_2.size()}")

        # TODO: check if this is correct
        key_states_1 = repeat_kv(key_states_1, self.num_key_value_groups)
        key_states_2 = repeat_kv(key_states_2, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # print("after repeat_kv:")
        # print(f"query_states.size() = {query_states.size()}")
        # print(f"query_states_1.size() = {query_states_1.size()}")
        # print(f"query_states_2.size() = {query_states_2.size()}")
        # print(f"key_states.size() = {key_states.size()}")
        # print(f"key_states_1.size() = {key_states_1.size()}")
        # print(f"key_states_2.size() = {key_states_2.size()}")
        # print(f"value_states.size() = {value_states.size()}")
        # print("time 4 == ", time.time() - start)
        # split in 2 along head dimension
        # print("cuda memory: ", torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))

        # query_states = query_states.reshape(bsz, self.num_heads, 2, q_len, self.head_dim // 2)
        # seq_total_len = key_states.size(2) # need to explicit because it's usually > 1 in first pass but ==1 when past_key_value is not None
        # key_states = key_states.reshape(bsz, self.num_heads, 2, seq_total_len, self.head_dim // 2)

        # print("after reshape:")
        # print(f"query_states.size() = {query_states.size()}")
        # print(f"key_states.size() = {key_states.size()}")
        # print("time 5 == ", time.time() - start)
        # print("cuda memory: ", torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0))
        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim)

        # print(f"attn_weights.size() = {attn_weights.size()}")
        # print(f"attn_weights_1.size() = {attn_weights_1.size()}")
        # print(f"attn_weights_2.size() = {attn_weights_2.size()}")
        # print("time 6 == ", time.time() - start)

        if attention_mask is not None:  # no matter the length, we just slice it
            # print(f"attention_mask.size() = {attention_mask.size()}")
            # print(f"key_states_1.size() = {key_states_1.size()}")
            # print(f"key_states_2.size() = {key_states_2.size()}")
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]][:,:,None].repeat(1,1,2,1,1)
            causal_mask = attention_mask[:, :, :, : key_states_1.shape[-2]]
            # print(f"causal_mask.size() = {causal_mask.size()}")
            # print(f"causal_mask.size() = {causal_mask.size()}")
            attn_weights_1 = attn_weights_1 + causal_mask
            attn_weights_2 = attn_weights_2 + causal_mask
        # print("time 7 == ", time.time() - start)

        # upcast attention to fp32
        attn_weights_1 = nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_2 = nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_1 = nn.functional.dropout(attn_weights_1, p=self.attention_dropout, training=self.training)
        attn_weights_2 = nn.functional.dropout(attn_weights_2, p=self.attention_dropout, training=self.training)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights_1 - lambda_full * attn_weights_2
        attn_output = torch.matmul(attn_weights, value_states)
        # print(f"attn_output.size() = {attn_output.size()}")
        # print("time 8 == ", time.time() - start)
        # print()
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