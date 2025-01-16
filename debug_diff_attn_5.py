
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaForCausalLM, LlamaModel, LlamaAttention, LlamaFlashAttention2, LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from models.generators.DiffTransformer.flash_attn import flash_attn_func
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlamaCustomAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

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
        # if self.layer_idx in [0]:
        #     print(f"BASE | layer {self.layer_idx} | hidden_states[-1, -1, :10] = ", hidden_states[-1,-1,:10])
        #     global BASE_layer0_hidden_states
        #     BASE_layer0_hidden_states = hidden_states
        # if self.layer_idx in [1]:
        #     print(f"BASE | layer {self.layer_idx} | hidden_states[-1, -1, :10] = ", hidden_states[-1,-1,:10])
        #     global BASE_layer1_hidden_states
        #     BASE_layer1_hidden_states = hidden_states
        if self.config.pretraining_tp > 1:
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # if self.layer_idx in [0,1]:
        #     print(f"BASE | layer {self.layer_idx} | query_states[-1, -1, -1, :10] = ", query_states[-1,-1,-1,:10])
        #     print(f"BASE | layer {self.layer_idx} | key_states[-1, -1, -1, :10] = ", key_states[-1,-1,-1,:10])

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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # if self.layer_idx in [0]:
        #     print(f"BASE | layer {self.layer_idx} | query_states[-1,-1,-1,:10] = ", query_states[-1,-1,-1,:10])
        #     print(f"BASE | layer {self.layer_idx} | key_states[-1,-1,-1,:10] = ", key_states[-1,-1,-1,:10])
        #     global BASE_layer0_query_states
        #     BASE_layer0_query_states = query_states
        #     global BASE_layer0_key_states
        #     BASE_layer0_key_states = key_states
        # if self.layer_idx in [1]:
        #     print(f"BASE | layer {self.layer_idx} | query_states[-1,-1,-1,:10] = ", query_states[-1,-1,-1,:10])
        #     print(f"BASE | layer {self.layer_idx} | key_states[-1,-1,-1,:10] = ", key_states[-1,-1,-1,:10])
        #     global BASE_layer1_query_states
        #     BASE_layer1_query_states = query_states
        #     global BASE_layer1_key_states
        #     BASE_layer1_key_states = key_states
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # if self.layer_idx in [0]:
        #     print(f"BASE | layer {self.layer_idx} | attn_weights[-1,-1,-1,:10] = ", attn_weights[-1,-1,-1,:10])
        #     global BASE_layer0_attn_weights
        #     BASE_layer0_attn_weights = attn_weights
        # if self.layer_idx in [1]:
        #     print(f"BASE | layer {self.layer_idx} | attn_weights[-1,-1,-1,:10] = ", attn_weights[-1,-1,-1,:10])
        #     global BASE_layer1_attn_weights
        #     BASE_layer1_attn_weights = attn_weights

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # if self.layer_idx == 0:
            #     print("ADD ATTENTION MASK")
            #     print("attention_mask.shape = ", attention_mask.shape)
            #     print("causal_mask.shape = ", causal_mask.shape)
            #     print("causal_mask = ", causal_mask)
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        # if self.layer_idx == 0:
        #     print("QUERY STATES DTYPE = ", query_states.dtype)
        #     global BASE_layer0_attn_weights_beforesoftmax
        #     BASE_layer0_attn_weights_beforesoftmax = attn_weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # if self.layer_idx == 0:
        #     print("training? ", self.training)
        #     global BASE_layer0_attn_weights_softmax
        #     BASE_layer0_attn_weights_softmax = attn_weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # if self.layer_idx == 0:
        #     global BASE_layer0_attn_weights_dropout
        #     BASE_layer0_attn_weights_dropout = attn_weights
        attn_output = torch.matmul(attn_weights, value_states)
        # if self.layer_idx in [0]:
        #     print(f"BASE | layer {self.layer_idx} | attn_output[-1,-1,-1,:10] = ", attn_output[-1,-1,-1,:10])
        #     global BASE_layer0_attn_output
        #     BASE_layer0_attn_output = attn_output

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
            # if self.layer_idx in [0,1]:
            #     print(f"BASE | layer {self.layer_idx} | attn_output[-1, -1, :10] = ", attn_output[-1,-1,:10])

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class DiffAttentionMixin:
    def init_diff_attn_lora(self):
        """ same init as https://github.com/huggingface/peft/blob/a4f35971cda2bace54b297ad797ebc98a8f50292/src/peft/tuners/lora/layer.py#L158 """

        if not self.lora_negative_term_only:
            nn.init.kaiming_uniform_(self.wq_lora_A1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_A1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.wq_lora_B1.weight)
            nn.init.zeros_(self.wk_lora_B1.weight)

        if self.negative_term_full_dim:
            nn.init.kaiming_uniform_(self.wq_2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_2.weight, a=math.sqrt(5))
        elif self.negative_term_lora_only: # if adapters only gradients don't propagate if B is 0
            nn.init.kaiming_uniform_(self.wq_lora_B2.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_B2.weight, a=math.sqrt(5))
        else:
            nn.init.zeros_(self.wq_lora_B2.weight)
            nn.init.zeros_(self.wk_lora_B2.weight)
        nn.init.kaiming_uniform_(self.wq_lora_A2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wk_lora_A2.weight, a=math.sqrt(5))

    def reset_weights_from_base_model(self):
        """
        Reset the weights W_Q, W_K, W_V, W_O
        """
        nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.o_proj.weight, a=math.sqrt(5))

    def set_weights_from_base_model(self, base_model_layer_attn):
        """
        Set the weights W_Q, W_K, W_V, W_O to the base model weights
        """
        self.q_proj.weight.data = base_model_layer_attn.q_proj.weight.data.clone()
        self.k_proj.weight.data = base_model_layer_attn.k_proj.weight.data.clone()
        self.v_proj.weight.data = base_model_layer_attn.v_proj.weight.data.clone()
        self.o_proj.weight.data = base_model_layer_attn.o_proj.weight.data.clone()

    def freeze_parameters(self):
        """
        Freeze all parameters except for the LoRA parameters
        """
        for name, param in self.named_parameters():
            if 'lambda' in name or 'lora' in name or 'subln' in name or (self.negative_term_full_dim and ('wq_2' in name or 'wk_2' in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def extra_repr(self):
        """
        overloads the nn.Module method to include lambdas and/or other diff-attn stuff when printing model 
        (some stuff is not printed by default because they are not named submodules, just parameters)
        """
        lambdas_repr = ""
        if self.learn_lambda:
            lambdas_repr = f"(lambda_q1): Parameter({self.lambda_q1.shape})\n(lambda_k1): Parameter({self.lambda_k1.shape})\n(lambda_q2): Parameter({self.lambda_q2.shape})\n(lambda_k2): Parameter({self.lambda_k2.shape})"
        else:
            lambdas_repr = f"(lambda_fixed): {self.lambda_init}"
        if self.relu:
            lambdas_repr += f"\n(relu_on_differential): {self.relu}"
        return  lambdas_repr


class LlamaLoraCustomDiffAttention(LlamaCustomAttention, DiffAttentionMixin):
    """Multi-headed differential attention from 'Differential Transformer' paper: https://arxiv.org/abs/2410.05258"""

    def __init__(self, config: LlamaLoraDiffTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.learn_lambda = config.learn_lambda
        if self.learn_lambda:
            self.lambda_init = lambda_init_fn(layer_idx)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
        else:
            self.lambda_init = config.diff_attn_lambda

        self.lora_negative_term_only = config.lora_negative_term_only
        self.negative_term_lora_only = config.negative_term_lora_only
        self.negative_term_full_dim = config.negative_term_full_dim
        if not self.lora_negative_term_only:
            self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B1 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B1 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        if self.negative_term_full_dim:
            self.wq_2 = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
            self.wk_2 = nn.Linear(self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B2 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B2 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r if config.attention_lora_r is not None and config.attention_lora_alpha is not None else 1.0
        self.subln = LlamaRMSNorm(self.head_dim, eps=1e-5) if config.groupnorm else None
        self.relu = nn.ReLU() if config.relu_on_differential else None
        
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
        bsz, q_len, _ = hidden_states.size() # X = (batch, q_len, hidden_dim) where hidden_dim = num_heads * head_dim; note in QA task q_len > 1 in first pass and q_len=1 in next passes (cached context);
        # if self.layer_idx in [0]:
        #     print(f"DIFF | layer {self.layer_idx} | hidden_states[-1, -1, :10] = ", hidden_states[-1,-1,:10])
        #     global DIFF_layer0_hidden_states
        #     DIFF_layer0_hidden_states = hidden_states
        # if self.layer_idx in [1]:
        #     print(f"DIFF | layer {self.layer_idx} | hidden_states[-1, -1, :10] = ", hidden_states[-1,-1,:10])
        #     global DIFF_layer1_hidden_states
        #     DIFF_layer1_hidden_states = hidden_states
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaDiffAttention")

        else:
            query_states = self.q_proj(hidden_states) # X @ W_q = (b, q_len, hidden_dim) @ (hidden_dim, self.num_heads * self.head_dim) = (b, q_len, self.num_heads * self.head_dim)  where self.num_heads * self.head_dim = hidden_dim = 32*128 = 4096
            key_states = self.k_proj(hidden_states)   # X @ W_k = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            value_states = self.v_proj(hidden_states) # X @ W_v = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            # if self.layer_idx in [0,1]:
            #     print(f"DIFF | layer {self.layer_idx} | query_states[-1, -1, :10] = ", query_states[-1,-1,:10])
            #     print(f"DIFF | layer {self.layer_idx} | key_states[-1, -1, :10] = ", key_states[-1,-1,:10])
            #     print(f"DIFF | layer {self.layer_idx} | value_states[-1, -1, :10] = ", value_states[-1,-1,:10])
            assert all((
                query_states.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                key_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                value_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            )), f"query_states.size() = {query_states.size()}, key_states.size() = {key_states.size()}, value_states.size() = {value_states.size()}"

            # compute adapter query/key states
            if not self.lora_negative_term_only:
                lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
                lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            if self.negative_term_full_dim:
                lora_query_states_2 = self.wq_2(self.lora_dropout(hidden_states))
                lora_key_states_2 = self.wk_2(self.lora_dropout(hidden_states))
            else:
                lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
                lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1
            if not self.lora_negative_term_only:
                assert all((
                    lora_query_states_1.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                    lora_query_states_2.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                    lora_key_states_1.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                    lora_key_states_2.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                )), f"lora_query_states_1.size() = {lora_query_states_1.size()}, lora_query_states_2.size() = {lora_query_states_2.size()}, lora_key_states_1.size() = {lora_key_states_1.size()}, lora_key_states_2.size() = {lora_key_states_2.size()}"
            
            # add adapter query/key states to original query/key states
            if not self.lora_negative_term_only:
                query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
                key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
            else:
                query_states_1 = query_states
                key_states_1 = key_states
            if self.negative_term_lora_only or self.negative_term_full_dim:
                query_states_2 = lora_query_states_2
                key_states_2 = lora_key_states_2
            else:
                query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
                key_states_2 = key_states + lora_key_states_2 # (b, q_len, self.num_key_value_heads * self.head_dim)

        query_states_1 = query_states_1.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_heads, q_len, head_dim)
        key_states_1 = key_states_1.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        key_states_2 = key_states_2.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # reshape to (b, num_key_value_heads, q_len, head_dim)

        # if self.layer_idx in [0,1]:
        #     print(f"DIFF | layer {self.layer_idx} | query_states_1[-1, -1, -1, :10] = ", query_states_1[-1,-1, -1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | query_states_2[-1, -1, -1, :10] = ", query_states_2[-1,-1, -1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_1[-1, -1, -1, :10] = ", key_states_1[-1,-1, -1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_2[-1, -1, -1, :10] = ", key_states_2[-1,-1, -1,:10])

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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # concatenate key_states along the head_dim dimension for cache 
            # this way we can store both key_1 and key_2 in the cache without modifying how cache works
            key_states_cache = torch.cat([key_states_1, key_states_2], dim=-1) # concat to shape (b, num_key_value_heads, q_len, 2 * head_dim)
            key_states_cache, value_states = past_key_value.update(key_states_cache, value_states, self.layer_idx, cache_kwargs)
            total_q_len = key_states_cache.size(-2)
            key_states_1, key_states_2 = key_states_cache.split(self.head_dim, dim=-1) # split each key back to shape (b, num_key_value_heads, q_len, head_dim)
            assert all((
                key_states_1.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
                key_states_2.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
                value_states.size() == torch.Size([bsz, self.num_key_value_heads, total_q_len, self.head_dim]),
            )), f"key_states_1.size() = {key_states_1.size()}, key_states_2.size() = {key_states_2.size()}, value_states.size() = {value_states.size()}"
        # if self.layer_idx in [0,1]:
            # print(f"DIFF | layer {self.layer_idx} | attn_output[-1, -1, -1, :10] = ", attn_output[-1,-1,-1,:10])
        total_q_len = key_states_1.size(-2)
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

        # if self.layer_idx in [0]:
        #     print(f"DIFF | layer {self.layer_idx} | query_states_1[-1,-1,-1,:10] = ", query_states_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | query_states_2[-1,-1,-1,:10] = ", query_states_2[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_1[-1,-1,-1,:10] = ", key_states_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_2[-1,-1,-1,:10] = ", key_states_2[-1,-1,-1,:10])
        #     global DIFF_layer0_query_states_1
        #     DIFF_layer0_query_states_1 = query_states_1
        #     global DIFF_layer0_key_states_1
        #     DIFF_layer0_key_states_1 = key_states_1
        # if self.layer_idx in [1]:
        #     print(f"DIFF | layer {self.layer_idx} | query_states_1[-1,-1,-1,:10] = ", query_states_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | query_states_2[-1,-1,-1,:10] = ", query_states_2[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_1[-1,-1,-1,:10] = ", key_states_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | key_states_2[-1,-1,-1,:10] = ", key_states_2[-1,-1,-1,:10])
        #     global DIFF_layer1_query_states_1
        #     DIFF_layer1_query_states_1 = query_states_1
        #     global DIFF_layer1_key_states_1
        #     DIFF_layer1_key_states_1 = key_states_1
        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim) # (b, num_heads, q_len, head_dim) @ (b, num_heads, q_len, head_dim).T(2,3) -> (b, num_heads, q_len, q_len)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim) # same
        # if self.layer_idx in [0]:
        #     print(f"DIFF | layer {self.layer_idx} | attn_weights_1[-1,-1,-1,:10] = ", attn_weights_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | attn_weights_2[-1,-1,-1,:10] = ", attn_weights_2[-1,-1,-1,:10])
        #     global DIFF_layer0_attn_weights_1
        #     DIFF_layer0_attn_weights_1 = attn_weights_1
        # if self.layer_idx in [1]:
        #     print(f"DIFF | layer {self.layer_idx} | attn_weights_1[-1,-1,-1,:10] = ", attn_weights_1[-1,-1,-1,:10])
        #     print(f"DIFF | layer {self.layer_idx} | attn_weights_2[-1,-1,-1,:10] = ", attn_weights_2[-1,-1,-1,:10])
        #     global DIFF_layer1_attn_weights_1
        #     DIFF_layer1_attn_weights_1 = attn_weights_1
        assert all((
            attn_weights_1.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
            attn_weights_2.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
        )), f"attn_weights_1.size() = {attn_weights_1.size()}, attn_weights_2.size() = {attn_weights_2.size()}"

        if attention_mask is not None:  # no matter the length, we just slice it
            # TODO: this can if loop probably be removed since attn_implementation bug is fixed
            # if attention_mask.dim() == 2: # depending on gpu type and inference setup (training or not, flash-attention, etc) attention mask can be 2D or 4D
            #     # Expand attention mask to 4D
            #     attention_mask = attention_mask[:, None, None, :].expand(-1, 1, hidden_states.size(1), -1) # TODO: check this is correct (+ implement flash attention)
            causal_mask = attention_mask[:, :, :, : key_states_1.shape[-2]] # (b, 1, q_len, q_len)
            # if self.layer_idx == 0:
            #     print("ADD ATTENTION MASK")
            #     print("attention_mask.shape = ", attention_mask.shape)
            #     print("causal_mask.shape = ", causal_mask.shape)
            #     print("causal_mask = ", causal_mask)
            attn_weights_1 = attn_weights_1 + causal_mask
            attn_weights_2 = attn_weights_2 + causal_mask

        # upcast attention to fp32
        # if self.layer_idx == 0:
        #     print("QUERY STATES DTYPE = ", query_states.dtype)
        #     global DIFF_layer0_attn_weights_1_beforesoftmax
        #     DIFF_layer0_attn_weights_1_beforesoftmax = attn_weights_1
        attn_weights_1 = nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_2 = nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print("training? ", self.training)
        # if self.layer_idx in [0]:
        #     global DIFF_layer0_attn_weights_1_softmax
        #     DIFF_layer0_attn_weights_1_softmax = attn_weights_1
        attn_weights_1 = nn.functional.dropout(attn_weights_1, p=self.attention_dropout, training=self.training)
        attn_weights_2 = nn.functional.dropout(attn_weights_2, p=self.attention_dropout, training=self.training)
        # if self.layer_idx in [0]:
        #     global DIFF_layer0_attn_weights_1_dropout
        #     DIFF_layer0_attn_weights_1_dropout = attn_weights_1

        if self.learn_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
        else:
            lambda_full = self.lambda_init

        attn_weights = attn_weights_1 - lambda_full * attn_weights_2 # diff attn
        # if self.layer_idx in [0]:
        #     global DIFF_layer0_attn_weights_diff
        #     DIFF_layer0_attn_weights_diff = attn_weights
        if self.relu:
            attn_weights = self.relu(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states) # (b, num_heads, q_len, q_len) @ (b, num_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
        # if self.layer_idx in [0]:
        #     global DIFF_layer0_attn_output
        #     DIFF_layer0_attn_output = attn_output
        # if self.layer_idx in [1]:
        #     print(f"DIFF | layer {self.layer_idx} | attn_output[-1, -1, -1, :10] = ", attn_output[-1,-1,-1,:10])

        assert attn_output.size() == torch.Size([bsz, self.num_heads, q_len, self.head_dim]), f"attn_output.size() = {attn_output.size()}"
        # GroupNorm is layer normalization but applied to each head independently
        if self.subln is not None:
            attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous() # (b, num_heads, q_len, head_dim) -> (b, q_len, num_heads, head_dim)

        attn_output = attn_output.reshape(bsz, q_len, -1) # (b, q_len, num_heads, head_dim) -> (b, q_len, num_heads * head_dim) = (b, q_len, hidden_dim)

        attn_output = self.o_proj(attn_output) # (b, q_len, hidden_dim) @ (hidden_dim, hidden_dim) = (b, q_len, hidden_dim)
        # if self.layer_idx in [0,1]:
        #     print(f"DIFF | layer {self.layer_idx} | attn_output[-1, -1, :10] = ", attn_output[-1,-1,:10])
        if not output_attentions:
            attn_weights = None
        elif output_attentions:
            attn_weights = torch.cat([attn_weights.detach().clone().unsqueeze(2), attn_weights_1.detach().clone().unsqueeze(2), lambda_full * attn_weights_2.detach().clone().unsqueeze(2)], dim=2) # (b, num_heads, q_len, q_len) -> (b, num_heads, 2, q_len, q_len)

        return attn_output, attn_weights, past_key_value

class LlamaLoraCustomFlashDiffAttention2(LlamaFlashAttention2, DiffAttentionMixin):
    """Flash attention implementation using https://github.com/xiayuqing0622/flex_head_fa """

    def __init__(self, config: LlamaLoraDiffTransformerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.learn_lambda = config.learn_lambda
        if self.learn_lambda:
            self.lambda_init = lambda_init_fn(layer_idx)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
            self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1), requires_grad=True)
        else:
            self.lambda_init = config.diff_attn_lambda

        self.lora_negative_term_only = config.lora_negative_term_only
        self.negative_term_lora_only = config.negative_term_lora_only
        self.negative_term_full_dim = config.negative_term_full_dim
        if not self.lora_negative_term_only:
            self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B1 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B1 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        if self.negative_term_full_dim:
            self.wq_2 = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
            self.wk_2 = nn.Linear(self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim, bias=False)
        else:
            self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B2 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B2 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r if config.attention_lora_r is not None and config.attention_lora_alpha is not None else 1.0
        self.subln = LlamaRMSNorm(self.head_dim, eps=1e-5) if config.groupnorm else None
        self.deterministic_backward = config.fa_deterministic_backward

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
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()  # X = (batch, q_len, hidden_dim) where hidden_dim = num_heads * head_dim; note in QA task q_len > 1 in first pass and q_len=1 in next passes (cached context);

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if not self.lora_negative_term_only:
            lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
            lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
        if self.negative_term_full_dim:
            lora_query_states_2 = self.wq_2(self.lora_dropout(hidden_states))
            lora_key_states_2 = self.wk_2(self.lora_dropout(hidden_states))
        else:
            lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
            lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1

        if not self.lora_negative_term_only:
            query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
            key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
        else:
            query_states_1 = query_states
            key_states_1 = key_states
        if self.negative_term_lora_only or self.negative_term_full_dim:
            query_states_2 = lora_query_states_2
            key_states_2 = lora_key_states_2
        else:
            query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
            key_states_2 = key_states + lora_key_states_2 # (b, q_len, self.num_key_value_heads * self.head_dim)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
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

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states_1 = query_states_1.transpose(1, 2)
        query_states_2 = query_states_2.transpose(1, 2)
        key_states_1 = key_states_1.transpose(1, 2)
        key_states_2 = key_states_2.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            # query_states = query_states.to(target_dtype)
            # key_states = key_states.to(target_dtype)
            query_states_1 = query_states_1.to(target_dtype)
            query_states_2 = query_states_2.to(target_dtype)
            key_states_1 = key_states_1.to(target_dtype)
            key_states_2 = key_states_2.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output_1 = flash_attn_func(query_states_1, key_states_1, value_states, dropout_p=dropout_rate, causal=True, deterministic=self.deterministic_backward)
        attn_output_2 = flash_attn_func(query_states_2, key_states_2, value_states, dropout_p=dropout_rate, causal=True, deterministic=self.deterministic_backward)

        if self.learn_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
        else:
            lambda_full = self.lambda_init

        # Differential Attn: A = (sm(Q1K1^T/sqrt(d)) - lambda * sm(Q2K2^T/sqrt(d))) @ V
        #              = sm(Q1K1^T/sqrt(d) @ V - lambda * sm(Q2K2^T/sqrt(d)) @ V
        #              = flashattention(Q1, K1, V) - lambda * flashattention(Q2, K2, V)
        attn_output = attn_output_1 - lambda_full * attn_output_2

        if self.subln is not None:
            attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaLoraCustomDiffAttention,
    "flash_attention_2": LlamaLoraCustomFlashDiffAttention2,
}

class LlamaLoraDiffTransformerModel(LlamaModel):
    config_class = LlamaLoraDiffTransformerConfig

    def __init__(self, config: LlamaLoraDiffTransformerConfig):
        super().__init__(config)
        for layer_idx,layer in enumerate(self.layers):
            if isinstance(layer.self_attn, LlamaAttention) and layer_idx in config.layers_to_transform:
                # HF might set config._attn_implementation to 'sdpa' by default when loading a checkpoint so we need a custom attribute for diff attn implementation
                if hasattr(config, "diff_attn_implementation"):
                    attn_implementation = config.diff_attn_implementation
                elif hasattr(config, "_attn_implementation"):
                    attn_implementation = config._attn_implementation
                else:
                    print(f"WARNING: [loading attn at layer {layer_idx}] no attn implementation found in config. Setting it to 'eager'.")
                    attn_implementation = "eager"
                if attn_implementation not in LLAMA_ATTENTION_CLASSES:
                    print(f"WARNING: [loading attn at layer {layer_idx}] attn implementation `{attn_implementation}` is unknown or not implemented for diff attention. Setting it to 'eager'.")
                    attn_implementation = "eager"
                layer.self_attn = LLAMA_ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)

class LlamaLoraDiffTransformerForCausalLM(LlamaForCausalLM, GenerationMixin):
    # edit base __init__ to change self.model + freeze params + load base model weights + other configs if any
    def __init__(self, config: LlamaLoraDiffTransformerConfig, base_model: LlamaForCausalLM = None):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaLoraDiffTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for layer in self.model.layers:
            if isinstance(layer.self_attn, LlamaLoraCustomDiffAttention):
                layer.self_attn.init_diff_attn_lora()
        print(f"Initialized diff attn weights for layer(s) {config.layers_to_transform}")

        # freeze all params (except attention)
        for _, param in self.named_parameters():
            param.requires_grad = False
        for i,layer in enumerate(self.model.layers):
            if i in config.layers_to_transform:
                layer.self_attn.freeze_parameters() # this activates the LoRA parameters
            else:
                for _, param in layer.named_parameters():
                    param.requires_grad = False

        # load the state dict of the base model everywhere (except the custom parameters of the attention layers)
        if base_model:
            self.load_base_weights(base_model.state_dict())
            
        # optionally reset all attn weights
        if not config.diff_attn_init_with_base_weights:
            print("Resetting attn weights for layer(s) ", config.layers_to_transform)
            for i,layer in enumerate(self.model.layers):
                if i in config.layers_to_transform:
                    layer.self_attn.reset_weights_from_base_model()
        self.enable_input_require_grads()  # needed for gradient checkpointing: https://github.com/huggingface/peft/issues/137

    def load_base_weights(self, base_model_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(base_model_state_dict, strict=False) # TODO: maybe a for loop to load modules 1 by 1 and free memory so we don't store 2 models at the same time
        assert len(unexpected_keys) == 0, "Unexpected keys found in the model state dict. Please check the model architecture."
        if self.config.verbose:
            print("Loaded base weights.")
        missing_keys_without_lora_params = [key for key in missing_keys if 'lora' not in key and 'subln' not in key and 'lambda' not in key and 'wk_2' not in key and 'wq_2' not in key]
        assert len(missing_keys_without_lora_params) == 0, f"Missing keys (excluding LoRA, subln, lambda): {missing_keys_without_lora_params}"

VAR = 4

if VAR == 0:

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
    config.layers_to_transform = list(range(0, 1))
    config.diff_attn_init_with_base_weights = True
    config.lora_negative_term_only = True
    config.negative_term_lora_only = False
    config.negative_term_full_dim = True
    config.attention_lora_alpha = None
    config.attention_lora_r = None
    config.attention_lora_dropout = 0.1
    config.groupnorm = False
    config.relu_on_differential = False
    config.verbose = True


    # print(config._attn_implementation)
    model = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to("cuda").bfloat16().eval()

    # del base_model
    # gc.collect()
    # torch.cuda.empty_cache()

    print(model)
    # print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model.device)

    # Forward pass for both models
    output_model = model(input_ids).logits
    print(output_model)

    # model.save_pretrained("experiments/debug/debug_attn_implementation_save/", state_dict=model.state_dict(), safe_serialization=True)

elif VAR == 1:

    model = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/tune_diff_attn_multiqa_lowlambda_nogroupnorm/train_LoraDiffAtt_multiqa_llama38b_instruct_spladeberta_top3_basicprompt_nogroupnorm_lambda01_base_init_alllayers_rightloraonly_r512/train/checkpoint-1401", torch_dtype=torch.bfloat16, device_map='auto')
    print(model)

elif VAR == 2:
    model = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/debug/debug_attn_implementation_save", torch_dtype=torch.bfloat16, device_map='auto')
    print(model)

elif VAR == 3:
    # model1 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_negativetermloraonly_multiqa_spladeberta_top5_lmb0.5/train/checkpoint-700", torch_dtype=torch.bfloat16, device_map='auto')
    # model2 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_negativetermloraonly_multiqa_spladeberta_top5_lmb0.5/train/checkpoint-1400", torch_dtype=torch.bfloat16, device_map='auto')
    # model3 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_negativetermloraonly_multiqa_spladeberta_top5_lmb0.5/train/checkpoint-7007", torch_dtype=torch.bfloat16, device_map='auto')

    model1 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_multiqa_spladeberta_top5_0.5/train/checkpoint-700", torch_dtype=torch.bfloat16, device_map='cpu')
    model2 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_multiqa_spladeberta_top5_0.5/train/checkpoint-1400", torch_dtype=torch.bfloat16, device_map='cpu')
    model3 = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path="experiments/llama32_1B_instruct/train_LoraDiffAtt_multiqa_spladeberta_top5_0.5/train/checkpoint-7007", torch_dtype=torch.bfloat16, device_map='cpu')

    with torch.no_grad():
        for model in [model1, model2, model3]:
            # print(model)
            # print wq_lora_B2 first few elements
            print(model.model.layers[0].self_attn.wq_lora_B1.weight[0, :20].cpu())
            print(model.model.layers[0].self_attn.wq_lora_B2.weight[0, :20].cpu())

elif VAR == 4:
    # define base model with LlamaCustomAttention
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    base_model_config = LlamaConfig.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation='eager', torch_dtype=torch.bfloat16, device_map='auto').to("cuda").bfloat16().eval()
    state_dict = base_model.state_dict()
    for i,layer in enumerate(base_model.model.layers):
        if isinstance(layer.self_attn, LlamaAttention):
            layer.self_attn = LlamaCustomAttention(base_model_config, layer_idx=i).to("cuda").bfloat16().eval()
            # print("Replaced LlamaAttention with LlamaCustomAttention at layer", i)
    base_model.load_state_dict(state_dict)
    # print(base_model)
    
    # define diff transformer model
    diff_transformer_config_path = "config/generator/llama-3-8b-instruct-diff-transformer.yaml"
    model_config = OmegaConf.load(diff_transformer_config_path)
    diff_transformer_config = OmegaConf.to_container(model_config)
    base_config = AutoConfig.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    concat_config = {**base_config.to_dict(), **diff_transformer_config['init_args']['model_config']}
    
    config = LlamaLoraDiffTransformerConfig(**concat_config)
    config.learn_lambda = False
    config.diff_attn_lambda = 0.0
    config.layers_to_transform = list(range(0, 16))
    config.attn_implementation = 'eager'  # needed for internal llama stuff (e.g. attention mask)
    config.diff_attn_implementation = 'eager'
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

    model = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to("cuda").bfloat16().eval()
    # print(model)
    print(base_model.config._attn_implementation)
    print(model.config._attn_implementation)

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(base_model.device)

    # Forward pass for both models
    with torch.no_grad():
        output_base_model = base_model(input_ids).logits
        output_model = model(input_ids).logits
    
    print("\n### BASE MODEL EAGER OUTPUT ###")
    print(output_base_model)
    
    print("\n### EAGER MODEL OUTPUT ###")
    print(output_model)

    # check globals
    # print("are equal BASE_layer0_hidden_states, DIFF_layer0_hidden_states", torch.allclose(BASE_layer0_hidden_states, DIFF_layer0_hidden_states), BASE_layer0_hidden_states[-1,-1,:10], DIFF_layer0_hidden_states[-1,-1,:10])
    # diff = torch.abs(BASE_layer0_hidden_states - DIFF_layer0_hidden_states)
    # print("diff", diff[torch.where(diff > 1e-4)])
    # print(torch.sum(diff > 1e-4))
    # print(diff.shape)
    # print("are equal BASE_layer0_query_states, DIFF_layer0_query_states_1", torch.allclose(BASE_layer0_query_states, DIFF_layer0_query_states_1))
    # print("are equal BASE_layer0_key_states, DIFF_layer0_key_states_1", torch.allclose(BASE_layer0_key_states, DIFF_layer0_key_states_1))
    # print("are equal BASE_layer0_attn_weights, DIFF_layer0_attn_weights_1", torch.allclose(BASE_layer0_attn_weights, DIFF_layer0_attn_weights_1))
    # print("are equal BASE_layer0_attn_weights_beforesoftmax, DIFF_layer0_attn_weights_1_beforesoftmax", torch.allclose(BASE_layer0_attn_weights_beforesoftmax, DIFF_layer0_attn_weights_1_beforesoftmax))
    
    
    # print("are equal BASE_layer0_attn_weights_softmax, DIFF_layer0_attn_weights_1_softmax", torch.allclose(BASE_layer0_attn_weights_softmax, DIFF_layer0_attn_weights_1_softmax))
    # diff = torch.abs(BASE_layer0_attn_weights_softmax - DIFF_layer0_attn_weights_1_softmax)
    # print("diff", diff[torch.where(diff > 1e-4)])
    # print("args", torch.argwhere(diff > 1e-4))
    # print(torch.sum(diff > 1e-4))
    # print(diff.shape)
    # print(BASE_layer0_attn_weights_softmax.shape, DIFF_layer0_attn_weights_1_softmax.shape)
    
    # print("are equal BASE_layer0_attn_weights_dropout, DIFF_layer0_attn_weights_1_dropout", torch.allclose(BASE_layer0_attn_weights_dropout, DIFF_layer0_attn_weights_1_dropout))
    # print("are equal BASE_layer0_attn_weights_dropout, DIFF_layer0_attn_weights_diff", torch.allclose(BASE_layer0_attn_weights_dropout, DIFF_layer0_attn_weights_diff))
    # print("are equal BASE_layer0_attn_output, DIFF_layer0_attn_output", torch.allclose(BASE_layer0_attn_output, DIFF_layer0_attn_output))
    # print("are equal BASE_layer1_hidden_states, DIFF_layer1_hidden_states", torch.allclose(BASE_layer1_hidden_states, DIFF_layer1_hidden_states))
    # # print where the hidden states are not all close
    # diff = torch.abs(BASE_layer1_hidden_states - DIFF_layer1_hidden_states)
    # print("diff", diff[torch.where(diff > 1e-4)])
    # print(torch.sum(diff > 1e-4))
    # print(diff.shape)
    # print("are equal BASE_layer1_query_states, DIFF_layer1_query_states_1", torch.allclose(BASE_layer1_query_states, DIFF_layer1_query_states_1), BASE_layer1_query_states[-1,-1,-1,:10], DIFF_layer1_query_states_1[-1,-1,-1,:10])
    # print("are equal BASE_layer1_key_states, DIFF_layer1_key_states_1", torch.allclose(BASE_layer1_key_states, DIFF_layer1_key_states_1), BASE_layer1_key_states[-1,-1,-1,:10], DIFF_layer1_key_states_1[-1,-1,-1,:10])
    # print("are equal BASE_layer1_attn_weights, DIFF_layer1_attn_weights_1", torch.allclose(BASE_layer1_attn_weights, DIFF_layer1_attn_weights_1), BASE_layer1_attn_weights[-1,-1,-1,:10], DIFF_layer1_attn_weights_1[-1,-1,-1,:10])

    input_phrase = "Hi, what is the capital of France?"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    input_phrase = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer factoid questions."},
            {"role": "user", "content": input_phrase},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer(input_phrase, return_tensors="pt").input_ids.to(base_model.device)
    print(input_phrase)
    with torch.no_grad():
        output_base_model = base_model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_model = model.generate(input_ids, do_sample=False, max_new_tokens=10)

    print("\n### BASE MODEL EAGER OUTPUT ###")
    print(tokenizer.decode(output_base_model[0], skip_special_tokens=True))

    print("\n### EAGER MODEL OUTPUT ###")
    print(tokenizer.decode(output_model[0], skip_special_tokens=True))