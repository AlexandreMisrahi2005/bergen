'''
Some necessary subclasses of transformers Llama classes for Differential Transformer
'''

import os
import math
from typing import List, Optional, Tuple, Union, Callable
import warnings
import copy

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaForCausalLM, LlamaModel, LlamaAttention, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig
from transformers.cache_utils import Cache
from transformers.utils import logging

logger = logging.get_logger(__name__)

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class LlamaLoraDiffAttention(LlamaAttention):
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
        if not self.lora_negative_term_only:
            self.wq_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wq_lora_B1 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
            self.wk_lora_A1 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
            self.wk_lora_B1 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        self.wq_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
        self.wq_lora_B2 = nn.Linear(config.attention_lora_r, self.num_heads * self.head_dim, bias=False)
        self.wk_lora_A2 = nn.Linear(self.num_heads * self.head_dim, config.attention_lora_r, bias=False)
        self.wk_lora_B2 = nn.Linear(config.attention_lora_r, self.num_key_value_heads * self.head_dim, bias=False)

        self.lora_dropout = nn.Dropout(p=config.attention_lora_dropout)
        self.lora_scaling = config.attention_lora_alpha / config.attention_lora_r
        self.subln = None
        if config.groupnorm:
            self.subln = LlamaRMSNorm(self.head_dim, eps=1e-5)
        # self.freeze_parameters(config)
        
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
            query_states = self.q_proj(hidden_states) # X @ W_q = (b, q_len, hidden_dim) @ (hidden_dim, self.num_heads * self.head_dim) = (b, q_len, self.num_heads * self.head_dim)  where self.num_heads * self.head_dim = hidden_dim = 32*128 = 4096
            key_states = self.k_proj(hidden_states)   # X @ W_k = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            value_states = self.v_proj(hidden_states) # X @ W_v = (b, q_len, hidden_dim) @ (hidden_dim, self.num_key_value_heads * self.head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            assert all((
                query_states.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                key_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                value_states.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
            )), f"query_states.size() = {query_states.size()}, key_states.size() = {key_states.size()}, value_states.size() = {value_states.size()}"

            if not self.lora_negative_term_only:
                lora_query_states_1 = self.wq_lora_B1(self.wq_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wq_A1 @ wq_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, hidden_dim) = (b, q_len, hidden_dim)
            lora_query_states_2 = self.wq_lora_B2(self.wq_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as query_states_1
            if not self.lora_negative_term_only:
                lora_key_states_1 = self.wk_lora_B1(self.wk_lora_A1(self.lora_dropout(hidden_states))) * self.lora_scaling # X @ wk_A1 @ wk_B1 = (b, q_len, hidden_dim) @ (hidden_dim, r) @ (r, num_key_value_heads * head_dim) = (b, q_len, self.num_key_value_heads * self.head_dim)
            lora_key_states_2 = self.wk_lora_B2(self.wk_lora_A2(self.lora_dropout(hidden_states))) * self.lora_scaling # same as key_states_1
            if not self.lora_negative_term_only:
                assert all((
                    lora_query_states_1.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                    lora_query_states_2.size() == torch.Size([bsz, q_len, self.num_heads * self.head_dim]),
                    lora_key_states_1.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                    lora_key_states_2.size() == torch.Size([bsz, q_len, self.num_key_value_heads * self.head_dim]),
                )), f"lora_query_states_1.size() = {lora_query_states_1.size()}, lora_query_states_2.size() = {lora_query_states_2.size()}, lora_key_states_1.size() = {lora_key_states_1.size()}, lora_key_states_2.size() = {lora_key_states_2.size()}"
            
            if not self.lora_negative_term_only:
                query_states_1 = query_states + lora_query_states_1 # (b, q_len, hidden_dim)
                key_states_1 = key_states + lora_key_states_1 # (b, q_len, self.num_key_value_heads * self.head_dim)
            else:
                query_states_1 = query_states
                key_states_1 = key_states
            query_states_2 = query_states + lora_query_states_2 # (b, q_len, hidden_dim)
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

        attn_weights_1 = torch.matmul(query_states_1, key_states_1.transpose(2, 3)) / math.sqrt(self.head_dim) # (b, num_heads, q_len, head_dim) @ (b, num_heads, q_len, head_dim).T(2,3) -> (b, num_heads, q_len, q_len)
        attn_weights_2 = torch.matmul(query_states_2, key_states_2.transpose(2, 3)) / math.sqrt(self.head_dim) # same
        assert all((
            attn_weights_1.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
            attn_weights_2.size() == torch.Size([bsz, self.num_heads, q_len, total_q_len]),
        )), f"attn_weights_1.size() = {attn_weights_1.size()}, attn_weights_2.size() = {attn_weights_2.size()}"

        if attention_mask is not None:  # no matter the length, we just slice it
            # TODO: this can if loop probably be removed since attn_implementation bug is fixed
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

        if self.learn_lambda:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
        else:
            lambda_full = self.lambda_init

        attn_weights = attn_weights_1 - lambda_full * attn_weights_2 # diff attn
        attn_output = torch.matmul(attn_weights, value_states) # (b, num_heads, q_len, q_len) @ (b, num_heads, q_len, head_dim) -> (b, num_heads, q_len, head_dim)
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

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining tensor parallel not implemented for LlamaDiffAttention")
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output) # (b, q_len, hidden_dim) @ (hidden_dim, hidden_dim) = (b, q_len, hidden_dim)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def init_diff_attn_lora(self):
        """ same init as https://github.com/huggingface/peft/blob/a4f35971cda2bace54b297ad797ebc98a8f50292/src/peft/tuners/lora/layer.py#L158 """

        if not self.lora_negative_term_only:
            nn.init.kaiming_uniform_(self.wq_lora_A1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.wk_lora_A1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.wq_lora_B1.weight)
            nn.init.zeros_(self.wk_lora_B1.weight)

        nn.init.kaiming_uniform_(self.wq_lora_A2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wk_lora_A2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.wq_lora_B2.weight)
        nn.init.zeros_(self.wk_lora_B2.weight)

        # nn.init.normal_(self.lambda_q1, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_k1, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_q2, mean=0, std=0.01)
        # nn.init.normal_(self.lambda_k2, mean=0, std=0.01)

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
        Should be called from outside the model
        """
        self.q_proj.weight.data = base_model_layer_attn.q_proj.weight.data.clone()
        self.k_proj.weight.data = base_model_layer_attn.k_proj.weight.data.clone()
        self.v_proj.weight.data = base_model_layer_attn.v_proj.weight.data.clone()
        self.o_proj.weight.data = base_model_layer_attn.o_proj.weight.data.clone()

    def freeze_parameters(self, config: LlamaLoraDiffTransformerConfig = None):
        """
        Default: Freeze all parameters except for the LoRA parameters
        TODO: add cases from config
        """
        for name, param in self.named_parameters():
            if 'lambda' in name or 'lora' in name or 'subln' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def extra_repr(self):
        # overloads the nn.Module method to include lambdas when printing model (not printed by default because are not named submodules, just parameters)
        lambdas_repr = ""
        if self.learn_lambda:
            lambdas_repr = f"(lambda_q1): Parameter({self.lambda_q1.shape})\n(lambda_k1): Parameter({self.lambda_k1.shape})\n(lambda_q2): Parameter({self.lambda_q2.shape})\n(lambda_k2): Parameter({self.lambda_k2.shape})"
        return  lambdas_repr


class LlamaLoraDiffTransformerModel(LlamaModel):
    config_class = LlamaLoraDiffTransformerConfig

    def __init__(self, config: LlamaLoraDiffTransformerConfig):
        super().__init__(config)
        for layer_idx,layer in enumerate(self.layers):
            if isinstance(layer.self_attn, LlamaAttention) and layer_idx in config.layers_to_transform:
                layer.self_attn = LlamaLoraDiffAttention(config, layer_idx)

class LlamaLoraDiffTransformerForCausalLM(LlamaForCausalLM, GenerationMixin):
    # edit __init__ to change self.model to LlamaLoraDiffTransformerModel + freeze params + load base model weights
    def __init__(self, config: LlamaLoraDiffTransformerConfig, base_model: LlamaForCausalLM = None):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaLoraDiffTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for layer in self.model.layers:
            if isinstance(layer.self_attn, LlamaLoraDiffAttention):
                layer.self_attn.init_diff_attn_lora()

        # freeze all params (except attention)
        for _, param in self.named_parameters():
            param.requires_grad = False
        for i,layer in enumerate(self.model.layers):
            if i in config.layers_to_transform:
                layer.self_attn.freeze_parameters(config) # this activates the LoRA parameters
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

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # save only trainable params
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                del state_dict[name]
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token, save_peft_format, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        weights_only: bool = True,
        **kwargs,
    ) -> "PreTrainedModel":
        """
        Load base model and then load the custom model on top of it
        Not so clean, cls needs to be LlamaLoraDiffTransformerForCausalLM
        """
        config = LlamaLoraDiffTransformerConfig.from_pretrained(pretrained_model_name_or_path)
        base_model_path = config._name_or_path # something like meta-llama/Meta-Llama-3-8B-Instruct

        print("=============== YOU CAN SAFELY IGNORE THE WARNING BELOW ===============")
        model = super().from_pretrained(base_model_path, *model_args, config=config, cache_dir=cache_dir, ignore_mismatched_sizes=ignore_mismatched_sizes, force_download=force_download, local_files_only=local_files_only, token=token, revision=revision, use_safetensors=use_safetensors, weights_only=weights_only, **kwargs)
        print("=============== YOU CAN SAFELY IGNORE THE WARNING ABOVE ===============")

        # Now we load the adapters. Load all the model.safetensors files TODO: cleaner with cases if multiple shards
        adapters_state_dict = load_file(f"{pretrained_model_name_or_path}/model.safetensors")
        model.load_diff_attn_weights(adapters_state_dict)
        return model
    
    def load_base_weights(self, base_model_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(base_model_state_dict, strict=False) # TODO: maybe a for loop to load modules 1 by 1 and free memory so we don't store 2 models at the same time
        assert len(unexpected_keys) == 0, "Unexpected keys found in the model state dict. Please check the model architecture."
        if self.config.verbose:
            print("Loaded base weights.")
            missing_keys_without_lora_params = [key for key in missing_keys if 'lora' not in key and 'subln' not in key and 'lambda' not in key]
            print("Num missing keys =", len(missing_keys))
            print("Missing keys (excluding LoRA, subln, lambda): ", missing_keys_without_lora_params) # we expect no missing keys apart from the diff attn lora layers
            print("Unexpected keys: ", unexpected_keys)
            for key in missing_keys:
                assert "lora" in key or "subln" in key or "lambda" in key, f"Missing key {key}"

    def load_diff_attn_weights(self, adapters_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(adapters_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"{len(unexpected_keys)} unexpected keys found in the model state dict: \n{unexpected_keys}"
        if self.config.verbose:
            print("Loaded diff attn weights.")
            print("Num missing keys when loading adapters =", len(missing_keys))
            print("Num keys that are not LoRA, subln, lambda = ", len([key for key in self.state_dict().keys() if 'lora' not in key and 'subln' not in key and 'lambda' not in key]))

            
    def load_base_weights_and_adapters(self, concat_state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(concat_state_dict, strict=True)
        assert len(missing_keys) == 0 and len(unexpected_keys) == 0, f"Missing keys = {len(missing_keys)} || Unexpected keys = {len(unexpected_keys)}"

    def unfreeze_adapters(self):
        for name, param in self.named_parameters():
            if 'lora' in name or 'subln' in name or 'lambda' in name:
                param.requires_grad = True

LlamaLoraDiffTransformerForCausalLM.register_for_auto_class("AutoModelForCausalLM")