from typing import List

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaLoraDiffTransformerConfig(LlamaConfig):
    model_type = "llama_lora_diff_transformer"

    def __init__(self, 
                 diff_attn_lambda: float = 0.0,
                 layers_to_transform: List[int] = list(range(0,32)),
                 diff_attn_init_with_base_weights: bool = True,
                 lora_negative_term_only: bool = False,
                 attention_lora_alpha: int = 64,
                 attention_lora_r: int = 32,
                 attention_lora_dropout: float = 0.1,
                 verbose: bool = False,
                 **kwargs):
        self.diff_attn_lambda = diff_attn_lambda
        self.layers_to_transform = list(layers_to_transform)
        self.diff_attn_init_with_base_weights = diff_attn_init_with_base_weights
        self.lora_negative_term_only = lora_negative_term_only
        self.attention_lora_alpha = attention_lora_alpha
        self.attention_lora_r = attention_lora_r
        self.attention_lora_dropout = attention_lora_dropout
        self.verbose = verbose
        super().__init__(**kwargs)

LlamaLoraDiffTransformerConfig.register_for_auto_class()