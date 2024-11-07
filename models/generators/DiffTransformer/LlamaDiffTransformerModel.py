'''
Some necessary subclasses of transformers Llama classes for Differential Transformer
'''

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaDiffTransformerConfig(LlamaConfig):
    def __init__(self, 
                 diff_attn_lambda: float = 0.0,
                 diff_attn_layers: List[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.diff_attn_lambda = diff_attn_lambda
        self.diff_attn_layers = diff_attn_layers

class LlamaDiffTransformerModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaDiffTransformerConfig):
        super().__init__(config)
        