from typing import List

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaLoraDiffTransformerConfig(LlamaConfig):
    model_type = "llama_lora_diff_transformer"

    def __init__(self, 
                 learn_lambda: bool = False,
                 diff_attn_lambda: float = 0.0,
                 layers_to_transform: List[int] = list(range(0,32)),
                 diff_attn_init_with_base_weights: bool = True,
                 lora_negative_term_only: bool = False,
                 attention_lora_alpha: int = 64,
                 attention_lora_r: int = 32,
                 attention_lora_dropout: float = 0.1,
                 groupnorm: bool = True,
                 verbose: bool = False,
                 **kwargs):
        """
        Args:
        - learn_lambda: Whether to learn the lambda parameter for the Diff Attn loss. Takes precedence over diff_attn_lambda.
        - diff_attn_lambda: The fixed lambda parameter for the Diff Attn loss. Ignored if learn_lambda is True.
        - layers_to_transform: List of layer indices to apply Diff Attn to.
        - diff_attn_init_with_base_weights: Whether to initialize Diff Attn weights with the base (pre-trained model) weights (for q_proj, k_proj, v_proj, and o_proj)
        - lora_negative_term_only: Whether to only apply adapters on the right/negative term of diff attn.
        - attention_lora_alpha: The alpha parameter for the LORA diff attention.
        - attention_lora_r: The rank for the adapters of diff attn.
        - attention_lora_dropout: The dropout rate for the LORA diff attention.
        - groupnorm: Whether to use GroupNorm (normalization across attention heads) (see diff attn paper).
        - verbose: Whether to print verbose logs.
        """
        if learn_lambda and diff_attn_lambda > 0.0:
            logger.warning("learn_lambda is True, but diff_attn_lambda is non-zero. Diff Attn lambdas will be learnable.")
        self.learn_lambda = learn_lambda
        self.diff_attn_lambda = diff_attn_lambda
        self.layers_to_transform = list(layers_to_transform)
        self.diff_attn_init_with_base_weights = diff_attn_init_with_base_weights
        self.lora_negative_term_only = lora_negative_term_only
        self.attention_lora_alpha = attention_lora_alpha
        self.attention_lora_r = attention_lora_r
        self.attention_lora_dropout = attention_lora_dropout
        self.groupnorm = groupnorm
        self.verbose = verbose
        super().__init__(**kwargs)

LlamaLoraDiffTransformerConfig.register_for_auto_class()