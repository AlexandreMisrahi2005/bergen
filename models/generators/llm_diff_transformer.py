import random
import os
import json
import gc
import torch
import warnings

from peft import AutoPeftModelForCausalLM, PeftConfig, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from models.generators.DiffTransformer.LlamaDiffAttention import LlamaLoraDiffAttention

from utils import prepare_labels, left_pad
from models.generators.generator import Generator
from models.generators.llm import LLM as BaseLLM

random.seed(42)


class LLM(BaseLLM):
    def __init__(self, 
                model_name: str = None,
                batch_size: int = 1, 
                max_new_tokens: int = 1, 
                max_doc_len: int = 100,
                max_length: int = None,
                prompt: str = None,
                quantization: str = None,
                attn_implementation: str = "flash_attention_2",
                path: str = None, # path to a local checkpoint
                ):
        """
        :model_name: hf model name or path to a local checkpoint
        :max_new_tokens: how many tokens to generate at most
        :max_doc_len: documents are cropped to a maximum of max_doc_len words
        :gguf_file: specify to use a gguf_file (see from_pretrained)
        :local_path: forces only local reading (i.e. no hf download)
        path: path to a local checkpoint, will load the model weights from this path
        """
        Generator.__init__(self,
                           model_name=model_name,
                           batch_size=batch_size,
                           max_new_tokens=max_new_tokens,
                           max_doc_len=max_doc_len,
                           max_length=max_length)
        # check type of gpu: if not A100 then change attn implementation to sdpa
        # TODO: adapt LlamaLoraDiffAttention to work with sdpa so we can use A100 when training
        # if "A100" not in torch.cuda.get_device_name(torch.cuda.current_device):
        #     attn_implementation="sdpa"

        self.path = path
        self.model = AutoModelForCausalLM.from_pretrained(
                path if path else model_name,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
        
        self.lora_config = LoraConfig(
                target_modules=["q_proj", "k_proj"],
                lora_alpha=64,
                r=32,
                lora_dropout=0.1,
                task_type="CAUSAL_LM",
                layers_to_transform=list(range(0,32)),
                )
        
        self.load_diff_attn()
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", clean_up_tokenization_spaces=True)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.bos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.bos_token
        elif self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: quantization

        self.model = self.model.bfloat16()

        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.prompt = prompt

    def _init_weights(self, init_type="normal"):
        pass

    def load_diff_attn(self):
        """ Load model architecture, with weights if path to weights is given """

        # freeze all model parameters (including embedding and LM head)
        for param in self.model.parameters():
            param.requires_grad = False

        # apply diff attn
        print("Setting LoraDiffAttention for layers ", self.lora_config.layers_to_transform)
        if self.path:
            print("Loading model weights from ", self.path)
        for i,layer in enumerate(self.model.model.layers):
            if isinstance(layer.self_attn, LlamaAttention) and i in self.lora_config.layers_to_transform:
                layer.self_attn = LlamaLoraDiffAttention(self.model.config, layer_idx=i, lora_config=self.lora_config).to(self.model.device)
                if self.path: # load weights if model is loaded from a checkpoint
                    layer.self_attn.load_weights(self.model.model.layers[i])
            else: # freeze model parameters
                for param in layer.parameters():
                    param.requires_grad = False

        print("Model loaded.")
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("trainable params: ", trainable_params, "     ||      total params: ", total_params)
        print("percent trainable: ", 100 * trainable_params / total_params, '%')
        # at which layer are trainable params?
        for i,layer in enumerate(self.model.model.layers):
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            if num_params > 0:
                print("layer ", i, " has ", num_params, " trainable params out of ", sum(p.numel() for p in layer.parameters()), " total params")
