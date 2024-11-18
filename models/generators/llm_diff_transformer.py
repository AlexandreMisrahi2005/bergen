import random
import os
import json
import gc
import torch
import warnings

from omegaconf import OmegaConf

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM
from models.generators.DiffTransformer.llama_lora_diff_transformer_config import LlamaLoraDiffTransformerConfig

from models.generators.generator import Generator
from models.generators.llm import LLM as BaseLLM

random.seed(42)


class LLMDiffTransformer(BaseLLM):
    """
    Slightly different from LLM as the new architecture is not necessarily pre-trained so we cannot use AutoModel.from_pretrained() directly.
    We need to treat cases for different initialization of the model.
    """
    def __init__(self, 
                model_name: str = None,
                batch_size: int = 1, 
                max_new_tokens: int = 1, 
                max_doc_len: int = 100,
                max_length: int = None,
                prompt: str = None,
                quantization: str = None,
                model_config: dict = None,
                attn_implementation: str = "eager",
                base_model_name: str = None,
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

        if attn_implementation != "eager" and model_config.layers_to_transform != list(range(0,32)):
            warnings.warn("Attn implementation is not 'eager' and not all attention layers are set to differential attention; the model might run but generate degraded results.")
        
        if model_name is not None: # we are loading a model pre-trained with the custom architecture. nothing to do, same as BaseLLM
            if quantization == "int4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quant_config,
                        attn_implementation=attn_implementation,
                        torch_dtype=torch.bfloat16,
                        device_map='auto',
                    )
            # TODO: add other quantization cases
            else:
                self.model = LlamaLoraDiffTransformerForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_name, attn_implementation=attn_implementation, torch_dtype=torch.bfloat16, device_map='auto')
                # self.model = AutoModelForCausalLM.from_pretrained(
                #         self.model_name,
                #         attn_implementation=attn_implementation,
                #         torch_dtype=torch.bfloat16,
                #         device_map='auto',
                #     )

        elif base_model_name is not None: # otherwise we initialize the model and possibly load the base model weights
            # note in this case we do not implement quantization. If really we want quantization we would have to first load the model using the methods below (without quantization), then use save_pretrained() to save the model (without quantization) and then load it with quantization

            self.model_config = OmegaConf.to_container(model_config)
            self.verbose = model_config.verbose
            assert base_model_name is not None, "`diff_attn_init_with_base_weights` is True but `base_model_name` is not provided."
            # here we load the base model, for example llama3-8b, and in load_diff_attn() we will save the required weights for proper initialization of the differential attention modules
            diff_transformer_config = OmegaConf.to_container(model_config)
            base_config = AutoConfig.from_pretrained(base_model_name) # some transformers config like LlamaConfig
            # concatenate base model config and diff transformer config to make a copy of the base model with the differential attention layers
            concat_config = {**base_config.to_dict(), **diff_transformer_config}

            base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, 
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )
            
            # self.model = self.load_diff_attn_model(base_model, concat_config)
            config = LlamaLoraDiffTransformerConfig(**concat_config)
            self.model = LlamaLoraDiffTransformerForCausalLM(config, base_model=base_model).to(base_model.device)
            

            del base_model
            gc.collect()
            torch.cuda.empty_cache()

            if self.verbose:
                print("=== DIFF ATTN MODEL LOADED ===")
                self.print_layers()

        else:
            raise ValueError("Model not found. Please provide a model name or a path to a local checkpoint.")
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", clean_up_tokenization_spaces=True)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.bos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.bos_token
        elif self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.bfloat16()

        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.prompt = prompt

    def print_layers(self):
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("trainable params: ", trainable_params, "     ||      total params: ", total_params)
        print("percent trainable: ", 100 * trainable_params / total_params, '%')
        # at which layer are trainable params?
        if len(set([(sum(p.numel() for p in layer.parameters() if p.requires_grad), sum(p.numel() for p in layer.parameters())) for layer in self.model.model.layers])) == 1:
            print("All layers have ", sum(p.numel() for p in self.model.model.layers[0].parameters() if p.requires_grad), " trainable params out of ", sum(p.numel() for p in self.model.model.layers[0].parameters()), " total params")
        else:
            for i,layer in enumerate(self.model.model.layers):
                num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                if num_params > 0:
                    print("layer ", i, " has ", num_params, " trainable params out of ", sum(p.numel() for p in layer.parameters()), " total params")