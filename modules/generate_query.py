from tqdm import tqdm
from hydra.utils import instantiate
import torch

class GenerateQueries:
    def __init__(self, 
                 generator=None,
                 init_args=None,
                 ):
        # instaniate model
        self.model = instantiate(init_args)
        if init_args.model == "generator":
            if generator is None:
                raise RuntimeError("No generator specified for getting search queries.")
            self.model.init_generator(generator)

    @torch.no_grad() 
    def eval(self, dataset):
        return self.model.generate(dataset["content"])

    def get_clean_model_name(self):
        return self.model.name
