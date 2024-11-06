import torch
from tqdm import tqdm
from typing import List

def reduce_rank_svd(matrix, reduction_factor=2):
    # Perform SVD
    _, _, Vt = torch.linalg.svd(matrix.float(), full_matrices=False)
    Vt = Vt.type_as(matrix)
    target_rank = matrix.size(0) // reduction_factor
    Vt_reduced = Vt[:target_rank, :]
    reduced_matrix = matrix @ Vt_reduced.T
    return reduced_matrix

def init_weights(model, layers: List[int] = None):
    for layer in tqdm(layers):
        for proj in ["q", "k", "v"]:
            print("Reducing rank of", model.model.layers[layer].self_attn._modules[f"{proj}_proj"].weight.data.size())
            reduced = reduce_rank_svd(
                    model.model.layers[layer].self_attn._modules[f"{proj}_proj"].weight.data
                )
            print("Reduced to", reduced.size())
            model.model.layers[layer].self_attn._modules[f"{proj}_proj"].weight.data = torch.nn.Parameter(reduced)
    return model