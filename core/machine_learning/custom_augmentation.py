import torch


class RandomChannelShuffle(object):
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.probability >= torch.rand(1):
            shuffled_indices = torch.randperm(tensor.shape[1])
            tensor = tensor.index_select(1, shuffled_indices)
        
        return tensor