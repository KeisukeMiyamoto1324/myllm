import torch

LayerKeyValueCache = tuple[torch.Tensor, torch.Tensor]
KeyValueCache = list[LayerKeyValueCache]
