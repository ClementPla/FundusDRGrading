import torch


def spatial_batch_normalization(batch: torch.Tensor):
    dims = batch.ndim - 1
    min_value = torch.amin(batch, (dims - 1, dims), keepdim=True)
    max_value = torch.amax(batch, (dims - 1, dims), keepdim=True)
    return (batch - min_value) / (max_value - min_value)
