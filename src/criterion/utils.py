import torch

def reduce_logging_outputs(logging_outputs, value_key, count_key):
    tot = sum([
        log[value_key] * log[count_key] for log in logging_outputs
    ])
    cnt = sum([log[count_key] for log in logging_outputs])
    return tot / cnt


def masked_mean_flat(tensor, ignore_mask):
    if ignore_mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    assert len(ignore_mask.shape) <= len(tensor.shape)
    assert ignore_mask.shape == tensor.shape[:len(ignore_mask.shape)]
    full_size = torch.prod(torch.tensor(tensor.shape[1:]))
    pad_size = torch.prod(torch.tensor(tensor.shape[len(ignore_mask.shape):]))

    expanded_size = list(ignore_mask.shape) + [1] * (len(tensor.shape) - len(ignore_mask.shape))
    
    tensor = tensor.masked_fill(ignore_mask.view(expanded_size), 0)
    tensor = tensor.sum(dim=list(range(1, len(tensor.shape))))
    size = full_size - pad_size * ignore_mask.sum(dim=list(range(1, len(ignore_mask.shape))))
    return tensor / size