import torch


@torch.no_grad()
def freeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
