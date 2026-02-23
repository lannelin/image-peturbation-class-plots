import torch
import torch.nn.functional as F


@torch.no_grad()
def freeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


class MaxPool2dWithIndices(torch.nn.Module):
    def __init__(self, old: torch.nn.MaxPool2d):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            return_indices=True,
            ceil_mode=old.ceil_mode,
        )

    def forward(self, x):
        values, _ = self.pool(x)
        return values  # keep original behavior


def patch_functional_maxpool2d():
    orig = F.max_pool2d

    def patched(
        input,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        # Always compute indices internally
        out, idx = orig(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices=True,
        )
        # Preserve original behavior
        return (out, idx) if return_indices else out

    F.max_pool2d = patched


def patch_pools(module: torch.nn.Module):
    """Patch all MaxPool2d layers in the model to return indices
    for proper hessian calculations"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.MaxPool2d):
            setattr(module, name, MaxPool2dWithIndices(child))
        else:
            patch_pools(child)

    patch_functional_maxpool2d()
