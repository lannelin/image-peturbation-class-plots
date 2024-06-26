import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import (
    Float,
    jaxtyped,
)


@jaxtyped(typechecker=beartype)
def get_random_1d_direction(size: int) -> Float[torch.Tensor, " {size}"]:
    x = torch.randn(size)
    return x / torch.linalg.norm(x)


# gram schmidt
@jaxtyped(typechecker=beartype)
def get_orthogonal_1d_direction(
    u: Float[torch.Tensor, " dim1"]
) -> Float[torch.Tensor, " dim1"]:
    v = torch.randn(u.shape[0])
    proj_v = (torch.inner(v, u) / torch.inner(u, u)) * u
    u2 = v - proj_v
    u2 = u2 / torch.linalg.norm(u2)
    # ensure that orthogonal
    assert torch.isclose(torch.dot(u, u2), torch.tensor(0.0), atol=1e-6)
    return u2


@jaxtyped(typechecker=beartype)
def get_gradient_based_direction(
    model: torch.nn.Module,
    imtensor: Float[torch.Tensor, " dim1 dim2 dim3"],
    normalize: Callable[
        [Float[torch.Tensor, " dim1 dim2 dim3"]],
        Float[torch.Tensor, " dim1 dim2 dim3"],
    ],
    label: int,
    device: str,
) -> Float[torch.Tensor, " dim4"]:
    """return (unit normed) gradient of loss wrt unnormalized image"""
    model.eval()

    imtensor = imtensor.unsqueeze(0).to(device)
    imtensor.requires_grad = True
    x = normalize(imtensor)
    logits = model(x)
    target = torch.tensor([label]).to(device)
    loss = F.nll_loss(logits, target)
    model.zero_grad()
    loss.backward()

    grad = imtensor.grad.data
    d = grad.detach().cpu().reshape(-1)
    return d / torch.linalg.norm(d)
