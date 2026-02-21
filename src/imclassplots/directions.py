import logging

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import (
    Float,
    jaxtyped,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype)
def get_random_1d_direction(size: int) -> Float[torch.Tensor, " {size}"]:
    x = torch.randn(size)
    return x / torch.linalg.norm(x)


# gram schmidt
@jaxtyped(typechecker=beartype)
def get_orthogonal_1d_direction(
    u: Float[torch.Tensor, " dim1"],
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
    imtensor: Float[torch.Tensor, " c h w"],
    normalize_fn: Callable[
        [Float[torch.Tensor, " c h w"]],
        Float[torch.Tensor, " c h w"],
    ],
    label: int,
    device: str,
) -> Float[torch.Tensor, " flatsize"]:
    """return (unit normed) gradient of loss wrt image"""
    model.eval()

    imtensor = imtensor.unsqueeze(0).to(device)
    imtensor.requires_grad = True
    x = normalize_fn(imtensor)
    logits = model(x)
    target = torch.tensor([label]).to(device)
    loss = F.nll_loss(logits, target)
    model.zero_grad()
    loss.backward()

    grad = imtensor.grad.data
    d = grad.detach().cpu().reshape(-1)
    # return d
    return d / torch.linalg.norm(d)


# compute hvp using autograd on cpu
@jaxtyped(typechecker=beartype)
def _calc_hvp(
    data: Float[torch.Tensor, " batch c h w"],
    grad: Float[torch.Tensor, " batch c h w"],
    v: Float[torch.Tensor, " flatsize"],
):
    v_tensor = v.view_as(data)
    hvp = torch.autograd.grad((grad * v_tensor).sum(), data, retain_graph=True)[0]
    return hvp.detach().cpu().reshape(-1)


@jaxtyped(typechecker=beartype)
def _eigvec_power_iteration(
    data: Float[torch.Tensor, " batch c h w"],
    grad: Float[torch.Tensor, " batch c h w"],
    n_power_iterations: int,
    deflation_v: Float[torch.Tensor, " flatsize"] | None = None,
    tol: float = 1e-9,
):
    """Power iteration to find top eigenvector of Hessian
    optional deflation to find subsequent eigenvectors"""

    v = torch.randn_like(grad).reshape(-1)
    v_norm = torch.linalg.norm(v)
    if v_norm < 1e-6:
        raise ValueError("Random initialization has near-zero norm, try again")
    if deflation_v is not None:
        # Ensure initial orthogonality to deflation_v
        v = v - (v * deflation_v).sum() * deflation_v
    v = v / v_norm

    lam_old = None
    for _ in tqdm(range(n_power_iterations), desc="Power iteration steps"):
        Hv = _calc_hvp(data=data, grad=grad, v=v)
        if deflation_v is not None:
            # Deflate: remove component along deflation_v each step
            Hv = Hv - (Hv * deflation_v).sum() * deflation_v

        v = Hv / torch.linalg.norm(Hv)

        # estimate eigenvalue with Rayleigh quotient
        # use for convergence check
        Hv = _calc_hvp(data=data, grad=grad, v=v)
        lam = (v * Hv).sum()

        if lam_old is not None and torch.abs(lam - lam_old) < tol * (
            1.0 + torch.abs(lam)
        ):
            break
        lam_old = lam

    Hv = _calc_hvp(data=data, grad=grad, v=v)
    lam = (v * Hv).sum()
    return lam.detach(), v.detach()


@jaxtyped(typechecker=beartype)
def get_hessian_eigenvectors(
    model: torch.nn.Module,
    imtensor: Float[torch.Tensor, " c h w"],
    normalize_fn: Callable[
        [Float[torch.Tensor, " c h w"]],
        Float[torch.Tensor, " c h w"],
    ],
    label: int,
    device: str,
    top_k: int = 2,
    n_power_iterations: int = 30,
) -> Float[torch.Tensor, " {top_k} flatsize"]:
    """return top_k eigenvectors of hessian of loss wrt image"""
    if top_k < 1 or top_k > 2:
        raise ValueError("top_k must be 1 or 2")

    model.eval()
    imtensor = imtensor.unsqueeze(0).to(device)
    imtensor.requires_grad = True
    x = normalize_fn(imtensor)
    logits = model(x)
    target = torch.tensor([label]).to(device)
    loss = F.nll_loss(logits, target)

    grad = torch.autograd.grad(loss, imtensor, create_graph=True)[0].cpu()
    e1, v1 = _eigvec_power_iteration(
        data=imtensor, grad=grad, n_power_iterations=n_power_iterations
    )
    eigvecs = [v1]
    if top_k == 2:
        e2, v2 = _eigvec_power_iteration(
            data=imtensor,
            grad=grad,
            deflation_v=v1,
            n_power_iterations=n_power_iterations,
        )
        logger.info(
            f"top eigenvalue: {e1.item():.4f}, second eigenvalue: {e2.item():.4f}"
        )

    eigvecs.append(v2)

    if top_k > 2:
        raise NotImplementedError("top_k > 2 not implemented yet")

    return torch.stack(eigvecs, dim=0)
