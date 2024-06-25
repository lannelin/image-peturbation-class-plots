import PIL
import torch
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import (
    Float,
    Int,
    jaxtyped,
)
from torchvision import transforms
from tqdm.auto import tqdm

from imclassplots.directions import (
    get_orthogonal_1d_direction,
    get_random_1d_direction,
)


@jaxtyped(typechecker=beartype)
def peturb(
    img: PIL.Image.Image,
    direction_a: Float[torch.Tensor, " dim1"],
    direction_b: Float[torch.Tensor, " dim1"],
    scale_a: float,
    scale_b: float,
) -> PIL.Image.Image:
    t = transforms.ToTensor()(img)
    a_permute = (direction_a * scale_a).reshape(t.shape)
    b_permute = (direction_b * scale_b).reshape(t.shape)
    t = (t + a_permute + b_permute).clip(0, 1)
    return transforms.ToPILImage()(t)


@jaxtyped(typechecker=beartype)
def peturb_and_predict(
    image: PIL.Image.Image,
    model: torch.nn.Module,
    grid_size: int,
    label: int,
    data_transform: Callable[
        [PIL.Image.Image],
        Float[torch.Tensor, " dim1 dim2 dim3"],
    ],
    device: str,
    scale_factor: float = 1.0,
) -> tuple[
    Int[torch.Tensor, "{grid_size} {grid_size}"],
    Float[torch.Tensor, " dim4"],
    Float[torch.Tensor, " dim4"],
]:
    """
    returns: Tuple[predictions, x_direction, y_direction]
    """

    #  directions
    im_size = image.height * image.width * len(image.getbands())
    x_direction = get_random_1d_direction(size=im_size)
    y_direction = get_orthogonal_1d_direction(u=x_direction)

    # sanity check
    with torch.inference_mode():
        x = data_transform(image)
        scores = model(x.unsqueeze(0).to(device))
        prediction = torch.argmax(scores).detach().cpu()
        assert (
            prediction == label
        ), f"test image: expected {label}, got {prediction}"

    predictions = torch.zeros((grid_size, grid_size), dtype=torch.int8)

    centre = grid_size // 2
    # TODO batching
    for i in tqdm(range(grid_size), leave=True):
        x_coord = i - centre
        for j in tqdm(range(grid_size), leave=False):
            y_coord = j - centre

            permuted = peturb(
                img=image,
                direction_a=x_direction,
                direction_b=y_direction,
                scale_a=x_coord * scale_factor,
                scale_b=y_coord * scale_factor,
            )

            x = data_transform(permuted)

            with torch.inference_mode():
                scores = model(x.unsqueeze(0).to(device))
            prediction = torch.argmax(scores).detach().cpu().item()
            predictions[j, i] = prediction

    return predictions, x_direction, y_direction
