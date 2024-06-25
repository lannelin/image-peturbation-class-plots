import math

import PIL
import torch
from beartype import beartype
from beartype.typing import (
    Callable,
    Optional,
)
from jaxtyping import (
    Float,
    Int,
    jaxtyped,
)
from lightning import Trainer
from torch.utils.data import Dataset
from torchvision import transforms

from imclassplots.directions import (
    get_orthogonal_1d_direction,
    get_random_1d_direction,
)


class PeturbationsDataset(Dataset):
    def __init__(
        self,
        original_img: PIL.Image.Image,
        num_peturbations: int,
        x_direction: Float[torch.Tensor, " dim1"],
        y_direction: Float[torch.Tensor, " dim1"],
        scale_factor: float,
        transform: Optional[
            Callable[
                [PIL.Image.Image],
                Float[torch.Tensor, " dim1 dim2 dim3"],
            ]
        ] = None,
        x_length: Optional[int] = None,
    ):

        self.original_img = original_img
        self.num_peturbations = num_peturbations
        self.x_direction = x_direction
        self.y_direction = y_direction
        self.scale_factor = scale_factor
        self.transform = transform
        self.set_grid_geom(
            num_peturbations=num_peturbations, x_length=x_length
        )

    def set_grid_geom(self, num_peturbations: int, x_length: Optional[int]):
        if x_length is None:
            # if x_size, y_size not specified then assume square
            x = math.sqrt(num_peturbations)

            if not x.is_integer():
                raise Exception(
                    "num_peturbations must be a square number"
                    " when x_length not specified"
                )

            # set to square
            y_length = x_length = int(x)
        else:
            y_length = num_peturbations / x_length
            if not y_length.is_integer():
                raise Exception(
                    "num_peturbations must be divisible by x_length"
                )

        self.x_length = x_length
        # no need to store y_length

        # now set offset to centre
        self.x_centre = self.x_length // 2
        self.y_centre = y_length // 2

    def __len__(self):
        return self.num_peturbations

    def __getitem__(self, idx):

        x, y = divmod(idx, self.x_length)
        x = x - self.x_centre
        y = y - self.y_centre

        perturbed = peturb(
            img=self.original_img,
            direction_a=self.x_direction,
            direction_b=self.y_direction,
            scale_a=x * self.scale_factor,
            scale_b=y * self.scale_factor,
        )

        return self.transform(perturbed)


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
    batch_size: int,
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

    dataset = PeturbationsDataset(
        original_img=image,
        num_peturbations=grid_size**2,
        x_direction=x_direction,
        y_direction=y_direction,
        scale_factor=scale_factor,
        transform=data_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    trainer = Trainer(accelerator=device, logger=False)
    logits = trainer.predict(model=model, dataloaders=dataloader)

    # argmax then transpose so we align with x,y directions
    predictions = (
        torch.argmax(torch.vstack(logits), dim=1)
        .reshape((grid_size, grid_size))
        .T
    )

    return predictions, x_direction, y_direction
