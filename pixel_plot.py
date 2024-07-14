import os
import random
from functools import partial

import numpy as np
import PIL
import torch
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Float
from jsonargparse import CLI
from matplotlib import pyplot as plt
from torchvision import transforms

from imclassplots.directions import (
    get_gradient_based_direction,
    get_orthogonal_1d_direction,
    get_random_1d_direction,
)
from imclassplots.peturb import (
    peturb,
    peturb_and_predict,
)
from imclassplots.plot import plot_predictions


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility"""
    random.seed(42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@beartype
def main(
    image_fpath: str,
    true_label: int,
    grid_size: int,
    scale_factor: float,
    model: torch.nn.Module,
    direction: str,
    batch_size: int,
    display_ims: bool,
    dataset_labels: list[str],
    dataset_transform: Callable[
        [PIL.Image.Image],
        Float[torch.Tensor, " dim1 dim2 dim3"],
    ],
    dataset_normalize: Callable[
        [Float[torch.Tensor, " dim1 dim2 dim3"]],
        Float[torch.Tensor, " dim1 dim2 dim3"],
    ],
    device: str = "cpu",
    random_seed: int = 42,
) -> None:
    """Evaluate image over grid of perturbations in two directions and plot.
    Saves (predictions,x_direction,y_direction,orig_img)

    Args:
        image_fpath: path to image. Will be resized to 32x32 if not already
        true_label: expected class label of image. TODO what does this mean
        grid_size: size of grid (square)
        scale_factor: scale factor for peturbations
        safetensors_fpath: path to safetensors for model
        direction: method to pick xdirection: random or gradient
        batch_size: batch size
        display_ims: visualise images alongside plot
        dataset_labels: labels used in classifier training
        dataset_transform: transform used in classifier training
        dataset_normalize: normalize used in classifier training
            (typically as part of transform)
        device: device to run model on
        random_seed: seed for random number generator
    """

    seed_everything(random_seed)

    img = PIL.Image.open(image_fpath).resize(
        (32, 32), PIL.Image.Resampling.LANCZOS
    )

    model = model.eval()
    model.to(device)

    #  directions
    if direction == "random":
        im_size = img.height * img.width * len(img.getbands())
        x_direction = get_random_1d_direction(size=im_size)
    elif direction == "gradient":
        x_direction = get_gradient_based_direction(
            model=model,
            imtensor=transforms.ToTensor()(img),
            normalize=dataset_normalize,
            label=true_label,
            device=device,
        )
    else:
        raise ValueError(
            f"direction must be random or gradient, not {direction}"
        )

    y_direction = get_orthogonal_1d_direction(u=x_direction)

    predictions = peturb_and_predict(
        image=img,
        model=model,
        label=true_label,
        grid_size=grid_size,
        data_transform=dataset_transform,
        x_direction=x_direction,
        y_direction=y_direction,
        device=device,
        batch_size=batch_size,
        scale_factor=scale_factor,
    )

    plot_directory = "./plots"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    data_fname = os.path.join(
        plot_directory,
        f"predictionsAndDirs_label"
        f"{true_label}_gridsize{grid_size}_sf{scale_factor}.pt",
    )
    torch.save(
        (predictions, x_direction, y_direction, transforms.ToTensor()(img)),
        data_fname,
    )
    print(f"saved (predictions,x_direction,y_direction) tuple at {data_fname}")

    figure_fname = os.path.join(
        plot_directory,
        f"fig_{true_label}_gridsize{grid_size}_sf{scale_factor}.png",
    )

    im_gen_fn = (
        partial(
            peturb, img=img, direction_a=x_direction, direction_b=y_direction
        )
        if display_ims
        else None
    )
    fig = plot_predictions(
        predictions=predictions,
        class_labels=dataset_labels,
        true_image_label=true_label,
        display_ims=display_ims,
        im_generation_fn=im_gen_fn,
        scale_factor=scale_factor,
    )
    fig.savefig(figure_fname)
    print(f"saved figure at {figure_fname}")
    plt.show()


if __name__ == "__main__":
    CLI(main, as_positional=False, fail_untyped=False)
