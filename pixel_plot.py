"""
TODO improve this explanation
downloads cifar 10
loads model
loads cifar image
chooses random directions
peturbs image
runs through model
plots predictions
"""

import os
from functools import partial

import PIL
import torch
from beartype import beartype
from jsonargparse import CLI
from lightning import seed_everything
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from lightning_resnet.resnet18 import ResNet18
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

CIFAR_NORMALIZE = transforms.Normalize(
    mean=[
        0.4913725490196078,
        0.4823529411764706,
        0.4466666666666667,
    ],
    std=[
        0.24705882352941178,
        0.24352941176470588,
        0.2615686274509804,
    ],
)
CIFAR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_NORMALIZE,
    ]
)

CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@beartype
def main(
    image_fpath: str,
    label: int,
    grid_size: int,
    scale_factor: float,
    safetensors_fpath: str,
    direction: str,
    batch_size: int,
    display_ims: bool,
    device: str = "auto",
    random_seed: int = 42,
) -> None:
    """Evaluate image over grid of perturbations in two directions and plot.
    Saves (predictions,x_direction,y_direction,orig_img)

    Args:
        image_fpath: path to image. Will be resized to 32x32 if not already
        label: expected class label of image. TODO what does this mean
        grid_size: size of grid (square)
        scale_factor: scale factor for peturbations
        safetensors_fpath: path to safetensors for model
        direction: method to pick xdirection: random or gradient
        batch_size: batch size
        display_ims: visualise images alongside plot
        device: device to run model on
        random_seed: seed for random number generator
    """
    seed_everything(random_seed)

    if device == "auto":
        device = _AcceleratorConnector._choose_auto_accelerator()

    img = PIL.Image.open(image_fpath).resize(
        (32, 32), PIL.Image.Resampling.LANCZOS
    )
    model = ResNet18(num_classes=10, safetensors_path=safetensors_fpath)

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
            normalize=CIFAR_NORMALIZE,
            label=label,
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
        label=label,
        grid_size=grid_size,
        data_transform=CIFAR_TRANSFORM,
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
        f"{label}_gridsize{grid_size}_sf{scale_factor}.pt",
    )
    torch.save(
        (predictions, x_direction, y_direction, transforms.ToTensor()(img)),
        data_fname,
    )
    print(f"saved (predictions,x_direction,y_direction) tuple at {data_fname}")

    figure_fname = os.path.join(
        plot_directory,
        f"fig_{label}_gridsize{grid_size}_sf{scale_factor}.png",
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
        class_labels=CIFAR_CLASSES,
        true_image_label=label,
        display_ims=display_ims,
        im_generation_fn=im_gen_fn,
        scale_factor=scale_factor,
    )
    fig.savefig(figure_fname)
    print(f"saved figure at {figure_fname}")
    plt.show()


if __name__ == "__main__":
    CLI(main, as_positional=False)
