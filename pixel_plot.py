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

import argparse
import os
from functools import partial

import PIL
import torch
from beartype import beartype
from lightning import seed_everything
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from lightning_resnet.resnet18 import ResNet18
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10

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

CIFAR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
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
        ),
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
def load_first_cifar_image(cifar_root_dir: str, label: int) -> PIL.Image.Image:
    data = CIFAR10(root=cifar_root_dir, train=True, download=True)
    for img, img_label in data:
        if img_label == label:
            return img
    else:
        raise Exception("didn't find a matching label in the data")


@beartype
def main(
    cifar_root_dir: str,
    label: int,
    grid_size: int,
    scale_factor: float,
    safetensors_fpath: str,
    direction: str,
    device: str,
    batch_size: int,
    display_ims: bool,
) -> None:
    img = load_first_cifar_image(cifar_root_dir=cifar_root_dir, label=label)
    model = ResNet18(num_classes=10, safetensors_path=safetensors_fpath)
    """ evaluate image over grid of perturbations in two directions.
    Saves (predictions,x_direction,y_direction,orig_img) """

    model = model.eval()
    model.to(device)

    #  directions
    if direction == "random":
        im_size = img.height * img.width * len(img.getbands())
        x_direction = get_random_1d_direction(size=im_size)
    elif direction == "gradient":
        x_direction = get_gradient_based_direction(
            model=model,
            data_sample=CIFAR_TRANSFORM(img),
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
        f"predictionsAndDirs_mirror_label"
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
    )
    fig.savefig(figure_fname)
    print(f"saved figure at {figure_fname}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Pixel Plot",
        description="Plots CIFAR10 class predictions over peturbations"
        " on given image",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed",
        required=False,
    )
    parser.add_argument(
        "--cifar_root_dir",
        type=str,
        help="root dir to find/store cifar10 dataset",
        required=True,
    )
    parser.add_argument(
        "--cifar_label",
        type=int,
        help="cifar class label to pick image from",
        required=True,
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="size of grid (square)",
        required=True,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        help="scale factor for peturbations",
        required=True,
    )
    parser.add_argument(
        "--resnet_safetensors_fpath",
        type=str,
        help="path to .safetensors for "
        "lightning_resnet.resnet18.ResNet18 model",
        required=True,
    )
    parser.add_argument(
        "--direction",
        type=str,
        help="method to pick xdirection: random or gradient",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run model on",
        required=False,
        default="auto",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--display_ims",
        action=argparse.BooleanOptionalAction,
        help="visualise images in plot",
    )

    args = parser.parse_args()
    display_ims = True if args.display_ims else False  # rather than None

    seed_everything(args.random_seed)

    if args.device == "auto":
        device = _AcceleratorConnector._choose_auto_accelerator()
    else:
        device = args.device

    main(
        cifar_root_dir=args.cifar_root_dir,
        label=args.cifar_label,
        grid_size=args.grid_size,
        scale_factor=args.scale_factor,
        safetensors_fpath=args.resnet_safetensors_fpath,
        direction=args.direction,
        device=device,
        batch_size=args.batch_size,
        display_ims=display_ims,
    )
