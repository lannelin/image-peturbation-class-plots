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

import PIL
import torch
from beartype import beartype
from lightning import seed_everything
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from lightning_resnet.resnet18 import ResNet18
from torchvision import transforms
from torchvision.datasets import CIFAR10

from imclassplots.peturb import peturb_and_predict
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
) -> None:
    img = load_first_cifar_image(cifar_root_dir=cifar_root_dir, label=label)

    device = _AcceleratorConnector._choose_auto_accelerator()
    model = ResNet18(num_classes=10, safetensors_path=safetensors_fpath)
    """ evaluate image over grid of perturbations in two random directions.
    Saves (predictions,x_direction,y_direction) """

    model = model.eval()
    model.to(device)
    predictions, x_direction, y_direction = peturb_and_predict(
        image=img,
        model=model,
        label=label,
        grid_size=grid_size,
        data_transform=CIFAR_TRANSFORM,
        device=device,
        scale_factor=scale_factor,
    )

    plot_directory = "./plots"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    fname = os.path.join(
        plot_directory,
        f"predictionsAndDirs_mirror_label"
        f"{label}_gridsize{grid_size}_sf{scale_factor}.pt",
    )
    torch.save(
        (predictions, x_direction, y_direction),
        fname,
    )
    print(f"saved (predictions,x_direction,y_direction) tuple at {fname}")

    plot_predictions(
        predictions=predictions,
        class_labels=CIFAR_CLASSES,
        true_image_label=label,
    )


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

    args = parser.parse_args()

    seed_everything(args.random_seed)

    main(
        cifar_root_dir=args.cifar_root_dir,
        label=args.cifar_label,
        grid_size=args.grid_size,
        scale_factor=args.scale_factor,
        safetensors_fpath=args.resnet_safetensors_fpath,
    )
