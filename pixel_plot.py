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
from collections import OrderedDict

import colorcet as cc
import PIL
import seaborn as sns
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from lightning import seed_everything
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)
from lightning_resnet.resnet18 import ResNet18
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

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
def permute(
    img: PIL.Image.Image,
    direction_a: Float[torch.Tensor, " dim1"],  # noqa:
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
def plot_predictions(
    predictions: Int[torch.Tensor, " dim1 dim1"],
    classes: list[str],
    label: int,
    grid_size: int,
) -> None:
    print(predictions)

    # will modify predictions so clone
    predictions = predictions.clone()

    cifar_class_map = OrderedDict(enumerate(classes))
    used_cifar_class_map = {
        k: v for k, v in cifar_class_map.items() if k in predictions
    }

    prediction_value_mapping = {
        val: i for i, val in enumerate(used_cifar_class_map.keys())
    }

    # inplace apply
    predictions.apply_(lambda x: prediction_value_mapping[x])

    n = len(used_cifar_class_map)

    cmap = sns.color_palette(cc.glasbey, n)
    start = -grid_size // 2
    labels = list(range(start, start + grid_size))
    ax = sns.heatmap(
        predictions.numpy(), cmap=cmap, xticklabels=labels, yticklabels=labels
    )
    print(cmap)

    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    # change offset
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(used_cifar_class_map.values()))
    ax.set_title(f"peturbations of {classes[label]} example")
    plt.show()


@jaxtyped(typechecker=beartype)
def peturb_and_predict(
    image: PIL.Image.Image,
    model: torch.nn.Module,
    grid_size: int,
    label: int,
    device: str,
    scale_factor: float = 1.0,
) -> tuple[
    Int[torch.Tensor, "{grid_size} {grid_size}"],
    Float[torch.Tensor, " dim1"],
    Float[torch.Tensor, " dim1"],
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
        x = CIFAR_TRANSFORM(image)
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

            permuted = permute(
                img=image,
                direction_a=x_direction,
                direction_b=y_direction,
                scale_a=x_coord * scale_factor,
                scale_b=y_coord * scale_factor,
            )

            x = CIFAR_TRANSFORM(permuted)

            with torch.inference_mode():
                scores = model(x.unsqueeze(0).to(device))
            prediction = torch.argmax(scores).detach().cpu().item()
            predictions[i, j] = prediction

    return predictions, x_direction, y_direction


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
        classes=CIFAR_CLASSES,
        label=label,
        grid_size=grid_size,
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
