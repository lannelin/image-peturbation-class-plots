from collections import OrderedDict

import colorcet as cc
import seaborn as sns
import torch
from beartype import beartype
from jaxtyping import (
    Int,
    jaxtyped,
)
from matplotlib import pyplot as plt


@jaxtyped(typechecker=beartype)
def plot_predictions(
    predictions: Int[torch.Tensor, " dim1 dim1"],
    class_labels: list[str],
    true_image_label: int,
) -> None:

    grid_size = predictions.shape[0]

    # will modify predictions so clone
    predictions = predictions.clone()

    class_map = OrderedDict(enumerate(class_labels))
    used_class_map = {k: v for k, v in class_map.items() if k in predictions}

    prediction_value_mapping = {
        val: i for i, val in enumerate(used_class_map.keys())
    }

    # inplace apply
    predictions.apply_(lambda x: prediction_value_mapping[x])

    n = len(used_class_map)

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
    colorbar.set_ticklabels(list(used_class_map.values()))
    ax.set_title(f"peturbations of {class_labels[true_image_label]} example")
    plt.show()
