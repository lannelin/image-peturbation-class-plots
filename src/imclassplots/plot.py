import math
from collections import OrderedDict

import colorcet as cc
import numpy as np
import PIL
import seaborn as sns
import torch
from beartype import beartype
from beartype.typing import (
    Callable,
    Optional,
)
from jaxtyping import (
    Int,
    jaxtyped,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


@jaxtyped(typechecker=beartype)
def plot_predictions(
    predictions: Int[torch.Tensor, " dim1 dim1"],
    class_labels: list[str],
    true_image_label: int,
    display_ims: bool = False,
    im_generation_fn: Optional[
        Callable[[float, float], PIL.Image.Image]
    ] = None,
    scale_factor: Optional[float] = None,
) -> Figure:

    grid_size = predictions.shape[0]

    # will modify predictions so clone
    predictions = predictions.clone()

    # restrict class labels to first 8 chars if longer
    char_limit = 8
    class_labels = [
        f"{label[:char_limit]}.." if len(label) > char_limit else label
        for label in class_labels
    ]
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

    if display_ims:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))

    sns.heatmap(
        predictions.flip(0).numpy(),
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels[::-1],
        ax=ax0,
        square=True,
        linecolor="black",
        linewidths=0.1,
    )

    colorbar = ax0.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    # change offset
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(used_class_map.values()))
    ax0.set_title(f"peturbations of {class_labels[true_image_label]} example")

    # sns heatmap doesn't provide response x,y by default
    def format_coord(x, y):
        return f"x={math.floor(x + start)}, y={math.floor(-start - y)}"

    ax0.format_coord = format_coord

    if display_ims:
        if im_generation_fn is None or scale_factor is None:
            raise ValueError(
                "if display_ims is True, "
                " then im_generation_fn and scale_factor must be provided"
            )

        newval2classname = {
            v: used_class_map[k] for k, v in prediction_value_mapping.items()
        }
        orig_im_arr = np.asarray(im_generation_fn(scale_a=0.0, scale_b=0.0))
        ax1.imshow(orig_im_arr)

        def get_ax1_title(x, y):
            prediction = newval2classname[
                predictions[y - start, x - start].item()
            ]
            return (
                "click heatmap to visualise image at grid point\n"
                f"current: ({x},{y}) prediction={prediction}"
            )

        ax1.set_title(get_ax1_title(x=0, y=0))

        ax2.imshow(np.zeros_like(orig_im_arr))
        ax2.set_title("diff (magnitude) to orig")

        # add a handler to display image at grid point on click
        def onclick(event):
            if ax0.contains(event)[0]:
                x_loc = math.floor(event.xdata + start)
                y_loc = math.floor(-start - event.ydata)
                scale_x = scale_factor * x_loc
                scale_y = scale_factor * y_loc
                im = np.asarray(
                    im_generation_fn(scale_a=scale_x, scale_b=scale_y)
                )
                ax1.imshow(im)
                # magnitude of update to stop wrap around
                ax2.imshow(
                    np.abs(im.astype(np.int32) - orig_im_arr).astype(np.uint8)
                )

                ax1.set_title(get_ax1_title(x_loc, y_loc))

            fig.canvas.draw_idle()

        # add callback for mouse moves
        fig.canvas.mpl_connect("button_press_event", onclick)

    plt.tight_layout()
    return fig
