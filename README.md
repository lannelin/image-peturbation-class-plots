# image-peturbation-class-plots

Plots CIFAR10 class predictions over peturbations of given image. X and Y axes describe two directions in pixel space.

e.g.

random directions:
![horse_deer_random](https://github.com/lannelin/image-peturbation-class-plots/assets/26149456/90a4090a-b7a0-4569-b58d-97d91d6326a3)


gradient-based x direction:
![horse_deer_gradient](https://github.com/lannelin/image-peturbation-class-plots/assets/26149456/0e63aa8e-a2dc-4d78-b79a-29d13c0ec838)




generated with:
```bash
python pixel_plot.py \
    --image_fpath ./demo_images/horse.jpeg \
    --cifar_label 7 \
    --grid_size 20 \
    --scale_factor 1.0 \
    --resnet_safetensors_fpath [/path/to/model.safetensors] \
    --display_ims \
    --device auto \
    --batch_size 32 \
    --direction [random|gradient]
```

(Inspired by diagram in slide 11 of Nicholas Carlini's talk here: https://nicholas.carlini.com/slides/2023_adversarial_alignment.pdf)


This repo currently relies on having weights for a resnet18 model trained on CIFAR10 as per https://github.com/lannelin/cifar10-resnet-lightning. The example image was generated using weights that can be found in the [0.3.0 release](https://github.com/lannelin/cifar10-resnet-lightning/releases/tag/v0.3.0) of that repo.



TODO:

- documentation
- further exploration of adversarial directions
- drop lightning dep?

install project

```bash
pip install -e .
```

then:

```bash
python pixel_plot.py --help
```

## dev

extras:
```bash
pip install pre-commit
pre-commit install
```
