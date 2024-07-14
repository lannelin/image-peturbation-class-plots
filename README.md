# image-peturbation-class-plots

Plots class predictions over peturbations of given image. X and Y axes describe two directions in pixel space.

e.g. using model trained on CIFAR10

random directions:
![horse_deer_random](https://github.com/user-attachments/assets/d22004ac-4852-4f32-b5c4-951e7ac2798c)


gradient-based x direction:
![horse_deer_gradient](https://github.com/user-attachments/assets/307e4e93-3471-4b05-9ccd-70b7b78ce8bb)



generated with the following command (requires extra dep `pip install git+https://github.com/lannelin/cifar10-resnet-lightning`):
```bash
python pixel_plot.py --config ./configs/cifar10.yaml \
    --image_fpath ./demo_images/horse.jpeg \
    --true_label 7 \
    --grid_size 20 \
    --scale_factor 1.0 \
    --display_ims true \
    --batch_size 32 \
    --direction [random | gradient] \
    --device cpu \
    --model.class_path lightning_resnet.resnet18.ResNet18 \
    --model.safetensors_path PATH/TO/SAFETENSORS \
    --model.num_classes 10
```

(Inspired by diagram in slide 11 of Nicholas Carlini's talk here: https://nicholas.carlini.com/slides/2023_adversarial_alignment.pdf)


This example relies on having weights for a resnet18 model trained on CIFAR10 as per https://github.com/lannelin/cifar10-resnet-lightning. The example image was generated using weights that can be found in the [0.3.0 release](https://github.com/lannelin/cifar10-resnet-lightning/releases/tag/v0.3.0) of that repo.

The example is also tied to the CIFAR10 dataset and uses specific labels and transforms listed in `configs/cifar10.yaml`.



TODO:

- documentation
- further exploration of adversarial directions

install project

```bash
pip install -e .
```

then:

```bash
python pixel_plot.py --help
```

(see example above for more info)

## dev

extras:
```bash
pip install pre-commit
pre-commit install
```
