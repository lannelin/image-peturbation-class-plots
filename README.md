# image-peturbation-class-plots

Plots class predictions over peturbations of given image. X and Y axes describe two directions in pixel space.


Inspired by diagram in slide 11 of Nicholas Carlini's talk here: https://nicholas.carlini.com/slides/2023_adversarial_alignment.pdf


## ImageNet1k example

Using a stock torchvision model trained on ImageNet1k (image resized to 224x224).

random directions:

![sorrel_redwolf_random_imagenet](https://github.com/user-attachments/assets/0bc7b116-c420-477b-8c39-e577cef7891c)



```bash
python pixel_plot.py --config ./configs/imagenet1k.yaml \
    --image_fpath ./demo_images/horse.jpeg \
    --true_label 339 \
    --grid_size 20 \
    --scale_factor 10.0 \
    --display_ims true \
    --batch_size 32 \
    --direction random \
    --device mps \
    --model_fn torchvision.models.inception.inception_v3 \
    --model_fn_kwargs.pretrained true
```

The example is tied to the ImageNet1k dataset and uses specific labels and transforms listed in `configs/imagenet1k.yaml`.

## CIFAR10 example

Using a custom model trained on CIFAR10 (image resized to 32x32):

requires extra dep `pip install git+https://github.com/lannelin/cifar10-resnet-lightning`

random directions:
![horse_deer_random](/demo_images/horse_deer_random.png)



gradient-based x direction:
![horse_deer_gradient](/demo_images/horse_deer_gradient.png)


hessian eigenvector-based x and y directions (NOTE: currently untested impl)
![horse_deer_gradient](/demo_images/horse_deer_hess-eig.png)



random example generated on macbook with:
```bash
python pixel_plot.py --config ./configs/cifar10.yaml \
    --image_fpath ./demo_images/horse.jpeg \
    --true_label 7 \
    --grid_size 100 \
    --scale_factor 0.15 \
    --display_ims true \
    --batch_size 32 \
    --direction random \
    --device mps \
    --model.class_path lightning_resnet.resnet18.ResNet18 \
    --model.safetensors_path PATH/TO/SAFETENSORS \
    --model.num_classes 10
```


This example relies on having weights for a resnet18 model trained on CIFAR10 as per https://github.com/lannelin/cifar10-resnet-lightning. The example image was generated using weights that can be found in the [0.3.0 release](https://github.com/lannelin/cifar10-resnet-lightning/releases/tag/v0.3.0) of that repo. These can be downloaded with
```bash
wget https://github.com/lannelin/cifar10-resnet-lightning/releases/download/v0.3.0/resnet18-cifar10-86.4val.safetensors
```

The example is also tied to the CIFAR10 dataset and uses specific labels and transforms listed in `configs/cifar10.yaml`.

Note the much lower scale_factor for this example. This gives us smaller peturbations.


## TODOs

TODO:

- documentation
- further exploration of adversarial directions

## Install

install project

```bash
pip install -e .
```

then:

```bash
python pixel_plot.py --help
```

(see examples above for more info)

## dev

extras:
```bash
pip install pre-commit
pre-commit install
```
