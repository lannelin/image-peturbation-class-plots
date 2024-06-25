# image-peturbation-class-plots

Plots CIFAR10 class predictions over peturbations of given image. X and Y axes describe two random directions in pixel space.

e.g.

![image](https://github.com/lannelin/image-peturbation-class-plots/assets/26149456/205e1d2a-9a06-46c4-a996-8f9d2b2a924d)


(Inspired by diagram in slide 11 of Nicholas Carlini's talk here: https://nicholas.carlini.com/slides/2023_adversarial_alignment.pdf)


This repo currently relies on having weights for a resnet18 model trained on CIFAR10 as per https://github.com/lannelin/cifar10-resnet-lightning. The example image was generated using weights that can be found in the [0.3.0 release](https://github.com/lannelin/cifar10-resnet-lightning/releases/tag/v0.3.0) of that repo.



TODO:

- supply own image
- display image, peturbed images
- documentation
- batching

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
