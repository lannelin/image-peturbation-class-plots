# image-peturbation-class-plots

Plots CIFAR10 class predictions over peturbations of given image. X and Y axes describe two random directions in pixel space.

e.g.

![image](https://github.com/lannelin/image-peturbation-class-plots/assets/26149456/205e1d2a-9a06-46c4-a996-8f9d2b2a924d)


(Inspired by diagram in slide 11 of Nicholas Carlini's talk here: https://nicholas.carlini.com/slides/2023_adversarial_alignment.pdf)


This repo currently relies on having a ckpt file for a resnet18 model trained on CIFAR10 as per https://github.com/lannelin/cifar10-resnet-lightning



TODO:

- supply own image
- display image, peturbed images
- documentation
- batching

install project then:

```bash
python pixel_plot.py
```

## dev

extras:
```bash
pip install pre-commit
pre-commit install
```
