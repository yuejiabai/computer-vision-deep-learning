# Computer Vision and Deep Learning — Portfolio Components

This package contains cleaned portfolio versions of three COMP3710 experiments.

## Eigenfaces and PCA

`eigenfaces_pca.ipynb`

Face recognition using the LFW dataset, PCA/SVD eigenfaces, dimensionality
reduction, and Random Forest classification.

Verified test accuracy: **60.56%**

## CNN Face Classification

`face_cnn.ipynb`

A PyTorch convolutional neural network trained on the same LFW subset.

Verified test accuracy: **91.61%**

The CNN experiment provides a direct comparison with the PCA + Random Forest
baseline and demonstrates the benefit of learned spatial features.

## ResNet-18 on CIFAR-10

`resnet18_cifar10.py`

A ResNet-18 implementation adapted for CIFAR-10 classification, including
residual blocks, data augmentation, mixed-precision training, SGD with momentum,
and OneCycleLR scheduling.

## Result Images

- `results/part2/` — mean face, eigenfaces, PCA compactness, example predictions
- `results/part3/` — CNN loss, accuracy, and example predictions

## Project Context

These experiments were developed as part of **COMP3710** at
**The University of Queensland**.
