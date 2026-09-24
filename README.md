# Brain MRI Deep Learning with U-Net and VAE

A deep learning project exploring brain MRI image analysis using
U-Net and Variational Autoencoder (VAE).

The project focuses on image segmentation and latent representation
learning using the OASIS brain MRI dataset.

## Project Overview

This project implements two deep learning architectures:

- **U-Net** for brain MRI image segmentation
- **Variational Autoencoder (VAE)** for learning latent representations
  of brain MRI images

The models were implemented using PyTorch and evaluated through
training loss, Dice score, reconstruction quality, and latent-space
visualisation.

## Dataset

The project uses preprocessed brain MRI images from the
**OASIS (Open Access Series of Imaging Studies)** dataset.

## U-Net

U-Net was used to perform image segmentation on brain MRI images.

Model performance was evaluated using:

- Training loss
- Dice score
- Segmentation visualisation

### Segmentation Results

![U-Net Segmentation](p4_unet_segmentation.png)

### Training Loss

![U-Net Loss](p4_unet_loss.png)

### Dice Score

![U-Net Dice](p4_unet_dice.png)

## Variational Autoencoder

A Variational Autoencoder (VAE) was implemented to learn a compact
latent representation of brain MRI images.

The model was evaluated using:

- Reconstruction loss
- Image reconstruction quality
- Latent-space distribution
- Latent manifold visualisation

### Reconstruction

![VAE Reconstruction](p4_vae_reconstruction.png)

### Training Loss

![VAE Loss](p4_vae_loss.png)

### Latent Space

![VAE Latent Space](p4_vae_latent_scatter.png)

### Latent Manifold

![VAE Manifold](p4_vae_manifold.png)

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Deep Learning
- Computer Vision

## Skills Demonstrated

- Deep learning model implementation with PyTorch
- Medical image analysis
- Image segmentation using U-Net
- Representation learning using Variational Autoencoders
- Model evaluation using Dice score and training loss
- Visualisation of learned latent representations

## Project Context

This project was developed as part of COMP3710 at
The University of Queensland.
