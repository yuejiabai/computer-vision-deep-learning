# Brain MRI Deep Learning with U-Net and VAE

A deep learning project exploring brain MRI image analysis using
U-Net and Variational Autoencoder (VAE) on the OASIS dataset.

The project focuses on image segmentation, image reconstruction,
and latent representation learning.

## Project Task

This project investigates two deep learning approaches for brain MRI analysis:

- **U-Net** for brain MRI image segmentation
- **Variational Autoencoder (VAE)** for image reconstruction and latent representation learning

The models were trained and evaluated using PyTorch.

## Dataset

The project uses preprocessed brain MRI images from the
**OASIS (Open Access Series of Imaging Studies)** dataset.

## My Contribution

My work included:

- Training and evaluating the U-Net segmentation model
- Training and analysing the Variational Autoencoder
- Configuring experiments and model parameters
- Evaluating segmentation performance using Dice score and training loss
- Analysing VAE reconstruction quality and latent representations
- Producing visualisations of model performance and experimental results

## U-Net Image Segmentation

U-Net was used to perform segmentation on brain MRI images.

The model was evaluated using:

- Training loss
- Dice score
- Segmentation visualisation

### Segmentation Results

![U-Net Segmentation](results/p4_unet_segmentation.png)

### Training Loss

![U-Net Loss](results/p4_unet_loss.png)

### Dice Score

![U-Net Dice](results/p4_unet_dice.png)

## Variational Autoencoder

A Variational Autoencoder (VAE) was used to learn compact latent
representations of brain MRI images and reconstruct input images.

The model was evaluated using:

- Reconstruction loss
- Image reconstruction quality
- Latent-space distribution
- Latent manifold visualisation

### Reconstruction

![VAE Reconstruction](results/p4_vae_reconstruction.png)

### Training Loss

![VAE Loss](results/p4_vae_loss.png)

### Latent Space

![VAE Latent Space](results/p4_vae_latent_scatter.png)

### Latent Manifold

![VAE Manifold](results/p4_vae_manifold.png)

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Deep Learning
- Computer Vision

## Key Skills

- Deep learning model training and evaluation with PyTorch
- Medical image analysis
- Image segmentation using U-Net
- Representation learning using Variational Autoencoders
- Model evaluation using Dice score and training loss
- Visualisation and interpretation of deep learning results

## Project Structure

```text
deep-learning-unet-vae/
├── README.md
├── unet.py
├── vae.py
└── results/
    ├── p4_unet_dice.png
    ├── p4_unet_loss.png
    ├── p4_unet_segmentation.png
    ├── p4_vae_latent_scatter.png
    ├── p4_vae_loss.png
    ├── p4_vae_manifold.png
    └── p4_vae_reconstruction.png
