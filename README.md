# COMP3710 Lab Demonstration 2 – Part 4: OASIS VAE

## Overview

This project implements the **Easy task in Part 4** of COMP3710 Lab Demonstration 2.

The aim is to train a **Variational Autoencoder (VAE)** on preprocessed OASIS brain MRI slices and visualise the learned latent manifold.

## Dataset

The dataset is the preprocessed OASIS dataset provided on the UQ Rangpur cluster:

```text
/home/groups/comp3710/OASIS/
```

The VAE uses:

```text
keras_png_slices_train
keras_png_slices_validate
keras_png_slices_test
```

The original images are grayscale `256 × 256` PNG slices.

For this implementation, the images are resized to `128 × 128` before being passed to the network.

Dataset sizes from the successful run:

```text
Training images:   9664
Validation images: 1120
Test images:       544
```

## Model

The model is a convolutional Variational Autoencoder.

### Encoder

The encoder contains four stride-2 convolution layers:

```text
1 × 128 × 128
→ 32 × 64 × 64
→ 64 × 32 × 32
→ 128 × 16 × 16
→ 256 × 8 × 8
```

The encoded feature map is flattened and mapped to:

```text
mu
logvar
```

The latent dimension is:

```text
2
```

A 2D latent space was chosen so that the learned manifold can be visualised directly.

### Reparameterisation

The latent vector is sampled using:

```text
z = mu + sigma × epsilon
```

where:

```text
sigma = exp(0.5 × logvar)
epsilon ~ N(0, I)
```

This allows stochastic sampling while still allowing gradients to propagate through the model.

### Decoder

The decoder maps the latent vector back to the image space using transposed convolutions.

The final layer uses a sigmoid activation so that reconstructed pixel values remain between `0` and `1`.

## Loss Function

The VAE objective contains two parts:

```text
Total Loss = Reconstruction Loss + beta × KL Divergence
```

This implementation uses:

```text
Reconstruction loss: Mean Squared Error
beta:                0.0001
```

The reconstruction term encourages the decoded MRI image to match the original image.

The KL-divergence term regularises the latent distribution toward a standard normal distribution.

## Training

The model was trained on an **NVIDIA A100-PCIE-40GB GPU** on the UQ Rangpur cluster.

Successful run settings:

```text
Epochs:         30
Batch size:     64
Learning rate:  0.0003
Latent dim:     2
beta:           0.0001
```

Training used full-precision GPU computation for numerical stability.

## Results

Results from the successful Rangpur run:

```text
Best epoch:                 25
Best validation loss:       0.00528561
Test total loss:            0.00520910
Test reconstruction loss:   0.00498404
Test KL loss:               2.25064846
Training time:              183.02 seconds
```

The training and validation losses decreased quickly and then stabilised, indicating that the model converged.

The reconstructions preserve the overall brain structure, although they are smoother than the original MRI images. This is expected because the images are compressed into only two latent dimensions and the reconstruction objective uses mean squared error.

The decoded 2D latent grid shows gradual changes between neighbouring generated brain images, demonstrating a continuous learned manifold.

## Output Files

The training script produces:

```text
p4_vae_best.pt
p4_vae_results.txt
p4_vae_loss.png
p4_vae_reconstruction.png
p4_vae_latent_scatter.png
p4_vae_manifold.png
```

### Important visualisations

`p4_vae_reconstruction.png`

Shows original test MRI slices and their VAE reconstructions.

`p4_vae_latent_scatter.png`

Shows the distribution of test images in the 2D latent space.

`p4_vae_manifold.png`

Samples a grid of coordinates across the 2D latent space and decodes them into MRI images to visualise the learned manifold.

`p4_vae_loss.png`

Shows the training and validation loss across epochs.

## Running on Rangpur

Submit the GPU job with:

```bash
sbatch COMP3710_Lab2_Part4_VAE_FIXED_job.sh
```

Check the queue with:

```bash
squeue --me
```

After the job finishes, view the results with:

```bash
cat p4_vae_results.txt
```

The main implementation is:

```text
COMP3710_Lab2_Part4_VAE_FIXED.py
```

## What I Learned

This task demonstrates the difference between a normal autoencoder and a VAE.

A normal autoencoder maps an image to a single deterministic latent representation.

A VAE instead learns a probability distribution in latent space using `mu` and `logvar`. The KL-divergence term makes the latent space smoother and more structured, which makes it possible to sample new points and generate meaningful outputs.

The 2D latent manifold provides a direct visualisation of this idea.

## AI Assistance

AI tools were used to assist with code review, debugging, and explanation.

During development, an earlier version of the VAE produced `NaN` losses because the latent statistics became numerically unstable. The issue was identified from the training output and the implementation was revised by:

- removing mixed-precision training for the VAE,
- reducing the input size to `128 × 128`,
- lowering the learning rate,
- clamping `logvar`,
- adding gradient clipping,
- and explicitly checking for non-finite loss values.

The final model was then rerun and validated on Rangpur using the generated loss curves, reconstruction images, latent-space plots, manifold visualisation, and saved test results.

## Files in This Repository

Recommended repository contents:

```text
COMP3710_Lab2_Part4_VAE_FIXED.py
COMP3710_Lab2_Part4_VAE_FIXED_job.sh
README.md
p4_vae_loss.png
p4_vae_reconstruction.png
p4_vae_latent_scatter.png
p4_vae_manifold.png
```
