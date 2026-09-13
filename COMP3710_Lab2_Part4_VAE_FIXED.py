import argparse
import glob
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SEED = 42


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OASISPNGDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not self.files:
            raise RuntimeError(f"No PNG files found in {folder}")

        # Keep grayscale MRI slices and scale pixels into [0, 1].
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        x = self.transform(img)
        return x


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim

        # 1x128x128 -> 32x64x64 -> 64x32x32
        # -> 128x16x16 -> 256x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.feature_shape = (256, 8, 8)
        self.flat_dim = 256 * 8 * 8

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # 256x8x8 -> 128x16x16 -> 64x32x32
        # -> 32x64x64 -> 1x128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Keep latent statistics numerically safe.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def decode(self, z):
        h = self.fc_decode(z).view(-1, *self.feature_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, logvar = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta):
    # Everything here is float32 for stability.
    recon = recon.float()
    x = x.float()
    mu = mu.float()
    logvar = logvar.float()

    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_n = 0

    for x in loader:
        x = x.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        # IMPORTANT: no FP16 autocast here.
        # The previous version became numerically unstable in the
        # latent mean/log-variance heads and produced NaNs.
        recon, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(
            recon, x, mu, logvar, beta
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite VAE loss detected. "
                f"loss={loss.item()}, recon={recon_loss.item()}, "
                f"KL={kl_loss.item()}"
            )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        n = x.size(0)
        total_loss += loss.detach().item() * n
        total_recon += recon_loss.detach().item() * n
        total_kl += kl_loss.detach().item() * n
        total_n += n

    return (
        total_loss / total_n,
        total_recon / total_n,
        total_kl / total_n,
    )


@torch.no_grad()
def save_reconstructions(model, loader, device, path):
    model.eval()
    x = next(iter(loader))[:8].to(device)
    recon, _, _ = model(x)
    x = x.cpu()
    recon = recon.cpu()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title("Original", fontsize=8)

        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        axes[1, i].set_title("Recon", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_latent_scatter(model, loader, device, path, max_points=2000):
    model.eval()
    zs = []
    count = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        mu, _ = model.encode(x)
        z = mu.cpu().numpy()
        zs.append(z)
        count += len(z)
        if count >= max_points:
            break

    z = np.concatenate(zs, axis=0)[:max_points]

    plt.figure(figsize=(7, 6))
    plt.scatter(z[:, 0], z[:, 1], s=7, alpha=0.5)
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("OASIS VAE Latent Space")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_manifold(model, device, path, grid_size=10, span=2.5):
    if model.latent_dim != 2:
        raise ValueError("2D manifold requires latent_dim=2")

    model.eval()
    values = np.linspace(-span, span, grid_size)
    tile = 128
    canvas = np.zeros(
        (grid_size * tile, grid_size * tile),
        dtype=np.float32
    )

    for r, y in enumerate(values[::-1]):
        for c, x in enumerate(values):
            z = torch.tensor([[x, y]], dtype=torch.float32, device=device)
            img = model.decode(z)[0, 0].cpu().numpy()
            canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = img

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("2D VAE Manifold - Decoded Latent Grid")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--data-root",
        default="/home/groups/comp3710/OASIS"
    )
    args = parser.parse_args()

    seed_all()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_set = OASISPNGDataset(
        os.path.join(args.data_root, "keras_png_slices_train")
    )
    val_set = OASISPNGDataset(
        os.path.join(args.data_root, "keras_png_slices_validate")
    )
    test_set = OASISPNGDataset(
        os.path.join(args.data_root, "keras_png_slices_test")
    )

    print("Train images:", len(train_set))
    print("Validation images:", len(val_set))
    print("Test images:", len(test_set))
    print("Model input size: 1 x 128 x 128")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    train_loader = DataLoader(
        train_set, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_set, shuffle=False, **loader_kwargs
    )

    model = ConvVAE(args.latent_dim).to(device)
    print(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    train_history = []
    val_history = []

    best_val = float("inf")
    best_epoch = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, device, args.beta, optimizer
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, val_loader, device, args.beta
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"val_recon={val_recon:.6f} | "
            f"val_KL={val_kl:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "beta": args.beta,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                "p4_vae_best.pt"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    checkpoint = torch.load(
        "p4_vae_best.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint["model"])

    test_loss, test_recon, test_kl = run_epoch(
        model, test_loader, device, args.beta
    )

    print()
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val)
    print("Test total loss:", test_loss)
    print("Test reconstruction loss:", test_recon)
    print("Test KL loss:", test_kl)
    print("Training time (s):", elapsed)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_history, label="Train")
    plt.plot(range(1, args.epochs + 1), val_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("VAE loss")
    plt.title("OASIS VAE Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("p4_vae_loss.png", dpi=160)
    plt.close()

    save_reconstructions(
        model, test_loader, device,
        "p4_vae_reconstruction.png"
    )
    save_latent_scatter(
        model, test_loader, device,
        "p4_vae_latent_scatter.png"
    )
    save_manifold(
        model, device,
        "p4_vae_manifold.png"
    )

    with open("p4_vae_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Train images: {len(train_set)}\n")
        f.write(f"Validation images: {len(val_set)}\n")
        f.write(f"Test images: {len(test_set)}\n")
        f.write("Input size: 1x128x128 (resized from 256x256)\n")
        f.write(f"Latent dimension: {args.latent_dim}\n")
        f.write(f"Beta: {args.beta}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val:.8f}\n")
        f.write(f"Test total loss: {test_loss:.8f}\n")
        f.write(f"Test reconstruction loss: {test_recon:.8f}\n")
        f.write(f"Test KL loss: {test_kl:.8f}\n")
        f.write(f"Training time (s): {elapsed:.2f}\n")

    print()
    print("Saved:")
    print("  p4_vae_best.pt")
    print("  p4_vae_results.txt")
    print("  p4_vae_loss.png")
    print("  p4_vae_reconstruction.png")
    print("  p4_vae_latent_scatter.png")
    print("  p4_vae_manifold.png")


if __name__ == "__main__":
    main()
