
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


SEED = 42


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pngs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")))


def pair_image_mask_files(image_dir, mask_dir):
    image_files_all = list_pngs(image_dir)
    mask_files_all = list_pngs(mask_dir)

    # Example:
    # image: case_001_slice_10.nii.png
    # mask:  seg_001_slice_10.nii.png
    #
    # Use the part after "case_" / "seg_" as the matching key.

    images = {}
    for path in image_files_all:
        name = os.path.basename(path)

        if name.startswith("case_"):
            key = name[len("case_"):]
        else:
            key = name

        images[key] = path

    masks = {}
    for path in mask_files_all:
        name = os.path.basename(path)

        if name.startswith("seg_"):
            key = name[len("seg_"):]
        else:
            key = name

        masks[key] = path

    common = sorted(set(images.keys()) & set(masks.keys()))

    if not common:
        raise RuntimeError(
            "No matching image/mask pairs.\n"
            f"Image dir: {image_dir}\n"
            f"Mask dir: {mask_dir}"
        )

    image_files = [images[key] for key in common]
    mask_files = [masks[key] for key in common]

    print(
        f"Paired {len(common)} image/mask files from "
        f"{os.path.basename(image_dir)} and "
        f"{os.path.basename(mask_dir)}"
    )

    if len(images) != len(common):
        print(
            f"Warning: {len(images) - len(common)} "
            "images had no matching mask."
        )

    if len(masks) != len(common):
        print(
            f"Warning: {len(masks) - len(common)} "
            "masks had no matching image."
        )

    return image_files, mask_files


def discover_mask_values(mask_files, max_files=400):
    """
    Discover discrete grayscale values used by segmentation masks.
    Uses evenly spaced samples from the mask set.
    """
    n = len(mask_files)
    if n <= max_files:
        chosen = mask_files
    else:
        indices = np.linspace(0, n - 1, max_files, dtype=int)
        chosen = [mask_files[i] for i in indices]

    values = set()
    for path in chosen:
        mask = np.asarray(Image.open(path).convert("L"))
        values.update(np.unique(mask).tolist())

    values = sorted(int(v) for v in values)

    if len(values) < 2:
        raise RuntimeError(f"Only found mask values: {values}")

    if len(values) > 20:
        raise RuntimeError(
            "Found too many distinct mask values for a categorical mask: "
            f"{values[:30]}. Please inspect the segmentation files."
        )

    print("Detected raw mask values:", values)
    print("Number of segmentation labels:", len(values))
    return values


def map_mask_to_classes(mask, raw_values):
    """
    Convert original grayscale label values (e.g. 0, 63, 127, ...)
    into contiguous class indices 0, 1, 2, ...
    """
    mask = mask.astype(np.int16)
    values = np.asarray(raw_values, dtype=np.int16)

    # Nearest-value mapping is robust if a mask contains an unexpected
    # grayscale value close to the known discrete labels.
    distance = np.abs(mask[..., None] - values[None, None, :])
    class_mask = np.argmin(distance, axis=-1).astype(np.int64)
    return class_mask


class OASISSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, raw_values, augment=False):
        self.image_files, self.mask_files = pair_image_mask_files(
            image_dir, mask_dir
        )
        self.raw_values = list(raw_values)
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.asarray(
            Image.open(self.image_files[idx]).convert("L"),
            dtype=np.float32,
        ) / 255.0

        mask_raw = np.asarray(
            Image.open(self.mask_files[idx]).convert("L")
        )

        mask = map_mask_to_classes(mask_raw, self.raw_values)

        # Synchronized augmentation for image and segmentation mask.
        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        x = torch.from_numpy(image).unsqueeze(0).float()
        y = torch.from_numpy(mask).long()

        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard encoder-decoder U-Net with skip connections.

    The final 1x1 convolution outputs one channel per class.
    Softmax over the channel dimension gives a categorical output.
    """

    def __init__(self, in_channels, num_classes, base=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.out_conv(d1)
        return logits


def estimate_class_weights(mask_files, raw_values, max_files=400):
    """
    Estimate inverse-sqrt-frequency weights from a representative subset.
    This helps rare segmentation labels without using extreme weights.
    """
    n = len(mask_files)
    if n <= max_files:
        chosen = mask_files
    else:
        indices = np.linspace(0, n - 1, max_files, dtype=int)
        chosen = [mask_files[i] for i in indices]

    counts = np.zeros(len(raw_values), dtype=np.float64)

    for path in chosen:
        raw = np.asarray(Image.open(path).convert("L"))
        y = map_mask_to_classes(raw, raw_values)
        counts += np.bincount(
            y.ravel(),
            minlength=len(raw_values),
        )

    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, 5.0)

    print("Estimated class frequencies:", freq)
    print("Categorical CE class weights:", weights)

    return torch.tensor(weights, dtype=torch.float32)


def one_hot_target(target, num_classes):
    return F.one_hot(
        target,
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).float()


def categorical_ce_from_one_hot(logits, target_onehot, class_weights):
    """
    Explicit categorical cross entropy using one-hot targets.
    """
    log_probs = F.log_softmax(logits, dim=1)

    w = class_weights.view(1, -1, 1, 1)
    loss_map = -(target_onehot * log_probs * w).sum(dim=1)
    return loss_map.mean()


def soft_dice_loss(logits, target_onehot, eps=1e-6):
    probs = F.softmax(logits, dim=1)

    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    denominator = probs.sum(dims) + target_onehot.sum(dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def combined_loss(
    logits,
    target,
    num_classes,
    class_weights,
    ce_weight=0.5,
    dice_weight=0.5,
):
    target_oh = one_hot_target(target, num_classes)

    ce = categorical_ce_from_one_hot(
        logits,
        target_oh,
        class_weights,
    )

    dice_loss = soft_dice_loss(
        logits,
        target_oh,
    )

    total = ce_weight * ce + dice_weight * dice_loss
    return total, ce, dice_loss


@torch.no_grad()
def update_dice_counts(
    logits,
    target,
    intersections,
    pred_counts,
    target_counts,
    num_classes,
):
    pred = logits.argmax(dim=1)

    pred_oh = one_hot_target(pred, num_classes)
    target_oh = one_hot_target(target, num_classes)

    intersections += (
        pred_oh * target_oh
    ).sum(dim=(0, 2, 3)).double().cpu()

    pred_counts += pred_oh.sum(
        dim=(0, 2, 3)
    ).double().cpu()

    target_counts += target_oh.sum(
        dim=(0, 2, 3)
    ).double().cpu()


def dice_from_counts(intersections, pred_counts, target_counts):
    eps = 1e-8
    return (
        2.0 * intersections + eps
    ) / (
        pred_counts + target_counts + eps
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    num_classes,
    class_weights,
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_n = 0

    use_amp = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp,
        ):
            logits = model(x)

            loss, ce, dice_loss = combined_loss(
                logits,
                y,
                num_classes,
                class_weights,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()

        n = x.size(0)

        total_loss += loss.detach().item() * n
        total_ce += ce.detach().item() * n
        total_dice_loss += dice_loss.detach().item() * n
        total_n += n

    return (
        total_loss / total_n,
        total_ce / total_n,
        total_dice_loss / total_n,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    class_weights,
):
    model.eval()

    total_loss = 0.0
    total_n = 0

    intersections = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )
    pred_counts = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )
    target_counts = torch.zeros(
        num_classes,
        dtype=torch.float64,
    )

    use_amp = device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp,
        ):
            logits = model(x)
            loss, _, _ = combined_loss(
                logits,
                y,
                num_classes,
                class_weights,
            )

        n = x.size(0)
        total_loss += loss.item() * n
        total_n += n

        update_dice_counts(
            logits,
            y,
            intersections,
            pred_counts,
            target_counts,
            num_classes,
        )

    dice = dice_from_counts(
        intersections,
        pred_counts,
        target_counts,
    )

    return total_loss / total_n, dice.numpy()


@torch.no_grad()
def save_segmentation_examples(
    model,
    loader,
    device,
    path,
    num_classes,
    max_examples=6,
):
    model.eval()

    x, y = next(iter(loader))
    x = x[:max_examples].to(device)
    y = y[:max_examples]

    logits = model(x)

    # Categorical probabilities and hard one-hot output.
    probs = F.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1)
    pred_onehot = F.one_hot(
        pred_idx,
        num_classes=num_classes,
    ).permute(0, 3, 1, 2)

    # Sanity check: each pixel belongs to exactly one category.
    check = pred_onehot.sum(dim=1).unique().cpu().tolist()
    print("One-hot prediction channel-sum values:", check)

    pred_idx = pred_idx.cpu()
    x = x.cpu()

    fig, axes = plt.subplots(
        max_examples,
        3,
        figsize=(9, 3 * max_examples),
    )

    if max_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(max_examples):
        axes[i, 0].imshow(
            x[i, 0],
            cmap="gray",
        )
        axes[i, 0].set_title("MRI")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(
            y[i],
            vmin=0,
            vmax=max(num_classes - 1, 1),
        )
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(
            pred_idx[i],
            vmin=0,
            vmax=max(num_classes - 1, 1),
        )
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_loaders(
    data_root,
    raw_values,
    batch_size,
    workers,
    device,
):
    train_image_dir = os.path.join(
        data_root,
        "keras_png_slices_train",
    )
    val_image_dir = os.path.join(
        data_root,
        "keras_png_slices_validate",
    )
    test_image_dir = os.path.join(
        data_root,
        "keras_png_slices_test",
    )

    train_mask_dir = os.path.join(
        data_root,
        "keras_png_slices_seg_train",
    )
    val_mask_dir = os.path.join(
        data_root,
        "keras_png_slices_seg_validate",
    )
    test_mask_dir = os.path.join(
        data_root,
        "keras_png_slices_seg_test",
    )

    train_set = OASISSegmentationDataset(
        train_image_dir,
        train_mask_dir,
        raw_values,
        augment=True,
    )

    val_set = OASISSegmentationDataset(
        val_image_dir,
        val_mask_dir,
        raw_values,
        augment=False,
    )

    test_set = OASISSegmentationDataset(
        test_image_dir,
        test_mask_dir,
        raw_values,
        augment=False,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **common,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common,
    )

    return (
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
    )


def print_dice(dice, raw_values, prefix):
    print(prefix)
    for i, score in enumerate(dice):
        print(
            f"  label {i} "
            f"(raw value {raw_values[i]}): "
            f"DSC={score:.4f}"
        )

    print(
        f"  minimum DSC across labels: "
        f"{float(np.min(dice)):.4f}"
    )


def train_mode(args):
    seed_all()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_mask_dir = os.path.join(
        args.data_root,
        "keras_png_slices_seg_train",
    )

    train_mask_files = list_pngs(train_mask_dir)

    raw_values = discover_mask_values(
        train_mask_files,
        max_files=args.label_scan,
    )

    num_classes = len(raw_values)

    (
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        args.data_root,
        raw_values,
        args.batch_size,
        args.workers,
        device,
    )

    class_weights = estimate_class_weights(
        train_set.mask_files,
        raw_values,
        max_files=args.weight_scan,
    ).to(device)

    model = UNet(
        in_channels=1,
        num_classes=num_classes,
        base=args.base,
    ).to(device)

    print(model)
    print("Categorical output channels:", num_classes)
    print("Raw mask values -> class indices:")
    for i, value in enumerate(raw_values):
        print(f"  {value} -> class {i}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    train_losses = []
    val_losses = []
    min_val_dices = []

    best_min_dice = -1.0
    best_epoch = 0
    target_streak = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ce, train_dice_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            num_classes,
            class_weights,
        )

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            class_weights,
        )

        min_dice = float(np.min(val_dice))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        min_val_dices.append(min_dice)

        lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"CE={train_ce:.5f} | "
            f"soft_dice_loss={train_dice_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"min_val_DSC={min_dice:.4f} | "
            f"lr={lr:.6f}"
        )

        print_dice(
            val_dice,
            raw_values,
            "Validation DSC by label:",
        )

        scheduler.step(min_dice)

        if min_dice > best_min_dice:
            best_min_dice = min_dice
            best_epoch = epoch

            torch.save(
                {
                    "model": model.state_dict(),
                    "raw_values": raw_values,
                    "num_classes": num_classes,
                    "base": args.base,
                    "epoch": epoch,
                    "best_min_val_dice": best_min_dice,
                },
                args.checkpoint,
            )

            print(
                f"Saved new best checkpoint: "
                f"{args.checkpoint}"
            )

        if min_dice >= args.target_dice:
            target_streak += 1
            print(
                f"All-label target reached for "
                f"{target_streak} consecutive epoch(s)."
            )
        else:
            target_streak = 0

        if target_streak >= args.target_patience:
            print(
                "Stopping early because all labels have "
                f"DSC >= {args.target_dice:.3f}."
            )
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    test_loss, test_dice = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        class_weights,
    )

    print()
    print("===== FINAL TEST RESULT =====")
    print("Best epoch:", best_epoch)
    print("Best minimum validation DSC:", best_min_dice)
    print("Test loss:", test_loss)
    print("Training time (s):", elapsed)

    print_dice(
        test_dice,
        raw_values,
        "Test DSC by label:",
    )

    requirement_met = bool(
        np.all(test_dice > 0.90)
    )

    print(
        "Requirement >0.90 DSC for ALL labels:",
        "ACHIEVED" if requirement_met else "NOT YET ACHIEVED",
    )

    save_segmentation_examples(
        model,
        test_loader,
        device,
        "p4_unet_segmentation.png",
        num_classes,
    )

    # Training/validation loss.
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train",
    )
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Combined loss")
    plt.title("OASIS U-Net Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "p4_unet_loss.png",
        dpi=160,
    )
    plt.close()

    # Minimum per-label validation Dice across epochs.
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(min_val_dices) + 1),
        min_val_dices,
    )
    plt.axhline(
        0.90,
        linestyle="--",
        label="Required DSC = 0.90",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Minimum validation DSC")
    plt.title("Worst-label Validation DSC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "p4_unet_dice.png",
        dpi=160,
    )
    plt.close()

    with open(
        "p4_unet_results.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(
                f"GPU: {torch.cuda.get_device_name(0)}\n"
            )

        f.write(
            f"Train pairs: {len(train_set)}\n"
        )
        f.write(
            f"Validation pairs: {len(val_set)}\n"
        )
        f.write(
            f"Test pairs: {len(test_set)}\n"
        )
        f.write(
            f"Raw mask values: {raw_values}\n"
        )
        f.write(
            f"Number of categorical classes: "
            f"{num_classes}\n"
        )
        f.write(
            f"Best epoch: {best_epoch}\n"
        )
        f.write(
            f"Best minimum validation DSC: "
            f"{best_min_dice:.6f}\n"
        )
        f.write(
            f"Test loss: {test_loss:.6f}\n"
        )

        for i, score in enumerate(test_dice):
            f.write(
                f"Label {i} "
                f"(raw {raw_values[i]}) DSC: "
                f"{score:.6f}\n"
            )

        f.write(
            "All-label DSC > 0.90: "
            f"{requirement_met}\n"
        )
        f.write(
            f"Training time (s): {elapsed:.2f}\n"
        )

    print()
    print("Saved:")
    print(f"  {args.checkpoint}")
    print("  p4_unet_results.txt")
    print("  p4_unet_loss.png")
    print("  p4_unet_dice.png")
    print("  p4_unet_segmentation.png")


def demo_mode(args):
    seed_all()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    raw_values = list(
        checkpoint["raw_values"]
    )
    num_classes = int(
        checkpoint["num_classes"]
    )
    base = int(
        checkpoint["base"]
    )

    (
        _,
        _,
        test_set,
        _,
        _,
        test_loader,
    ) = make_loaders(
        args.data_root,
        raw_values,
        args.batch_size,
        args.workers,
        device,
    )

    # Demo does not need exact training class weights for Dice metrics.
    class_weights = torch.ones(
        num_classes,
        device=device,
    )

    model = UNet(
        in_channels=1,
        num_classes=num_classes,
        base=base,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    test_loss, test_dice = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        class_weights,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start

    print()
    print("===== LIVE DEMO TEST INFERENCE =====")
    print("Checkpoint:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test pairs:", len(test_set))
    print("Categorical output channels:", num_classes)
    print("Test loss:", test_loss)
    print("Full test inference time (s):", inference_time)

    print_dice(
        test_dice,
        raw_values,
        "Test DSC by label:",
    )

    requirement_met = bool(
        np.all(test_dice > 0.90)
    )

    print(
        "Requirement >0.90 DSC for ALL labels:",
        "ACHIEVED" if requirement_met else "NOT YET ACHIEVED",
    )

    save_segmentation_examples(
        model,
        test_loader,
        device,
        "p4_unet_demo_segmentation.png",
        num_classes,
    )

    print(
        "Saved live-demo visualisation: "
        "p4_unet_demo_segmentation.png"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "demo"],
        default="train",
    )

    parser.add_argument(
        "--data-root",
        default="/home/groups/comp3710/OASIS",
    )

    parser.add_argument(
        "--checkpoint",
        default="p4_unet_best.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--base",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--target-dice",
        type=float,
        default=0.905,
    )

    parser.add_argument(
        "--target-patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--label-scan",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--weight-scan",
        type=int,
        default=400,
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    else:
        demo_mode(args)


if __name__ == "__main__":
    main()
