import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)
        return out


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        # CIFAR-10 input is only 32x32, so use a CIFAR-style 3x3 stem.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 = [2, 2, 2, 2] BasicBlocks.
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialise_weights()

    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def _initialise_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def make_loaders(batch_size, workers):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    root = os.path.join(os.environ.get("HOME", "."), "data")

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )

    common = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    if workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, test_loader


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(x)

        correct += (logits.argmax(dim=1) == y).sum()
        total += y.size(0)

    return correct.item() / total


def load_checkpoint(model, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    print(
        f"Loaded checkpoint: {checkpoint} | "
        f"epoch={ckpt.get('epoch', '?')} | "
        f"test_acc={100 * ckpt.get('test_acc', 0):.2f}%"
    )


def train_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scaler,
    scheduler,
):
    model.train()

    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Keep these accumulators on the GPU. Calling .item() every batch
        # forces a CPU/GPU synchronisation and slows training.
        batch_n = y.size(0)
        loss_sum += loss.detach() * batch_n
        correct += (logits.argmax(dim=1) == y).sum()
        total += batch_n

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return (
        loss_sum.item() / total,
        correct.item() / total,
        elapsed,
    )


def main():
    parser = argparse.ArgumentParser()

    # Training configuration tuned for efficient CIFAR-10 experiments.
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=0.2)

    # Evaluate periodically to reduce unnecessary validation overhead.
    parser.add_argument("--eval-start", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)

    # Optional target accuracy used for early stopping.
    parser.add_argument("--target-acc", type=float, default=94.0)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="resnet18_cifar10_fast.pt",
    )

    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--demo", action="store_true")

    args = parser.parse_args()

    seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_loader, test_loader = make_loaders(
        args.batch_size,
        args.workers,
    )

    model = ResNet18CIFAR(num_classes=10).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    dummy = torch.zeros(2, 3, 32, 32, device=device)

    if device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        print("Model output shape:", tuple(model(dummy).shape))

    if args.inference_only:
        load_checkpoint(model, args.checkpoint, device)
        acc = evaluate(model, test_loader, device)
        print(f"Inference accuracy: {100 * acc:.2f}%")
        return

    if args.demo:
        load_checkpoint(model, args.checkpoint, device)

        before = evaluate(model, test_loader, device)
        print(f"Initial inference accuracy: {100 * before:.2f}%")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
            foreach=True,
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda"),
        )

        # Use a constant one-epoch scheduler so the shared training function
        # can be reused in single-epoch evaluation mode.
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )

        loss, train_acc, elapsed = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            scheduler,
        )

        after = evaluate(model, test_loader, device)

        print(
            f"One-epoch update | loss={loss:.4f} | "
            f"train_acc={100 * train_acc:.2f}% | "
            f"test_acc={100 * after:.2f}% | "
            f"time={elapsed:.1f}s"
        )
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.max_lr / 10,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
        foreach=True,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_acc = 0.0
    best_epoch = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            scheduler,
        )

        do_eval = (
            epoch == args.epochs
            or (
                epoch >= args.eval_start
                and epoch % args.eval_every == 0
            )
        )

        if do_eval:
            test_acc = evaluate(model, test_loader, device)

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"loss={loss:.4f} | "
                f"train_acc={100 * train_acc:.2f}% | "
                f"test_acc={100 * test_acc:.2f}% | "
                f"train_time={epoch_time:.2f}s"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "test_acc": test_acc,
                    },
                    args.checkpoint,
                )

            # Stop as soon as the challenge accuracy is reached.
            if 100 * test_acc >= args.target_acc:
                print(
                    f"Target reached: {100 * test_acc:.2f}% "
                    f">= {args.target_acc:.2f}%"
                )
                break

        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"loss={loss:.4f} | "
                f"train_acc={100 * train_acc:.2f}% | "
                f"train_time={epoch_time:.2f}s"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - total_start

    print()
    print(f"Best test accuracy: {100 * best_acc:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Total train+evaluation time: {total_time:.1f}s")
    print("Saved checkpoint:", args.checkpoint)

    if best_acc >= args.target_acc / 100:
        print("DAWNBench 94% accuracy target: ACHIEVED")
    else:
        print("DAWNBench 94% accuracy target: NOT YET ACHIEVED")

    if total_time <= 360:
        print("Approx. 360-second time target: ACHIEVED")
    else:
        print("Approx. 360-second time target: NOT YET ACHIEVED")


if __name__ == "__main__":
    main()
