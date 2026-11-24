#!/usr/bin/env python3
"""
CNN Architecture Laboratory
LeNet, AlexNet, VGG16, and Inception v1.

Requirements:
    Python 3.10+
    PyTorch 2.0+

Examples:
    python cnn_architectures.py
    python cnn_architectures.py --architecture lenet
    python cnn_architectures.py --architecture alexnet --input-size 224
    python cnn_architectures.py --architecture vgg16 --input-size 224
    python cnn_architectures.py --architecture inception --input-size 224
    python cnn_architectures.py --architecture all
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

LOGGER = logging.getLogger("cnn-laboratory")


class Architecture(Enum):
    LENET = "lenet"
    ALEXNET = "alexnet"
    VGG16 = "vgg16"
    INCEPTION = "inception"


@dataclass(frozen=True)
class ArchitectureInfo:
    name: str
    year: int
    input_size: int
    description: str
    key_innovation: str


ARCHITECTURES = {
    Architecture.LENET: ArchitectureInfo(
        "LeNet-5", 1998, 32,
        "Early convolutional neural network for handwritten digit recognition.",
        "Convolution, subsampling and fully connected layers."
    ),
    Architecture.ALEXNET: ArchitectureInfo(
        "AlexNet", 2012, 224,
        "Deep CNN that demonstrated GPU-trained image classification.",
        "ReLU, dropout, overlapping pooling and data augmentation."
    ),
    Architecture.VGG16: ArchitectureInfo(
        "VGG16", 2014, 224,
        "Deep network built using repeated small convolution filters.",
        "Uniform 3x3 convolutions and 2x2 pooling."
    ),
    Architecture.INCEPTION: ArchitectureInfo(
        "Inception v1", 2014, 224,
        "Multi-branch network processing different receptive field sizes.",
        "Parallel convolution branches and 1x1 bottleneck convolutions."
    ),
}


class LeNet(nn.Module):
    """LeNet-5 adapted for 32x32 RGB or grayscale inputs."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNet(nn.Module):
    """AlexNet with an adaptive classifier input."""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16: thirteen convolutional and three fully connected layers."""

    CONFIG = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        classifier_width: int = 4096,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for item in self.CONFIG:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend([
                    nn.Conv2d(
                        current_channels,
                        item,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.ReLU(inplace=True),
                ])
                current_channels = item

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, classifier_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_width, classifier_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_width, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class InceptionBlock(nn.Module):
    """
    Inception module with four parallel branches.

    Branch 1: 1x1 convolution.
    Branch 2: 1x1 reduction followed by 3x3 convolution.
    Branch 3: 1x1 reduction followed by 5x5 convolution.
    Branch 4: 3x3 max pooling followed by 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c3_reduce: int,
        c3: int,
        c5_reduce: int,
        c5: int,
        pool_proj: int,
    ):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3_reduce, c3, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5_reduce, c5, 5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ],
            dim=1,
        )


class InceptionV1(nn.Module):
    """Compact Inception-v1-style classifier with multi-branch modules."""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.inception3a = InceptionBlock(
            192, 64, 96, 128, 16, 32, 32
        )

        self.inception3b = InceptionBlock(
            256, 128, 128, 192, 32, 96, 64
        )

        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionBlock(
            480, 192, 96, 208, 16, 48, 64
        )

        self.inception4b = InceptionBlock(
            512, 160, 112, 224, 24, 64, 64
        )

        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionBlock(
            512, 256, 160, 320, 32, 128, 128
        )

        self.inception5b = InceptionBlock(
            832, 384, 192, 384, 48, 128, 128
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


MODEL_FACTORIES: dict[Architecture, Callable[..., nn.Module]] = {
    Architecture.LENET: LeNet,
    Architecture.ALEXNET: AlexNet,
    Architecture.VGG16: VGG16,
    Architecture.INCEPTION: InceptionV1,
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size(model: nn.Module) -> float:
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def validate_input(
    architecture: Architecture,
    input_size: int,
    channels: int,
    batch_size: int,
    num_classes: int,
) -> None:
    if input_size < 32:
        raise ValueError("Input dimensions must be at least 32.")

    if channels not in (1, 3):
        raise ValueError("Only grayscale and RGB inputs are supported.")

    if batch_size < 1 or batch_size > 256:
        raise ValueError("Batch size must be between 1 and 256.")

    if num_classes < 2:
        raise ValueError("Classification requires at least two classes.")

    if architecture == Architecture.LENET and input_size < 32:
        raise ValueError("LeNet requires at least 32x32 inputs.")


def create_model(
    architecture: Architecture,
    num_classes: int,
    channels: int,
) -> nn.Module:
    return MODEL_FACTORIES[architecture](
        num_classes=num_classes,
        in_channels=channels,
    )


def inspect_model(
    architecture: Architecture,
    input_size: int,
    batch_size: int,
    channels: int,
    num_classes: int,
    device: torch.device,
) -> dict:
    validate_input(
        architecture,
        input_size,
        channels,
        batch_size,
        num_classes,
    )

    model = create_model(architecture, num_classes, channels)
    model = model.to(device)
    model.eval()

    dummy = torch.randn(
        batch_size,
        channels,
        input_size,
        input_size,
        device=device,
    )

    # Inference mode avoids allocating gradients during architecture inspection.
    with torch.inference_mode():
        output = model(dummy)

    if output.shape != (batch_size, num_classes):
        raise RuntimeError(
            f"Unexpected output shape: {tuple(output.shape)}"
        )

    result = {
        "architecture": ARCHITECTURES[architecture].name,
        "input_shape": tuple(dummy.shape),
        "output_shape": tuple(output.shape),
        "parameters": count_parameters(model),
        "size_mb": calculate_model_size(model),
        "device": str(device),
    }

    return result


def demonstrate_training_step(
    model: nn.Module,
    batch_size: int,
    input_size: int,
    channels: int,
    num_classes: int,
    device: torch.device,
) -> float:
    """Execute one synthetic optimization step."""

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001,
    )

    inputs = torch.randn(
        batch_size,
        channels,
        input_size,
        input_size,
        device=device,
    )

    targets = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)

    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss.")

    loss.backward()

    # Gradient clipping limits extreme updates.
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()

    return float(loss.detach().cpu())


def print_architecture_catalog() -> None:
    print("\nCNN ARCHITECTURE CATALOG")
    print("=" * 78)

    for architecture, info in ARCHITECTURES.items():
        print(
            f"{info.name:12} | {info.year} | "
            f"{info.input_size}x{info.input_size} | "
            f"{info.key_innovation}"
        )


def run(args) -> None:
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print_architecture_catalog()

    selected = (
        list(Architecture)
        if args.architecture == "all"
        else [Architecture(args.architecture)]
    )

    for architecture in selected:
        info = ARCHITECTURES[architecture]
        input_size = args.input_size or info.input_size

        LOGGER.info("Inspecting %s", info.name)

        model = create_model(
            architecture,
            args.classes,
            args.channels,
        )

        model = model.to(device)

        dummy = torch.randn(
            args.batch_size,
            args.channels,
            input_size,
            input_size,
            device=device,
        )

        model.eval()

        with torch.inference_mode():
            output = model(dummy)

        assert output.shape == (args.batch_size, args.classes)

        print("\n" + "-" * 65)
        print(f"Architecture: {info.name}")
        print(f"Input:        {tuple(dummy.shape)}")
        print(f"Output:       {tuple(output.shape)}")
        print(f"Parameters:   {count_parameters(model):,}")
        print(f"Model size:   {calculate_model_size(model):.2f} MB")
        print(f"Device:       {device}")

        if args.train_step:
            loss = demonstrate_training_step(
                model,
                args.batch_size,
                input_size,
                args.channels,
                args.classes,
                device,
            )
            print(f"Training loss: {loss:.6f}")

        del model, dummy, output

        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN architecture comparison laboratory"
    )

    parser.add_argument(
        "--architecture",
        choices=["lenet", "alexnet", "vgg16", "inception", "all"],
        default="lenet",
    )

    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--train-step", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        run(args)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
