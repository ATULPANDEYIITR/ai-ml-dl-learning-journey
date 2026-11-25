#!/usr/bin/env python3
"""
Modern Convolutional Neural Networks
ResNet, DenseNet, EfficientNet, and Residual Connections.

A self-contained architecture modeling and computational cost analysis tool.
Uses only the Python standard library.

Features:
- CNN tensor-shape calculations
- Convolution and pooling operations
- Residual connections and projection shortcuts
- Dense connectivity and channel concatenation
- Squeeze-and-Excitation blocks
- EfficientNet compound scaling
- Parameter and FLOP estimation
- Architecture validation
- Model comparison and JSON reporting
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("modern_cnn")


class ArchitectureError(ValueError):
    """Raised when an architecture configuration is invalid."""


class LayerType(str, Enum):
    CONV = "convolution"
    DEPTHWISE = "depthwise_convolution"
    POOL = "pooling"
    DENSE = "dense_layer"
    SE = "squeeze_excitation"
    RESIDUAL = "residual_add"
    CONCAT = "concatenation"
    BATCH_NORM = "batch_normalization"
    ACTIVATION = "activation"


@dataclass(frozen=True)
class TensorShape:
    channels: int
    height: int
    width: int

    def __post_init__(self):
        if min(self.channels, self.height, self.width) <= 0:
            raise ArchitectureError("Tensor dimensions must be positive.")

    @property
    def elements(self) -> int:
        return self.channels * self.height * self.width


@dataclass
class Layer:
    name: str
    layer_type: LayerType
    input_shape: TensorShape
    output_shape: TensorShape
    parameters: int = 0
    flops: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Architecture:
    name: str
    family: str
    input_shape: TensorShape
    layers: list[Layer] = field(default_factory=list)

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    @property
    def parameters(self) -> int:
        return sum(layer.parameters for layer in self.layers)

    @property
    def flops(self) -> int:
        return sum(layer.flops for layer in self.layers)

    @property
    def output_shape(self) -> TensorShape:
        return self.layers[-1].output_shape if self.layers else self.input_shape

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "input_shape": asdict(self.input_shape),
            "output_shape": asdict(self.output_shape),
            "layer_count": len(self.layers),
            "parameters": self.parameters,
            "parameters_millions": round(self.parameters / 1e6, 3),
            "flops": self.flops,
            "gflops": round(self.flops / 1e9, 4),
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.layer_type.value,
                    "input": asdict(layer.input_shape),
                    "output": asdict(layer.output_shape),
                    "parameters": layer.parameters,
                    "flops": layer.flops,
                    "details": layer.details
                }
                for layer in self.layers
            ]
        }


def conv_output_size(
    size: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0
) -> int:
    if min(size, kernel, stride) <= 0 or padding < 0:
        raise ArchitectureError("Invalid convolution dimensions.")

    result = (size + 2 * padding - kernel) // stride + 1

    if result <= 0:
        raise ArchitectureError("Convolution produces invalid dimensions.")

    return result


def add_convolution(
    model: Architecture,
    name: str,
    output_channels: int,
    kernel: int,
    stride: int = 1,
    padding: int | None = None,
    groups: int = 1,
    bias: bool = False
) -> TensorShape:
    source = model.output_shape

    if output_channels <= 0 or groups <= 0:
        raise ArchitectureError("Channels and groups must be positive.")

    if source.channels % groups or output_channels % groups:
        raise ArchitectureError("Channels must be divisible by groups.")

    if padding is None:
        padding = kernel // 2

    height = conv_output_size(source.height, kernel, stride, padding)
    width = conv_output_size(source.width, kernel, stride, padding)

    output = TensorShape(output_channels, height, width)

    parameters = (
        output_channels
        * (source.channels // groups)
        * kernel
        * kernel
        + (output_channels if bias else 0)
    )

    flops = (
        2 * output_channels * height * width
        * (source.channels // groups)
        * kernel * kernel
    )

    model.add(Layer(
        name,
        LayerType.DEPTHWISE if groups == source.channels else LayerType.CONV,
        source,
        output,
        parameters,
        flops,
        {
            "kernel": kernel,
            "stride": stride,
            "groups": groups,
            "bias": bias
        }
    ))

    return output


def add_batch_norm(model: Architecture, name: str) -> None:
    shape = model.output_shape
    model.add(Layer(
        name,
        LayerType.BATCH_NORM,
        shape,
        shape,
        2 * shape.channels,
        4 * shape.elements,
        {"trainable_parameters": 2 * shape.channels}
    ))


def add_activation(model: Architecture, name: str) -> None:
    shape = model.output_shape
    model.add(Layer(
        name,
        LayerType.ACTIVATION,
        shape,
        shape,
        0,
        shape.elements,
        {"function": "ReLU or SiLU"}
    ))


def add_pooling(
    model: Architecture,
    name: str,
    kernel: int = 2,
    stride: int = 2
) -> None:
    source = model.output_shape
    height = conv_output_size(source.height, kernel, stride)
    width = conv_output_size(source.width, kernel, stride)

    output = TensorShape(source.channels, height, width)

    model.add(Layer(
        name,
        LayerType.POOL,
        source,
        output,
        0,
        output.elements * kernel * kernel,
        {"kernel": kernel, "stride": stride}
    ))


def add_global_average_pooling(model: Architecture, name: str) -> None:
    source = model.output_shape
    output = TensorShape(source.channels, 1, 1)

    model.add(Layer(
        name,
        LayerType.POOL,
        source,
        output,
        0,
        source.elements,
        {"operation": "global_average"}
    ))


def add_classifier(
    model: Architecture,
    name: str,
    classes: int
) -> None:
    source = model.output_shape

    if classes <= 0:
        raise ArchitectureError("Number of classes must be positive.")

    output = TensorShape(classes, 1, 1)
    parameters = source.channels * classes + classes
    flops = 2 * source.channels * classes

    model.add(Layer(
        name,
        LayerType.DENSE,
        source,
        output,
        parameters,
        flops,
        {"classes": classes}
    ))


def residual_block(
    model: Architecture,
    name: str,
    channels: int,
    stride: int = 1,
    bottleneck: bool = False
) -> None:
    """
    Implements a basic or bottleneck residual block.

    The identity shortcut is used when tensor dimensions match.
    Otherwise, a 1x1 projection shortcut aligns the dimensions.
    """
    source = model.output_shape

    if bottleneck:
        middle = channels // 4
        if middle <= 0:
            raise ArchitectureError("Bottleneck channels are too small.")

        add_convolution(model, f"{name}_reduce", middle, 1)
        add_batch_norm(model, f"{name}_bn1")
        add_activation(model, f"{name}_relu1")

        add_convolution(
            model, f"{name}_spatial", middle, 3, stride, 1
        )
        add_batch_norm(model, f"{name}_bn2")
        add_activation(model, f"{name}_relu2")

        add_convolution(model, f"{name}_expand", channels, 1)
        add_batch_norm(model, f"{name}_bn3")

    else:
        add_convolution(
            model, f"{name}_conv1", channels, 3, stride, 1
        )
        add_batch_norm(model, f"{name}_bn1")
        add_activation(model, f"{name}_relu1")

        add_convolution(model, f"{name}_conv2", channels, 3, 1, 1)
        add_batch_norm(model, f"{name}_bn2")

    transformed = model.output_shape

    if source != transformed:
        # A learned 1x1 projection makes addition dimensionally valid.
        parameters = source.channels * transformed.channels
        flops = 2 * parameters * transformed.height * transformed.width

        model.add(Layer(
            f"{name}_projection",
            LayerType.CONV,
            source,
            transformed,
            parameters,
            flops,
            {"kernel": 1, "stride": stride}
        ))

    model.add(Layer(
        f"{name}_residual_add",
        LayerType.RESIDUAL,
        transformed,
        transformed,
        0,
        transformed.elements,
        {"shortcut": "identity_or_projection"}
    ))

    add_activation(model, f"{name}_output")


def build_resnet(
    depth: int = 18,
    input_size: int = 224,
    classes: int = 1000
) -> Architecture:
    configurations = {
        18: ([2, 2, 2, 2], False),
        34: ([3, 4, 6, 3], False),
        50: ([3, 4, 6, 3], True)
    }

    if depth not in configurations:
        raise ArchitectureError("Supported ResNet depths: 18, 34, 50.")

    blocks, bottleneck = configurations[depth]

    model = Architecture(
        f"ResNet-{depth}",
        "Residual Network",
        TensorShape(3, input_size, input_size)
    )

    add_convolution(model, "stem", 64, 7, 2, 3)
    add_batch_norm(model, "stem_bn")
    add_activation(model, "stem_relu")
    add_pooling(model, "stem_maxpool", 3, 2)

    for stage, (channels, count) in enumerate(
        zip([64, 128, 256, 512], blocks)
    ):
        for index in range(count):
            stride = 2 if stage > 0 and index == 0 else 1
            residual_block(
                model,
                f"stage{stage + 1}_block{index + 1}",
                channels * (4 if bottleneck else 1),
                stride,
                bottleneck
            )

    add_global_average_pooling(model, "global_average_pool")
    add_classifier(model, "classification_head", classes)

    return model


def dense_layer(
    model: Architecture,
    name: str,
    growth_rate: int
) -> None:
    """
    DenseNet connectivity:
    Each new layer receives concatenated feature maps from
    every preceding layer in the dense block.
    """
    source = model.output_shape
    bottleneck_channels = 4 * growth_rate

    add_batch_norm(model, f"{name}_bn1")
    add_activation(model, f"{name}_relu1")
    add_convolution(model, f"{name}_bottleneck", bottleneck_channels, 1)

    add_batch_norm(model, f"{name}_bn2")
    add_activation(model, f"{name}_relu2")
    add_convolution(model, f"{name}_growth", growth_rate, 3, 1, 1)

    generated = model.output_shape

    output = TensorShape(
        source.channels + growth_rate,
        source.height,
        source.width
    )

    model.add(Layer(
        f"{name}_concatenate",
        LayerType.CONCAT,
        source,
        output,
        0,
        output.elements,
        {
            "previous_channels": source.channels,
            "new_channels": growth_rate,
            "connectivity": "all_previous_features"
        }
    ))


def transition_layer(
    model: Architecture,
    name: str,
    compression: float
) -> None:
    if not 0 < compression <= 1:
        raise ArchitectureError("Compression must be in (0, 1].")

    output_channels = max(
        1, math.floor(model.output_shape.channels * compression)
    )

    add_batch_norm(model, f"{name}_bn")
    add_activation(model, f"{name}_relu")
    add_convolution(model, f"{name}_conv", output_channels, 1)
    add_pooling(model, f"{name}_pool", 2, 2)


def build_densenet(
    growth_rate: int = 32,
    block_layers: tuple[int, ...] = (6, 12, 24, 16),
    input_size: int = 224,
    classes: int = 1000,
    compression: float = 0.5
) -> Architecture:
    if growth_rate <= 0 or not block_layers:
        raise ArchitectureError("Invalid DenseNet configuration.")

    if any(count <= 0 for count in block_layers):
        raise ArchitectureError("Dense block sizes must be positive.")

    model = Architecture(
        "DenseNet-121",
        "Densely Connected Network",
        TensorShape(3, input_size, input_size)
    )

    add_convolution(model, "stem", 64, 7, 2, 3)
    add_batch_norm(model, "stem_bn")
    add_activation(model, "stem_relu")
    add_pooling(model, "stem_pool", 3, 2)

    for block_index, count in enumerate(block_layers):
        for layer_index in range(count):
            dense_layer(
                model,
                f"block{block_index + 1}_layer{layer_index + 1}",
                growth_rate
            )

        if block_index < len(block_layers) - 1:
            transition_layer(
                model,
                f"transition{block_index + 1}",
                compression
            )

    add_batch_norm(model, "final_bn")
    add_activation(model, "final_relu")
    add_global_average_pooling(model, "global_average_pool")
    add_classifier(model, "classification_head", classes)

    return model


@dataclass(frozen=True)
class EfficientNetScale:
    width: float
    depth: float
    resolution: int
    dropout: float


EFFICIENTNET_SCALES = {
    "B0": EfficientNetScale(1.0, 1.0, 224, 0.2),
    "B1": EfficientNetScale(1.0, 1.1, 240, 0.2),
    "B2": EfficientNetScale(1.1, 1.2, 260, 0.3),
    "B3": EfficientNetScale(1.2, 1.4, 300, 0.3),
    "B4": EfficientNetScale(1.4, 1.8, 380, 0.4),
    "B5": EfficientNetScale(1.6, 2.2, 456, 0.4),
    "B6": EfficientNetScale(1.8, 2.6, 528, 0.5),
    "B7": EfficientNetScale(2.0, 3.1, 600, 0.5)
}


def round_channels(channels: int, multiplier: float) -> int:
    scaled = channels * multiplier
    divisor = 8
    rounded = max(divisor, int(scaled + divisor / 2) // divisor * divisor)

    if rounded < 0.9 * scaled:
        rounded += divisor

    return rounded


def squeeze_excitation(
    model: Architecture,
    name: str,
    reduction: int = 4
) -> None:
    source = model.output_shape
    reduced = max(1, source.channels // reduction)

    # Two fully connected transformations operate on pooled channels.
    parameters = (
        source.channels * reduced + reduced
        + reduced * source.channels + source.channels
    )

    flops = (
        2 * source.channels * reduced
        + 2 * reduced * source.channels
        + source.elements
    )

    model.add(Layer(
        name,
        LayerType.SE,
        source,
        source,
        parameters,
        flops,
        {
            "reduction": reduction,
            "squeeze": "global_average_pooling",
            "excitation": "sigmoid_gating"
        }
    ))


def mobile_inverted_block(
    model: Architecture,
    name: str,
    output_channels: int,
    expansion: int,
    kernel: int,
    stride: int
) -> None:
    source = model.output_shape
    expanded = source.channels * expansion

    if expansion != 1:
        add_convolution(model, f"{name}_expand", expanded, 1)
        add_batch_norm(model, f"{name}_expand_bn")
        add_activation(model, f"{name}_swish")

    add_convolution(
        model,
        f"{name}_depthwise",
        expanded,
        kernel,
        stride,
        kernel // 2,
        groups=expanded
    )
    add_batch_norm(model, f"{name}_depthwise_bn")
    add_activation(model, f"{name}_depthwise_swish")

    squeeze_excitation(model, f"{name}_se")

    add_convolution(model, f"{name}_project", output_channels, 1)
    add_batch_norm(model, f"{name}_project_bn")

    transformed = model.output_shape

    if stride == 1 and source.channels == output_channels:
        model.add(Layer(
            f"{name}_skip_add",
            LayerType.RESIDUAL,
            transformed,
            transformed,
            0,
            transformed.elements,
            {"shortcut": "identity"}
        ))


def build_efficientnet(
    variant: str = "B0",
    classes: int = 1000
) -> Architecture:
    if variant not in EFFICIENTNET_SCALES:
        raise ArchitectureError("Unknown EfficientNet variant.")

    scale = EFFICIENTNET_SCALES[variant]
    model = Architecture(
        f"EfficientNet-{variant}",
        "Compound-Scaled CNN",
        TensorShape(3, scale.resolution, scale.resolution)
    )

    add_convolution(
        model, "stem", round_channels(32, scale.width), 3, 2, 1
    )
    add_batch_norm(model, "stem_bn")
    add_activation(model, "stem_swish")

    # Base MBConv configuration: expansion, channels, repeats, kernel, stride.
    stages = [
        (1, 16, 1, 3, 1),
        (6, 24, 2, 3, 2),
        (6, 40, 2, 5, 2),
        (6, 80, 3, 3, 2),
        (6, 112, 3, 5, 1),
        (6, 192, 4, 5, 2),
        (6, 320, 1, 3, 1)
    ]

    for stage_index, (expansion, channels, repeats, kernel, stride) in enumerate(stages):
        scaled_repeats = math.ceil(repeats * scale.depth)
        scaled_channels = round_channels(channels, scale.width)

        for index in range(scaled_repeats):
            mobile_inverted_block(
                model,
                f"stage{stage_index + 1}_block{index + 1}",
                scaled_channels,
                expansion,
                kernel,
                stride if index == 0 else 1
            )

    add_convolution(
        model, "head_conv", round_channels(1280, scale.width), 1
    )
    add_batch_norm(model, "head_bn")
    add_activation(model, "head_swish")
    add_global_average_pooling(model, "global_average_pool")
    add_classifier(model, "classification_head", classes)

    return model


def validate_architecture(model: Architecture) -> list[str]:
    errors = []
    previous = model.input_shape

    for layer in model.layers:
        if layer.parameters < 0 or layer.flops < 0:
            errors.append(f"Negative cost in {layer.name}.")

        if layer.input_shape != previous:
            errors.append(f"Broken tensor sequence at {layer.name}.")

        previous = layer.output_shape

    return errors


def compare_models(models: list[Architecture]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "name": model.name,
                "parameters": model.parameters,
                "gflops": round(model.flops / 1e9, 4),
                "layers": len(model.layers)
            }
            for model in models
        ],
        key=lambda item: item["parameters"]
    )


def run_tests() -> None:
    assert conv_output_size(224, 7, 2, 3) == 112
    assert round_channels(32, 1.0) == 32
    assert round_channels(35, 1.0) == 32

    resnet = build_resnet(18, 64, 10)
    assert not validate_architecture(resnet)
    assert resnet.output_shape == TensorShape(10, 1, 1)

    dense = build_densenet(
        growth_rate=12,
        block_layers=(2, 2, 2),
        input_size=64,
        classes=10
    )
    assert not validate_architecture(dense)
    assert dense.output_shape.channels == 10

    efficient = build_efficientnet("B0", 10)
    assert not validate_architecture(efficient)
    assert efficient.output_shape.channels == 10

    try:
        build_resnet(99)
        raise AssertionError("Invalid ResNet depth was accepted.")
    except ArchitectureError:
        pass

    logger.info("All architecture tests passed.")


def main() -> None:
    run_tests()

    models = [
        build_resnet(18, 224, 1000),
        build_resnet(50, 224, 1000),
        build_densenet(),
        build_efficientnet("B0"),
        build_efficientnet("B3")
    ]

    print("\nMODERN CNN ARCHITECTURE COMPARISON")
    print("=" * 90)

    for model in models:
        if validate_architecture(model):
            raise ArchitectureError(f"Invalid architecture: {model.name}")

        print(
            f"{model.name:20} "
            f"Parameters: {model.parameters / 1e6:9.3f} M "
            f"FLOPs: {model.flops / 1e9:9.3f} G "
            f"Layers: {len(model.layers):4}"
        )

    print("\nPARAMETER-ORDERED COMPARISON")
    for result in compare_models(models):
        print(result)

    output = {
        "experiment": "Modern CNN Architecture Analysis",
        "models": [model.report() for model in models]
    }

    output_path = Path("modern_cnn_report.json")
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    print(f"\nDetailed report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ArchitectureError, OSError, ValueError) as exc:
        logger.error("Execution failed: %s", exc)
        raise
