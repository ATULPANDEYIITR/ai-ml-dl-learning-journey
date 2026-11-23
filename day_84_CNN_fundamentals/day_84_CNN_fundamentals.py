#!/usr/bin/env python3
"""
CNN Fundamentals: Convolution, Filters, Kernels, Stride, and Padding.

A dependency-free implementation of two-dimensional convolution for
multi-channel inputs, multiple filters, configurable stride, VALID/SAME
padding, ReLU activation, and max pooling.

Run:
    python cnn_fundamentals.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple
import math


Tensor2D = List[List[float]]
Tensor3D = List[Tensor2D]


class Padding(Enum):
    VALID = "valid"
    SAME = "same"


class CNNError(Exception):
    """Base exception for CNN operations."""


class ShapeError(CNNError):
    """Raised when tensor dimensions are incompatible."""


class ConfigurationError(CNNError):
    """Raised when an operation has invalid parameters."""


@dataclass(frozen=True)
class ConvConfig:
    stride: int = 1
    padding: Padding = Padding.VALID

    def __post_init__(self):
        if self.stride < 1:
            raise ConfigurationError("Stride must be positive.")


@dataclass
class ConvResult:
    values: Tensor3D
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    padding_applied: Tuple[int, int, int, int]
    stride: int


def validate_matrix(matrix: Tensor2D) -> Tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ShapeError("Matrix cannot be empty.")

    width = len(matrix[0])

    for row in matrix:
        if len(row) != width:
            raise ShapeError("Matrix must be rectangular.")

        for value in row:
            if not math.isfinite(float(value)):
                raise ValueError("Matrix contains a non-finite value.")

    return len(matrix), width


def validate_tensor(tensor: Tensor3D) -> Tuple[int, int, int]:
    if not tensor:
        raise ShapeError("Tensor must contain at least one channel.")

    dimensions = [validate_matrix(channel) for channel in tensor]

    if len(set(dimensions)) != 1:
        raise ShapeError("All channels must have identical dimensions.")

    height, width = dimensions[0]
    return len(tensor), height, width


def output_dimension(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: Padding
) -> int:
    if kernel_size > input_size and padding == Padding.VALID:
        raise ShapeError("Kernel cannot exceed input dimensions with VALID padding.")

    if padding == Padding.SAME:
        return math.ceil(input_size / stride)

    return (input_size - kernel_size) // stride + 1


def calculate_padding(
    input_height: int,
    input_width: int,
    kernel_height: int,
    kernel_width: int,
    stride: int,
    padding: Padding
) -> Tuple[int, int, int, int]:

    if padding == Padding.VALID:
        return 0, 0, 0, 0

    output_height = math.ceil(input_height / stride)
    output_width = math.ceil(input_width / stride)

    total_height = max(
        (output_height - 1) * stride + kernel_height - input_height,
        0
    )

    total_width = max(
        (output_width - 1) * stride + kernel_width - input_width,
        0
    )

    top = total_height // 2
    bottom = total_height - top
    left = total_width // 2
    right = total_width - left

    return top, bottom, left, right


def pad_tensor(
    tensor: Tensor3D,
    padding: Tuple[int, int, int, int]
) -> Tensor3D:

    top, bottom, left, right = padding
    padded = []

    for channel in tensor:
        width = len(channel[0]) + left + right
        padded_channel = []

        for _ in range(top):
            padded_channel.append([0.0] * width)

        for row in channel:
            padded_channel.append(
                [0.0] * left +
                [float(value) for value in row] +
                [0.0] * right
            )

        for _ in range(bottom):
            padded_channel.append([0.0] * width)

        padded.append(padded_channel)

    return padded


def correlate2d(
    matrix: Tensor2D,
    kernel: Tensor2D,
    stride: int = 1,
    padding: Padding = Padding.VALID
) -> Tensor2D:
    """
    Compute 2D cross-correlation.

    Deep-learning libraries commonly call this operation convolution,
    although mathematical convolution formally flips the kernel.
    """

    height, width = validate_matrix(matrix)
    kernel_height, kernel_width = validate_matrix(kernel)

    if stride < 1:
        raise ConfigurationError("Stride must be positive.")

    if not isinstance(padding, Padding):
        raise ConfigurationError("Invalid padding mode.")

    pad = calculate_padding(
        height, width,
        kernel_height, kernel_width,
        stride, padding
    )

    padded = pad_tensor([matrix], pad)[0]

    output_height = output_dimension(
        height, kernel_height, stride, padding
    )

    output_width = output_dimension(
        width, kernel_width, stride, padding
    )

    output = []

    for output_row in range(output_height):
        current_row = []

        for output_col in range(output_width):
            total = 0.0

            for kernel_row in range(kernel_height):
                for kernel_col in range(kernel_width):
                    input_value = padded[
                        output_row * stride + kernel_row
                    ][
                        output_col * stride + kernel_col
                    ]

                    total += (
                        input_value *
                        kernel[kernel_row][kernel_col]
                    )

            current_row.append(total)

        output.append(current_row)

    return output


class FilterBank:
    """
    A filter bank containing multiple multi-channel filters.

    Each filter has shape:
        input_channels x kernel_height x kernel_width
    """

    def __init__(
        self,
        filters: Sequence[Tensor3D],
        biases: Sequence[float] = None
    ):
        if not filters:
            raise ConfigurationError("At least one filter is required.")

        self.filters = filters

        channel_count = len(filters[0])

        if channel_count == 0:
            raise ShapeError("Filters must contain input channels.")

        kernel_dimensions = []

        for current_filter in filters:
            if len(current_filter) != channel_count:
                raise ShapeError("Filter channel counts must match.")

            dimensions = [
                validate_matrix(channel)
                for channel in current_filter
            ]

            if len(set(dimensions)) != 1:
                raise ShapeError("Filter kernels must have equal dimensions.")

            kernel_dimensions.append(dimensions[0])

        if len(set(kernel_dimensions)) != 1:
            raise ShapeError("All filters must have identical kernel dimensions.")

        self.input_channels = channel_count
        self.kernel_height, self.kernel_width = kernel_dimensions[0]

        if biases is None:
            biases = [0.0] * len(filters)

        if len(biases) != len(filters):
            raise ConfigurationError("One bias is required per filter.")

        if not all(math.isfinite(float(b)) for b in biases):
            raise ValueError("Biases must be finite.")

        self.biases = list(map(float, biases))

    def apply(
        self,
        input_tensor: Tensor3D,
        config: ConvConfig
    ) -> ConvResult:

        channels, height, width = validate_tensor(input_tensor)

        if channels != self.input_channels:
            raise ShapeError(
                f"Expected {self.input_channels} channels, received {channels}."
            )

        padding = calculate_padding(
            height,
            width,
            self.kernel_height,
            self.kernel_width,
            config.stride,
            config.padding
        )

        padded = pad_tensor(input_tensor, padding)

        output_height = output_dimension(
            height,
            self.kernel_height,
            config.stride,
            config.padding
        )

        output_width = output_dimension(
            width,
            self.kernel_width,
            config.stride,
            config.padding
        )

        outputs = []

        for filter_index, current_filter in enumerate(self.filters):
            feature_map = []

            for output_row in range(output_height):
                row = []

                for output_col in range(output_width):
                    total = self.biases[filter_index]

                    for channel_index in range(channels):
                        channel = padded[channel_index]
                        kernel = current_filter[channel_index]

                        for kernel_row in range(self.kernel_height):
                            for kernel_col in range(self.kernel_width):

                                input_value = channel[
                                    output_row * config.stride + kernel_row
                                ][
                                    output_col * config.stride + kernel_col
                                ]

                                total += (
                                    input_value *
                                    kernel[kernel_row][kernel_col]
                                )

                    row.append(total)

                feature_map.append(row)

            outputs.append(feature_map)

        return ConvResult(
            values=outputs,
            input_shape=(channels, height, width),
            output_shape=(
                len(outputs),
                output_height,
                output_width
            ),
            padding_applied=padding,
            stride=config.stride
        )


def relu(tensor: Tensor3D) -> Tensor3D:
    """Apply the ReLU activation function element by element."""

    return [
        [
            [max(0.0, value) for value in row]
            for row in channel
        ]
        for channel in tensor
    ]


def sigmoid(tensor: Tensor3D) -> Tensor3D:
    """Numerically stable sigmoid activation."""

    def activate(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))

        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    return [
        [
            [activate(value) for value in row]
            for row in channel
        ]
        for channel in tensor
    ]


def max_pool2d(
    tensor: Tensor3D,
    pool_size: int = 2,
    stride: int = 2
) -> Tensor3D:
    """Apply non-overlapping or overlapping maximum pooling."""

    channels, height, width = validate_tensor(tensor)

    if pool_size < 1 or stride < 1:
        raise ConfigurationError("Pool size and stride must be positive.")

    if pool_size > height or pool_size > width:
        raise ShapeError("Pooling window exceeds input dimensions.")

    output_height = (height - pool_size) // stride + 1
    output_width = (width - pool_size) // stride + 1

    outputs = []

    for channel in tensor:
        feature_map = []

        for row in range(output_height):
            current_row = []

            for col in range(output_width):
                values = [
                    channel[row * stride + r][col * stride + c]
                    for r in range(pool_size)
                    for c in range(pool_size)
                ]

                current_row.append(max(values))

            feature_map.append(current_row)

        outputs.append(feature_map)

    return outputs


def count_parameters(filter_bank: FilterBank) -> int:
    """Count trainable weights and biases."""

    weights_per_filter = (
        filter_bank.input_channels *
        filter_bank.kernel_height *
        filter_bank.kernel_width
    )

    return len(filter_bank.filters) * (weights_per_filter + 1)


def print_tensor(name: str, tensor: Tensor3D):
    print(f"\n{name}")

    for channel_index, channel in enumerate(tensor):
        print(f"Feature map {channel_index + 1}:")

        for row in channel:
            print("  " + " ".join(f"{value:8.3f}" for value in row))


def demonstrate_basic_convolution():
    print("=" * 65)
    print("1. BASIC CONVOLUTION AND EDGE DETECTION")
    print("=" * 65)

    image = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ]

    vertical_edge_kernel = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]

    result = correlate2d(
        image,
        vertical_edge_kernel,
        stride=1,
        padding=Padding.VALID
    )

    print("Input dimensions: 5 x 5")
    print("Kernel dimensions: 3 x 3")
    print("Output dimensions: 3 x 3")

    print_tensor("Vertical edge feature map", [result])


def demonstrate_stride_and_padding():
    print("\n" + "=" * 65)
    print("2. STRIDE AND PADDING")
    print("=" * 65)

    image = [[float(i * 5 + j + 1) for j in range(5)] for i in range(5)]

    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]

    for padding in (Padding.VALID, Padding.SAME):
        for stride in (1, 2):
            result = correlate2d(
                image,
                kernel,
                stride=stride,
                padding=padding
            )

            print(
                f"\nPadding={padding.value}, stride={stride}, "
                f"output={len(result)}x{len(result[0])}"
            )

            print_tensor("Result", [result])


def demonstrate_multi_channel_convolution():
    print("\n" + "=" * 65)
    print("3. MULTI-CHANNEL CONVOLUTION")
    print("=" * 65)

    rgb_image = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [16, 15, 14, 13],
            [12, 11, 10, 9],
            [8, 7, 6, 5],
            [4, 3, 2, 1]
        ],
        [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1]
        ]
    ]

    filter_1 = [
        [[1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
        [[1, 1], [1, 1]]
    ]

    filter_2 = [
        [[-1, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, -1], [1, 0]]
    ]

    bank = FilterBank(
        filters=[filter_1, filter_2],
        biases=[0.5, -1.0]
    )

    result = bank.apply(
        rgb_image,
        ConvConfig(stride=1, padding=Padding.VALID)
    )

    print("Input shape:", result.input_shape)
    print("Output shape:", result.output_shape)
    print("Trainable parameters:", count_parameters(bank))

    print_tensor("Multi-filter output", result.values)


def demonstrate_activation_and_pooling():
    print("\n" + "=" * 65)
    print("4. ACTIVATION AND MAX POOLING")
    print("=" * 65)

    feature_maps = [
        [
            [-2, 1, 3, 4],
            [5, -1, 2, 8],
            [1, 7, -3, 2],
            [4, 6, 5, -2]
        ]
    ]

    activated = relu(feature_maps)
    pooled = max_pool2d(activated, pool_size=2, stride=2)

    print_tensor("Before ReLU", feature_maps)
    print_tensor("After ReLU", activated)
    print_tensor("After max pooling", pooled)


def demonstrate_parameter_calculation():
    print("\n" + "=" * 65)
    print("5. PARAMETER COUNT AND MODEL STRUCTURE")
    print("=" * 65)

    input_channels = 3
    number_of_filters = 16
    kernel_height = 3
    kernel_width = 3

    weights = (
        input_channels *
        number_of_filters *
        kernel_height *
        kernel_width
    )

    biases = number_of_filters

    print("Input channels:", input_channels)
    print("Number of filters:", number_of_filters)
    print("Kernel dimensions: 3 x 3")
    print("Weight parameters:", weights)
    print("Bias parameters:", biases)
    print("Total trainable parameters:", weights + biases)

    print("\nOutput shape calculation:")

    height, width = 32, 32
    kernel, stride = 3, 1

    valid = (
        (height - kernel) // stride + 1,
        (width - kernel) // stride + 1
    )

    same = (
        math.ceil(height / stride),
        math.ceil(width / stride)
    )

    print("Input: 32 x 32")
    print("VALID output:", valid)
    print("SAME output:", same)


def run_tests():
    print("\n" + "=" * 65)
    print("6. VALIDATION AND EDGE CASE TESTS")
    print("=" * 65)

    identity = [[1, 0], [0, 1]]
    image = [[1, 2], [3, 4]]

    result = correlate2d(image, identity)
    assert result == [[5.0]]

    same = correlate2d(
        image,
        [[1]],
        stride=2,
        padding=Padding.SAME
    )
    assert len(same) == 1
    assert len(same[0]) == 1

    try:
        correlate2d([[1, 2], [3]], [[1]])
        raise AssertionError("Ragged matrices should fail.")
    except ShapeError:
        pass

    try:
        correlate2d([[1]], [[1, 2], [3, 4]])
        raise AssertionError("Oversized VALID kernel should fail.")
    except ShapeError:
        pass

    try:
        ConvConfig(stride=0)
        raise AssertionError("Zero stride should fail.")
    except ConfigurationError:
        pass

    try:
        validate_matrix([[float("nan")]])
        raise AssertionError("NaN should fail.")
    except ValueError:
        pass

    print("Identity-kernel test: PASSED")
    print("SAME-padding test: PASSED")
    print("Ragged-matrix validation: PASSED")
    print("Oversized-kernel validation: PASSED")
    print("Invalid-stride validation: PASSED")
    print("Non-finite-value validation: PASSED")


def main():
    demonstrate_basic_convolution()
    demonstrate_stride_and_padding()
    demonstrate_multi_channel_convolution()
    demonstrate_activation_and_pooling()
    demonstrate_parameter_calculation()
    run_tests()

    print("\nCNN fundamentals demonstration completed successfully.")


if __name__ == "__main__":
    main()
