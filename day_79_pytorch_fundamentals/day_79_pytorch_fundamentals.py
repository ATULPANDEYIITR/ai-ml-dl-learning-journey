#!/usr/bin/env python3
"""
PyTorch Fundamentals
Tensors, devices, autograd, tensor operations, and model optimization.

Requirements:
    Python 3.10+
    PyTorch 2.x

Run:
    python pytorch_fundamentals.py
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------
# Configuration and reproducibility
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_printoptions(precision=4, sci_mode=False)


@dataclass(frozen=True)
class TensorConfig:
    dtype: torch.dtype = torch.float32
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        device = torch.device(self.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        return device


# ------------------------------------------------------------
# Tensor validation and construction
# ------------------------------------------------------------

def validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating-point dtype.")

    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or infinite values.")


def tensor_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("1. TENSOR FUNDAMENTALS")
    print("=" * 65)

    scalar = torch.tensor(7.0, device=device)
    vector = torch.tensor([1.0, 2.0, 3.0], device=device)
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device
    )

    zeros = torch.zeros((2, 3), device=device)
    ones = torch.ones((2, 3), device=device)
    random_tensor = torch.randn((2, 3), device=device)

    for name, tensor in [
        ("Scalar", scalar),
        ("Vector", vector),
        ("Matrix", matrix),
        ("Zeros", zeros),
        ("Ones", ones),
        ("Random", random_tensor),
    ]:
        print(
            f"{name:10} shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}"
        )
        print(tensor)

    print("Dimensions:", matrix.ndim)
    print("Number of elements:", matrix.numel())
    print("Memory layout:", matrix.layout)
    print("Element size:", matrix.element_size())


# ------------------------------------------------------------
# Tensor operations
# ------------------------------------------------------------

def tensor_operations(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("2. TENSOR OPERATIONS")
    print("=" * 65)

    a = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device
    )

    b = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        device=device
    )

    print("Addition:\n", a + b)
    print("Subtraction:\n", a - b)
    print("Element-wise multiplication:\n", a * b)
    print("Division:\n", b / a)

    print("Matrix multiplication:\n", torch.matmul(a, b))
    print("Transpose:\n", a.T)

    print("Sum:", a.sum().item())
    print("Mean:", a.mean().item())
    print("Maximum:", a.max().item())
    print("Minimum:", a.min().item())

    print("Row sums:", a.sum(dim=1))
    print("Column sums:", a.sum(dim=0))

    reshaped = a.reshape(4)
    print("Reshaped:", reshaped)

    stacked = torch.stack([a, b], dim=0)
    print("Stacked shape:", stacked.shape)

    concatenated = torch.cat([a, b], dim=0)
    print("Concatenated shape:", concatenated.shape)

    # Broadcasting expands compatible dimensions logically.
    bias = torch.tensor([10.0, 20.0], device=device)
    print("Broadcasting:\n", a + bias)

    # Boolean indexing.
    print("Values greater than 2:", a[a > 2])

    # In-place operations modify the existing tensor.
    mutable = a.clone()
    mutable.add_(1)
    print("In-place addition:\n", mutable)

    # Explicit shape checks prevent accidental broadcasting.
    assert a.shape == b.shape
    assert torch.matmul(a, b).shape == (2, 2)


# ------------------------------------------------------------
# Tensor views and memory
# ------------------------------------------------------------

def tensor_memory_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("3. TENSOR STORAGE AND VIEWS")
    print("=" * 65)

    original = torch.arange(12, device=device).reshape(3, 4)

    view = original.view(2, 6)
    reshaped = original.reshape(2, 6)
    cloned = original.clone()

    print("Original:\n", original)
    print("View:\n", view)
    print("Reshape:\n", reshaped)

    print("Original contiguous:", original.is_contiguous())
    print("Clone data pointer differs:",
          original.data_ptr() != cloned.data_ptr())

    # A view can share storage with its original tensor.
    view[0, 0] = 99
    print("Original after view modification:\n", original)

    # clone creates independent storage.
    cloned[0, 0] = -1
    print("Original after clone modification:\n", original)

    transposed = original.T
    print("Transposed contiguous:", transposed.is_contiguous())

    contiguous = transposed.contiguous()
    print("Contiguous copy:", contiguous.is_contiguous())


# ------------------------------------------------------------
# Device management
# ------------------------------------------------------------

def device_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("4. DEVICE MANAGEMENT")
    print("=" * 65)

    cpu_tensor = torch.randn(3, 3)

    selected_tensor = cpu_tensor.to(device)

    print("CPU device:", cpu_tensor.device)
    print("Selected device:", selected_tensor.device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
        print("CUDA memory allocated:",
              torch.cuda.memory_allocated(device))
        print("CUDA memory reserved:",
              torch.cuda.memory_reserved(device))

        # Synchronization is useful for accurate GPU timing.
        torch.cuda.synchronize(device)

    # Moving tensors between devices must be explicit.
    back_to_cpu = selected_tensor.cpu()
    print("Returned device:", back_to_cpu.device)


# ------------------------------------------------------------
# Autograd fundamentals
# ------------------------------------------------------------

def autograd_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("5. AUTOGRAD")
    print("=" * 65)

    x = torch.tensor(3.0, device=device, requires_grad=True)

    # y = x^2 + 2x + 1
    y = x**2 + 2 * x + 1

    y.backward()

    print("Input:", x.item())
    print("Output:", y.item())
    print("Gradient:", x.grad.item())

    # dy/dx = 2x + 2 = 8 when x = 3.
    assert math.isclose(x.grad.item(), 8.0, rel_tol=1e-5)

    # Gradients accumulate by default.
    y2 = x**3
    y2.backward()

    print("Accumulated gradient:", x.grad.item())

    # Clear gradients before independent computations.
    x.grad = None

    y3 = 4 * x
    y3.backward()

    print("Reset gradient:", x.grad.item())

    # A computation graph can be retained for repeated backward passes.
    z = x**2

    z.backward(retain_graph=True)
    first_gradient = x.grad.item()

    z.backward()
    second_gradient = x.grad.item()

    print("First backward gradient:", first_gradient)
    print("Accumulated backward gradient:", second_gradient)

    # no_grad disables gradient tracking for inference calculations.
    with torch.no_grad():
        inference_result = x * 10

    print("Inference result:", inference_result)
    print("Inference tracks gradients:", inference_result.requires_grad)

    # detach separates a tensor from its computation graph.
    detached = z.detach()
    print("Detached tensor:", detached)
    print("Detached tracks gradients:", detached.requires_grad)


# ------------------------------------------------------------
# Jacobians and gradient checking
# ------------------------------------------------------------

def gradient_check(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("6. GRADIENT VALIDATION")
    print("=" * 65)

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
        device=device,
        requires_grad=True
    )

    result = (x**3).sum()
    result.backward()

    expected = 3 * x.detach()**2

    print("Analytical gradient:", x.grad)
    print("Expected gradient:", expected)

    assert torch.allclose(x.grad, expected)

    # gradcheck compares analytical gradients with numerical estimates.
    check_input = torch.randn(
        3,
        dtype=torch.float64,
        device=device,
        requires_grad=True
    )

    passed = torch.autograd.gradcheck(
        lambda value: value.sin().sum(),
        (check_input,)
    )

    print("Gradient numerical validation:", passed)
    assert passed


# ------------------------------------------------------------
# Neural network and optimization
# ------------------------------------------------------------

class RegressionNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def training_example(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("7. NEURAL NETWORK TRAINING")
    print("=" * 65)

    # Synthetic regression dataset:
    # y = 3*x1 - 2*x2 + 1
    features = torch.randn(512, 2, device=device)

    targets = (
        3 * features[:, 0:1]
        - 2 * features[:, 1:2]
        + 1
    )

    model = RegressionNetwork(2, 32).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    batch_size = 64
    epochs = 80

    initial_loss = None

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(
            features.size(0),
            device=device
        )

        epoch_loss = 0.0

        for indices in permutation.split(batch_size):
            batch_x = features[indices]
            batch_y = targets[indices]

            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}"
                )

            loss.backward()

            # Gradient clipping limits unusually large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item() * len(indices)

        epoch_loss /= len(features)

        if initial_loss is None:
            initial_loss = epoch_loss

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{epochs} "
                f"Loss={epoch_loss:.6f}"
            )

    model.eval()

    with torch.no_grad():
        predictions = model(features)
        final_loss = criterion(predictions, targets).item()

    print("Initial epoch loss:", initial_loss)
    print("Final loss:", final_loss)

    assert final_loss < initial_loss

    # Inference using the trained model.
    sample = torch.tensor(
        [[0.5, 1.0]],
        device=device
    )

    with torch.inference_mode():
        prediction = model(sample)

    print("Sample input:", sample)
    print("Predicted output:", prediction)


# ------------------------------------------------------------
# Advanced tensor operations
# ------------------------------------------------------------

def advanced_operations(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("8. ADVANCED TENSOR OPERATIONS")
    print("=" * 65)

    # Broadcasting over a batch.
    batch = torch.randn(4, 3, 5, device=device)
    scale = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        device=device
    )

    scaled = batch * scale
    print("Batch shape:", batch.shape)
    print("Scaled shape:", scaled.shape)

    # Softmax converts scores into normalized probabilities.
    logits = torch.tensor(
        [[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]],
        device=device
    )

    probabilities = torch.softmax(logits, dim=1)

    print("Softmax:\n", probabilities)
    print("Probability sums:", probabilities.sum(dim=1))

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2, device=device)
    )

    # Gather selects values using indices.
    values = torch.tensor(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        device=device
    )

    indices = torch.tensor(
        [[2], [0]],
        device=device
    )

    print("Gather:\n", values.gather(1, indices))

    # Einsum performs a batch matrix multiplication.
    left = torch.randn(4, 2, 3, device=device)
    right = torch.randn(4, 3, 2, device=device)

    product = torch.einsum("bij,bjk->bik", left, right)

    print("Einsum result shape:", product.shape)


# ------------------------------------------------------------
# Error handling and performance
# ------------------------------------------------------------

def validation_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("9. VALIDATION AND FAILURE HANDLING")
    print("=" * 65)

    valid = torch.tensor([1.0, 2.0], device=device)
    validate_tensor(valid, "valid")

    invalid_examples = [
        ("empty", torch.tensor([], device=device)),
        ("nonfinite", torch.tensor([float("nan")], device=device)),
        ("integer", torch.tensor([1, 2], device=device)),
    ]

    for name, tensor in invalid_examples:
        try:
            validate_tensor(tensor, name)
        except (ValueError, TypeError) as error:
            print(f"Expected validation failure [{name}]: {error}")

    # Matrix multiplication requires compatible dimensions.
    try:
        torch.matmul(
            torch.randn(2, 3, device=device),
            torch.randn(4, 2, device=device)
        )
    except RuntimeError as error:
        print("Expected shape failure:", error)


def performance_examples(device: torch.device) -> None:
    print("\n" + "=" * 65)
    print("10. PERFORMANCE CONSIDERATIONS")
    print("=" * 65)

    size = 512

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    import time

    start = time.perf_counter()

    result = torch.matmul(a, b)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    print("Matrix multiplication shape:", result.shape)
    print(f"Execution time: {elapsed:.6f} seconds")

    # Avoid repeatedly transferring data between CPU and GPU.
    print("Execution device:", result.device)

    # Mixed precision is commonly useful for supported GPU workloads.
    if device.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            mixed_result = torch.matmul(a, b)

        print("Mixed precision dtype:", mixed_result.dtype)


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("PYTORCH FUNDAMENTALS")
    print("Tensors | Devices | Autograd | Tensor Operations")
    print("=" * 65)

    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)

    config = TensorConfig()
    device = config.resolve_device()

    print("Selected device:", device)

    tensor_examples(device)
    tensor_operations(device)
    tensor_memory_examples(device)
    device_examples(device)
    autograd_examples(device)
    gradient_check(device)
    training_example(device)
    advanced_operations(device)
    validation_examples(device)
    performance_examples(device)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
