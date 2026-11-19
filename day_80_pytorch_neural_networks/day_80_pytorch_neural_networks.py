#!/usr/bin/env python3
"""
PyTorch Neural Networks: Modules, Layers, Forward Methods, and Training Loops.

Demonstrates:
- Tensor operations and automatic differentiation
- nn.Module and custom neural network layers
- Forward propagation
- Loss functions and optimizers
- Training and validation loops
- Evaluation, checkpointing, and inference
- Reproducibility and numerical safeguards

Dependency: PyTorch
Run: python pytorch_neural_networks.py
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    input_features: int = 2
    hidden_features: int = 32
    output_features: int = 2
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    validation_fraction: float = 0.2
    seed: int = 42
    checkpoint_path: str = "neural_network_checkpoint.pt"

    def validate(self) -> None:
        if min(
            self.input_features,
            self.hidden_features,
            self.output_features,
            self.batch_size,
            self.epochs,
        ) <= 0:
            raise ValueError("Network dimensions and training values must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if not 0 < self.validation_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one.")


# ---------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# 3. Generate a classification dataset
# ---------------------------------------------------------

def generate_dataset(
    samples_per_class: int = 500,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:

    generator = torch.Generator().manual_seed(seed)

    class_zero = torch.randn(
        samples_per_class, 2, generator=generator
    ) * 0.65 + torch.tensor([-1.0, -1.0])

    class_one = torch.randn(
        samples_per_class, 2, generator=generator
    ) * 0.65 + torch.tensor([1.0, 1.0])

    features = torch.cat([class_zero, class_one], dim=0)

    labels = torch.cat([
        torch.zeros(samples_per_class, dtype=torch.long),
        torch.ones(samples_per_class, dtype=torch.long),
    ])

    permutation = torch.randperm(
        len(features), generator=generator
    )

    return features[permutation], labels[permutation]


def prepare_data(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:

    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional tensor.")

    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional tensor.")

    if len(features) != len(labels):
        raise ValueError("Feature and label counts must match.")

    if features.shape[1] != config.input_features:
        raise ValueError("Unexpected number of input features.")

    if not torch.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values.")

    if len(features) < 2:
        raise ValueError("At least two samples are required.")

    if labels.min() < 0 or labels.max() >= config.output_features:
        raise ValueError("Labels are outside the output class range.")

    validation_size = max(
        1,
        int(len(features) * config.validation_fraction),
    )

    training_size = len(features) - validation_size

    if training_size < 1:
        raise ValueError("Insufficient training samples.")

    dataset = TensorDataset(features.float(), labels.long())

    training_set, validation_set = torch.utils.data.random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    training_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return training_loader, validation_loader


# ---------------------------------------------------------
# 4. Custom neural network module
# ---------------------------------------------------------

class NeuralNetwork(nn.Module):
    """
    Fully connected feedforward neural network.

    Architecture:
        Input
          |
        Linear
          |
        ReLU
          |
        BatchNorm
          |
        Linear
          |
        ReLU
          |
        Dropout
          |
        Linear
          |
        Class logits
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.input_features,
                config.hidden_features,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_features),
            nn.Linear(
                config.hidden_features,
                config.hidden_features,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(
                config.hidden_features,
                config.output_features,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("Expected a batch of two-dimensional feature vectors.")

        return self.network(inputs)


# ---------------------------------------------------------
# 5. Model inspection
# ---------------------------------------------------------

def inspect_model(model: nn.Module) -> None:
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nMODEL ARCHITECTURE")
    print(model)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")


# ---------------------------------------------------------
# 6. Training one epoch
# ---------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward propagation.
        predictions = model(inputs)

        loss = loss_function(predictions, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss detected.")

        # Backpropagation.
        loss.backward()

        # Prevent unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        # Update parameters.
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predicted_classes = predictions.argmax(dim=1)

        correct += (
            predicted_classes == labels
        ).sum().item()

        total += batch_size

    return total_loss / total, correct / total


# ---------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = model(inputs)

        loss = loss_function(predictions, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss detected.")

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()

        total += batch_size

    return total_loss / total, correct / total


# ---------------------------------------------------------
# 8. Training orchestration and early stopping
# ---------------------------------------------------------

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=8,
        )

        self.history: list[dict[str, float]] = []

    def fit(
        self,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        patience: int = 20,
    ) -> list[dict[str, float]]:

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):

            training_loss, training_accuracy = train_one_epoch(
                self.model,
                training_loader,
                self.loss_function,
                self.optimizer,
                self.device,
            )

            validation_loss, validation_accuracy = evaluate(
                self.model,
                validation_loader,
                self.loss_function,
                self.device,
            )

            self.scheduler.step(validation_loss)

            metrics = {
                "epoch": float(epoch),
                "training_loss": training_loss,
                "training_accuracy": training_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }

            self.history.append(metrics)

            print(
                f"Epoch {epoch:03d}/{self.config.epochs} | "
                f"Train Loss: {training_loss:.4f} | "
                f"Train Accuracy: {training_accuracy:.4f} | "
                f"Validation Loss: {validation_loss:.4f} | "
                f"Validation Accuracy: {validation_accuracy:.4f}"
            )

            if validation_loss < best_loss:
                best_loss = validation_loss

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping activated.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history


# ---------------------------------------------------------
# 9. Model checkpointing
# ---------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> None:

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "pytorch_version": torch.__version__,
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, destination)

    print(f"\nCheckpoint saved: {destination}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> dict[str, Any]:

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return checkpoint


# ---------------------------------------------------------
# 10. Inference
# ---------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    if features.ndim == 1:
        features = features.unsqueeze(0)

    if features.ndim != 2:
        raise ValueError("Inference input must have two dimensions.")

    if not torch.isfinite(features).all():
        raise ValueError("Inference input contains invalid values.")

    model.eval()

    logits = model(features.float().to(device))

    probabilities = torch.softmax(logits, dim=1)

    predicted_classes = probabilities.argmax(dim=1)

    return predicted_classes.cpu(), probabilities.cpu()


# ---------------------------------------------------------
# 11. Main execution
# ---------------------------------------------------------

def main() -> None:

    config = TrainingConfig()
    config.validate()

    set_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"Execution device: {device}")

    features, labels = generate_dataset()

    training_loader, validation_loader = prepare_data(
        features,
        labels,
        config,
    )

    model = NeuralNetwork(config).to(device)

    inspect_model(model)

    trainer = Trainer(
        model,
        config,
        device,
    )

    trainer.fit(
        training_loader,
        validation_loader,
        patience=20,
    )

    save_checkpoint(
        model,
        config,
        trainer.optimizer,
        config.checkpoint_path,
    )

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_loader,
        trainer.loss_function,
        device,
    )

    print("\nFINAL VALIDATION")
    print(f"Loss: {validation_loss:.4f}")
    print(f"Accuracy: {validation_accuracy:.4f}")

    sample = torch.tensor(
        [[-1.2, -0.8], [1.1, 0.9], [0.0, 0.0]]
    )

    predicted, probabilities = predict(
        model,
        sample,
        device,
    )

    print("\nINFERENCE RESULTS")

    for index, prediction in enumerate(predicted):
        print(
            f"Sample {index + 1}: "
            f"Class={prediction.item()}, "
            f"Probabilities={probabilities[index].tolist()}"
        )

    # Confirm checkpoint loading.
    restored_model = NeuralNetwork(config).to(device)

    load_checkpoint(
        restored_model,
        config.checkpoint_path,
        device,
    )

    restored_predictions, _ = predict(
        restored_model,
        sample,
        device,
    )

    assert torch.equal(predicted, restored_predictions)

    print("\nCheckpoint restoration verified.")
    print("Training and inference completed successfully.")


if __name__ == "__main__":
    main()
