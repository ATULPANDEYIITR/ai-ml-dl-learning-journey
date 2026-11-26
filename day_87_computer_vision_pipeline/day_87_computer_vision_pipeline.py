#!/usr/bin/env python3
"""
Computer Vision Pipeline
Image preprocessing, data augmentation, feature extraction, and classification.

Supported input: P3 ASCII PPM images.
Dependencies: Python 3.10+ standard library only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("computer_vision")


class PipelineError(Exception):
    """Base exception for pipeline failures."""


class ImageFormatError(PipelineError):
    """Raised when an image is invalid or unsupported."""


class ValidationError(PipelineError):
    """Raised when pipeline configuration or input is invalid."""


class ImageClass(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class Image:
    width: int
    height: int
    pixels: list[list[tuple[int, int, int]]]
    max_value: int = 255

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValidationError("Image dimensions must be positive.")

        if len(self.pixels) != self.height:
            raise ValidationError("Image height does not match pixel data.")

        for row in self.pixels:
            if len(row) != self.width:
                raise ValidationError("Image width does not match pixel data.")

            for pixel in row:
                if len(pixel) != 3:
                    raise ValidationError("Every pixel must have three channels.")
                if any(
                    not isinstance(channel, int)
                    or channel < 0
                    or channel > self.max_value
                    for channel in pixel
                ):
                    raise ValidationError("Invalid pixel channel value.")

        if not 1 <= self.max_value <= 65535:
            raise ValidationError("Invalid maximum channel value.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, 3


@dataclass(frozen=True)
class PipelineConfig:
    target_width: int = 32
    target_height: int = 32
    normalization: str = "zero_one"
    random_seed: int = 42
    horizontal_flip: bool = True
    rotation: bool = True

    def validate(self) -> None:
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValidationError("Target dimensions must be positive.")

        if self.normalization not in {"zero_one", "minus_one_one"}:
            raise ValidationError("Unsupported normalization method.")


@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    sample_count: int


class PPMReader:
    """Reads validated P3 ASCII Portable Pixmap images."""

    @staticmethod
    def read(path: str | Path) -> Image:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        tokens = []

        with path.open("r", encoding="ascii") as file:
            for line in file:
                line = line.split("#", 1)[0]
                tokens.extend(line.split())

        if len(tokens) < 4 or tokens[0] != "P3":
            raise ImageFormatError("Only P3 ASCII PPM images are supported.")

        try:
            width = int(tokens[1])
            height = int(tokens[2])
            max_value = int(tokens[3])

            if width <= 0 or height <= 0:
                raise ValueError("Invalid dimensions.")

            if not 1 <= max_value <= 65535:
                raise ValueError("Invalid maximum channel value.")

            expected = width * height * 3

            if expected > 30_000_000:
                raise ValidationError("Image exceeds the pixel limit.")

            values = [int(value) for value in tokens[4:]]

            if len(values) != expected:
                raise ImageFormatError(
                    f"Expected {expected} channels, received {len(values)}."
                )

            if any(value < 0 or value > max_value for value in values):
                raise ImageFormatError("Pixel value outside permitted range.")

            pixels = []

            for y in range(height):
                row = []

                for x in range(width):
                    index = (y * width + x) * 3
                    row.append(tuple(values[index:index + 3]))

                pixels.append(row)

            image = Image(width, height, pixels, max_value)
            image.validate()
            return image

        except (ValueError, OverflowError) as exc:
            raise ImageFormatError(f"Invalid PPM image: {exc}") from exc

    @staticmethod
    def write(image: Image, path: str | Path) -> None:
        image.validate()

        with Path(path).open("w", encoding="ascii") as file:
            file.write(
                f"P3\n{image.width} {image.height}\n{image.max_value}\n"
            )

            for row in image.pixels:
                file.write(
                    " ".join(
                        str(channel)
                        for pixel in row
                        for channel in pixel
                    )
                    + "\n"
                )


class ImagePreprocessor:
    """Performs resizing, scaling, and normalization."""

    @staticmethod
    def resize(
        image: Image,
        target_width: int,
        target_height: int,
    ) -> Image:
        image.validate()

        if target_width <= 0 or target_height <= 0:
            raise ValidationError("Invalid resize dimensions.")

        resized = []

        for y in range(target_height):
            source_y = min(
                image.height - 1,
                int(y * image.height / target_height),
            )

            row = []

            for x in range(target_width):
                source_x = min(
                    image.width - 1,
                    int(x * image.width / target_width),
                )
                row.append(image.pixels[source_y][source_x])

            resized.append(row)

        return Image(
            target_width,
            target_height,
            resized,
            image.max_value,
        )

    @staticmethod
    def normalize(
        image: Image,
        method: str = "zero_one",
    ) -> list[list[tuple[float, float, float]]]:
        image.validate()

        if method not in {"zero_one", "minus_one_one"}:
            raise ValidationError("Unsupported normalization method.")

        result = []

        for row in image.pixels:
            normalized_row = []

            for pixel in row:
                channels = [
                    channel / image.max_value
                    for channel in pixel
                ]

                if method == "minus_one_one":
                    channels = [
                        value * 2.0 - 1.0
                        for value in channels
                    ]

                normalized_row.append(tuple(channels))

            result.append(normalized_row)

        return result

    @staticmethod
    def grayscale(image: Image) -> list[list[float]]:
        """Converts RGB pixels using perceptual luminance weights."""

        return [
            [
                (
                    0.2126 * pixel[0]
                    + 0.7152 * pixel[1]
                    + 0.0722 * pixel[2]
                ) / image.max_value
                for pixel in row
            ]
            for row in image.pixels
        ]

    @staticmethod
    def standardize(
        values: Sequence[float],
    ) -> list[float]:
        if not values:
            raise ValidationError("Cannot standardize empty data.")

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2 for value in values
        ) / len(values)

        standard_deviation = math.sqrt(variance)

        if standard_deviation == 0:
            return [0.0 for _ in values]

        return [
            (value - mean) / standard_deviation
            for value in values
        ]


class ImageAugmentor:
    """Creates reproducible transformations without modifying source images."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    @staticmethod
    def horizontal_flip(image: Image) -> Image:
        pixels = [
            list(reversed(row))
            for row in image.pixels
        ]

        return Image(
            image.width,
            image.height,
            pixels,
            image.max_value,
        )

    @staticmethod
    def vertical_flip(image: Image) -> Image:
        return Image(
            image.width,
            image.height,
            list(reversed(image.pixels)),
            image.max_value,
        )

    @staticmethod
    def rotate_90(image: Image) -> Image:
        """Rotates clockwise and exchanges image dimensions."""

        pixels = [
            [
                image.pixels[image.height - 1 - y][x]
                for y in range(image.height)
            ]
            for x in range(image.width)
        ]

        return Image(
            image.height,
            image.width,
            pixels,
            image.max_value,
        )

    def brightness(
        self,
        image: Image,
        factor: float,
    ) -> Image:
        if not 0 <= factor <= 3:
            raise ValidationError("Brightness factor must be between 0 and 3.")

        pixels = [
            [
                tuple(
                    min(
                        image.max_value,
                        max(0, round(channel * factor)),
                    )
                    for channel in pixel
                )
                for pixel in row
            ]
            for row in image.pixels
        ]

        return Image(
            image.width,
            image.height,
            pixels,
            image.max_value,
        )

    def random_augmentation(self, image: Image) -> Image:
        result = image

        if self.random.choice([True, False]):
            result = self.horizontal_flip(result)

        if self.random.choice([True, False]):
            result = self.vertical_flip(result)

        result = self.brightness(
            result,
            self.random.uniform(0.8, 1.2),
        )

        return result


class FeatureExtractor:
    """Extracts normalized RGB statistics and color proportions."""

    @staticmethod
    def extract(image: Image) -> dict[str, float]:
        image.validate()

        total = image.width * image.height

        channels = [
            [pixel[channel] for row in image.pixels for pixel in row]
            for channel in range(3)
        ]

        means = [
            sum(channel) / (total * image.max_value)
            for channel in channels
        ]

        variances = [
            sum(
                (
                    value / image.max_value - means[index]
                ) ** 2
                for value in channel
            ) / total
            for index, channel in enumerate(channels)
        ]

        return {
            "red_mean": means[0],
            "green_mean": means[1],
            "blue_mean": means[2],
            "red_std": math.sqrt(variances[0]),
            "green_std": math.sqrt(variances[1]),
            "blue_std": math.sqrt(variances[2]),
            "red_ratio": means[0] / max(sum(means), 1e-12),
            "green_ratio": means[1] / max(sum(means), 1e-12),
            "blue_ratio": means[2] / max(sum(means), 1e-12),
        }


class NearestCentroidClassifier:
    """
    A transparent baseline classifier using Euclidean distance.

    It is useful for pipeline demonstrations but is not a replacement
    for a trained convolutional neural network.
    """

    def __init__(self):
        self.centroids: dict[str, list[float]] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        samples: Iterable[tuple[dict[str, float], str]],
    ) -> None:
        grouped = defaultdict(list)
        self.feature_names = []

        for features, label in samples:
            if not label:
                raise ValidationError("Class label cannot be empty.")

            if not features:
                raise ValidationError("Feature vector cannot be empty.")

            if not self.feature_names:
                self.feature_names = sorted(features)

            if sorted(features) != self.feature_names:
                raise ValidationError("Inconsistent feature dimensions.")

            vector = [features[name] for name in self.feature_names]

            if not all(math.isfinite(value) for value in vector):
                raise ValidationError("Features must be finite.")

            grouped[label].append(vector)

        if not grouped:
            raise ValidationError("Training dataset is empty.")

        self.centroids = {}

        for label, vectors in grouped.items():
            self.centroids[label] = [
                sum(vector[i] for vector in vectors) / len(vectors)
                for i in range(len(self.feature_names))
            ]

        logger.info(
            "Trained classifier with %d classes.",
            len(self.centroids),
        )

    def predict(
        self,
        features: dict[str, float],
    ) -> ClassificationResult:
        if not self.centroids:
            raise PipelineError("Classifier has not been trained.")

        if sorted(features) != self.feature_names:
            raise ValidationError("Prediction feature dimensions do not match.")

        vector = [features[name] for name in self.feature_names]

        if not all(math.isfinite(value) for value in vector):
            raise ValidationError("Prediction contains non-finite values.")

        distances = {}

        for label, centroid in self.centroids.items():
            distances[label] = math.sqrt(
                sum(
                    (vector[i] - centroid[i]) ** 2
                    for i in range(len(vector))
                )
            )

        # Softmax over negative distances produces comparable scores.
        minimum = min(distances.values())

        weights = {
            label: math.exp(-(distance - minimum))
            for label, distance in distances.items()
        }

        denominator = sum(weights.values())

        probabilities = {
            label: weight / denominator
            for label, weight in weights.items()
        }

        predicted = max(
            probabilities,
            key=probabilities.get,
        )

        return ClassificationResult(
            predicted_class=predicted,
            confidence=probabilities[predicted],
            probabilities=probabilities,
            sample_count=len(self.centroids),
        )

    def save(self, path: str | Path) -> None:
        if not self.centroids:
            raise PipelineError("Cannot save an untrained classifier.")

        payload = {
            "feature_names": self.feature_names,
            "centroids": self.centroids,
        }

        Path(path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        names = payload.get("feature_names")
        centroids = payload.get("centroids")

        if (
            not isinstance(names, list)
            or not names
            or not isinstance(centroids, dict)
            or not centroids
        ):
            raise ValidationError("Invalid classifier model file.")

        if any(
            not isinstance(vector, list)
            or len(vector) != len(names)
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in vector
            )
            for vector in centroids.values()
        ):
            raise ValidationError("Invalid centroid data.")

        self.feature_names = names
        self.centroids = centroids


class ComputerVisionPipeline:
    """Coordinates preprocessing, augmentation, and classification."""

    def __init__(self, config: PipelineConfig):
        config.validate()

        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.augmentor = ImageAugmentor(config.random_seed)
        self.extractor = FeatureExtractor()
        self.classifier = NearestCentroidClassifier()

    def prepare(self, image: Image) -> Image:
        return self.preprocessor.resize(
            image,
            self.config.target_width,
            self.config.target_height,
        )

    def training_samples(
        self,
        dataset: Iterable[tuple[Image, str]],
        augment: bool = True,
    ) -> list[tuple[dict[str, float], str]]:
        samples = []

        for image, label in dataset:
            if not label.strip():
                raise ValidationError("Dataset contains an empty label.")

            processed = self.prepare(image)

            samples.append(
                (self.extractor.extract(processed), label)
            )

            if augment:
                flipped = self.augmentor.horizontal_flip(processed)
                samples.append(
                    (self.extractor.extract(flipped), label)
                )

                bright = self.augmentor.brightness(
                    processed,
                    1.1,
                )
                samples.append(
                    (self.extractor.extract(bright), label)
                )

        return samples

    def train(
        self,
        dataset: Iterable[tuple[Image, str]],
        augment: bool = True,
    ) -> None:
        samples = self.training_samples(dataset, augment)
        self.classifier.fit(samples)

    def predict(self, image: Image) -> ClassificationResult:
        processed = self.prepare(image)
        features = self.extractor.extract(processed)
        return self.classifier.predict(features)


def create_synthetic_image(
    color: tuple[int, int, int],
    width: int = 16,
    height: int = 16,
) -> Image:
    """Creates deterministic sample images for executable demonstrations."""

    pixels = []

    for y in range(height):
        row = []

        for x in range(width):
            variation = (x + y) % 12

            pixel = tuple(
                min(255, channel + variation)
                for channel in color
            )

            row.append(pixel)

        pixels.append(row)

    return Image(width, height, pixels)


def run_demo() -> None:
    config = PipelineConfig(
        target_width=16,
        target_height=16,
        random_seed=42,
    )

    pipeline = ComputerVisionPipeline(config)

    training_data = [
        (create_synthetic_image((230, 20, 20)), "red"),
        (create_synthetic_image((210, 30, 25)), "red"),
        (create_synthetic_image((20, 220, 20)), "green"),
        (create_synthetic_image((25, 200, 30)), "green"),
        (create_synthetic_image((20, 20, 230)), "blue"),
        (create_synthetic_image((25, 30, 210)), "blue"),
    ]

    pipeline.train(training_data, augment=True)

    test_images = {
        "red": create_synthetic_image((225, 25, 25)),
        "green": create_synthetic_image((25, 215, 25)),
        "blue": create_synthetic_image((25, 25, 220)),
    }

    print("\nCOMPUTER VISION CLASSIFICATION RESULTS")

    for expected, image in test_images.items():
        result = pipeline.predict(image)

        print(f"\nExpected: {expected}")
        print(f"Predicted: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.4f}")
        print(
            "Probabilities:",
            json.dumps(result.probabilities, indent=2),
        )

        assert result.predicted_class == expected

    model_path = Path("vision_centroids.json")
    pipeline.classifier.save(model_path)

    restored = NearestCentroidClassifier()
    restored.load(model_path)

    print("\nModel persistence test:", "PASSED")

    # Demonstrate transformations.
    original = create_synthetic_image((200, 80, 40))
    resized = pipeline.prepare(original)
    flipped = pipeline.augmentor.horizontal_flip(resized)
    rotated = pipeline.augmentor.rotate_90(resized)

    assert flipped.shape == resized.shape
    assert rotated.width == resized.height
    assert rotated.height == resized.width

    print("Preprocessing and augmentation tests: PASSED")
    print("Model saved:", model_path.resolve())


def process_image_file(
    image_path: str,
    model_path: str,
    output_path: str | None = None,
) -> None:
    image = PPMReader.read(image_path)

    classifier = NearestCentroidClassifier()
    classifier.load(model_path)

    config = PipelineConfig()
    pipeline = ComputerVisionPipeline(config)
    pipeline.classifier = classifier

    result = pipeline.predict(image)

    print(json.dumps(asdict(result), indent=2))

    if output_path:
        processed = pipeline.prepare(image)
        PPMReader.write(processed, output_path)
        logger.info("Processed image saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computer vision preprocessing and classification pipeline."
    )

    parser.add_argument(
        "--image",
        help="Input P3 PPM image for classification.",
    )

    parser.add_argument(
        "--model",
        default="vision_centroids.json",
        help="Saved classifier model.",
    )

    parser.add_argument(
        "--output",
        help="Optional path for resized PPM image.",
    )

    args = parser.parse_args()

    try:
        if args.image:
            process_image_file(
                args.image,
                args.model,
                args.output,
            )
        else:
            run_demo()

    except (PipelineError, OSError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
