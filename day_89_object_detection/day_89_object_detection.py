#!/usr/bin/env python3
"""
Object Detection Pipeline
Bounding Boxes, IoU, Non-Maximum Suppression (NMS), and Detection Evaluation.

Python 3.10+
Standard library only.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional


class DetectionError(Exception):
    """Raised when detection data or pipeline configuration is invalid."""


class BoxFormat(Enum):
    XYXY = "xyxy"
    XYWH = "xywh"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        values = (self.x1, self.y1, self.x2, self.y2)

        if not all(math.isfinite(v) for v in values):
            raise DetectionError("Bounding box coordinates must be finite.")

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise DetectionError("Bounding box must have positive dimensions.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )

    def clip(self, image_width: int, image_height: int) -> Optional["BoundingBox"]:
        if image_width <= 0 or image_height <= 0:
            raise DetectionError("Image dimensions must be positive.")

        x1 = max(0.0, min(float(image_width), self.x1))
        y1 = max(0.0, min(float(image_height), self.y1))
        x2 = max(0.0, min(float(image_width), self.x2))
        y2 = max(0.0, min(float(image_height), self.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return BoundingBox(x1, y1, x2, y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    @classmethod
    def from_xywh(
        cls, x: float, y: float, width: float, height: float
    ) -> "BoundingBox":
        return cls(x, y, x + width, y + height)


@dataclass(frozen=True)
class Detection:
    box: BoundingBox
    confidence: float
    class_id: int
    label: str
    image_id: str

    def __post_init__(self):
        if not math.isfinite(self.confidence):
            raise DetectionError("Confidence must be finite.")

        if not 0 <= self.confidence <= 1:
            raise DetectionError("Confidence must be between 0 and 1.")

        if self.class_id < 0:
            raise DetectionError("Class ID cannot be negative.")

        if not self.label.strip():
            raise DetectionError("Class label cannot be empty.")

        if not self.image_id.strip():
            raise DetectionError("Image ID cannot be empty.")


@dataclass
class PipelineConfig:
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    class_agnostic_nms: bool = False

    def __post_init__(self):
        if not 0 <= self.confidence_threshold <= 1:
            raise DetectionError("Invalid confidence threshold.")

        if not 0 <= self.iou_threshold <= 1:
            raise DetectionError("Invalid IoU threshold.")

        if self.max_detections < 1:
            raise DetectionError("Maximum detections must be positive.")


def intersection_over_union(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:
    """Calculate IoU using continuous coordinate geometry."""

    intersection_x1 = max(box_a.x1, box_b.x1)
    intersection_y1 = max(box_a.y1, box_b.y1)
    intersection_x2 = min(box_a.x2, box_b.x2)
    intersection_y2 = min(box_a.y2, box_b.y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    union_area = box_a.area + box_b.area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def generalized_iou(
    box_a: BoundingBox,
    box_b: BoundingBox,
) -> float:
    """Calculate Generalized IoU for bounding-box quality analysis."""

    iou = intersection_over_union(box_a, box_b)

    enclosing_x1 = min(box_a.x1, box_b.x1)
    enclosing_y1 = min(box_a.y1, box_b.y1)
    enclosing_x2 = max(box_a.x2, box_b.x2)
    enclosing_y2 = max(box_a.y2, box_b.y2)

    enclosing_area = (
        (enclosing_x2 - enclosing_x1)
        * (enclosing_y2 - enclosing_y1)
    )

    union_area = box_a.area + box_b.area

    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union_area -= intersection

    if enclosing_area <= 0:
        return iou

    return iou - (enclosing_area - union_area) / enclosing_area


def confidence_filter(
    detections: Iterable[Detection],
    threshold: float,
) -> list[Detection]:
    if not 0 <= threshold <= 1:
        raise DetectionError("Confidence threshold must be in [0, 1].")

    return [
        detection
        for detection in detections
        if detection.confidence >= threshold
    ]


def non_maximum_suppression(
    detections: Iterable[Detection],
    iou_threshold: float = 0.45,
    class_agnostic: bool = False,
    max_detections: int = 100,
) -> list[Detection]:
    """
    Greedy NMS.

    Sort by descending confidence. Retain the highest-confidence detection
    and suppress overlapping detections above the IoU threshold.

    Complexity: O(n^2) in the worst case.
    """

    if not 0 <= iou_threshold <= 1:
        raise DetectionError("IoU threshold must be in [0, 1].")

    if max_detections < 1:
        raise DetectionError("max_detections must be positive.")

    ordered = sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )

    selected: list[Detection] = []

    while ordered and len(selected) < max_detections:
        current = ordered.pop(0)
        selected.append(current)

        remaining = []

        for candidate in ordered:
            if not class_agnostic and (
                candidate.class_id != current.class_id
            ):
                remaining.append(candidate)
                continue

            overlap = intersection_over_union(
                current.box,
                candidate.box,
            )

            # Suppress only when overlap is strictly greater than threshold.
            if overlap <= iou_threshold:
                remaining.append(candidate)

        ordered = remaining

    return selected


def class_aware_nms(
    detections: Iterable[Detection],
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    """Run NMS independently for each class."""

    groups: dict[int, list[Detection]] = {}

    for detection in detections:
        groups.setdefault(detection.class_id, []).append(detection)

    results = []

    for class_detections in groups.values():
        results.extend(
            non_maximum_suppression(
                class_detections,
                iou_threshold,
                False,
                max_detections,
            )
        )

    return sorted(
        results,
        key=lambda detection: detection.confidence,
        reverse=True,
    )[:max_detections]


def soft_nms(
    detections: Iterable[Detection],
    iou_threshold: float = 0.45,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> list[Detection]:
    """
    Gaussian Soft-NMS.

    Instead of removing overlapping boxes, reduce their confidence scores.
    Original Detection instances remain immutable.
    """

    if not 0 <= iou_threshold <= 1:
        raise DetectionError("Invalid IoU threshold.")

    if sigma <= 0 or not math.isfinite(sigma):
        raise DetectionError("Sigma must be positive and finite.")

    working = list(detections)
    scores = {id(d): d.confidence for d in working}
    selected = []

    while working:
        working.sort(
            key=lambda d: scores[id(d)],
            reverse=True,
        )

        best = working.pop(0)

        if scores[id(best)] < score_threshold:
            break

        selected.append(
            Detection(
                best.box,
                scores[id(best)],
                best.class_id,
                best.label,
                best.image_id,
            )
        )

        remaining = []

        for candidate in working:
            if candidate.class_id != best.class_id:
                remaining.append(candidate)
                continue

            overlap = intersection_over_union(
                best.box,
                candidate.box,
            )

            if overlap > iou_threshold:
                scores[id(candidate)] *= math.exp(
                    -(overlap * overlap) / sigma
                )

            if scores[id(candidate)] >= score_threshold:
                remaining.append(candidate)

        working = remaining

    return selected


class DetectionPipeline:
    """Coordinates preprocessing, filtering, suppression and reporting."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = {
            "processed_images": 0,
            "input_detections": 0,
            "filtered_detections": 0,
            "output_detections": 0,
            "elapsed_seconds": [],
        }

    def process(
        self,
        image_id: str,
        detections: Iterable[Detection],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:

        if image_width <= 0 or image_height <= 0:
            raise DetectionError("Invalid image dimensions.")

        if not image_id.strip():
            raise DetectionError("Image ID is required.")

        start = time.perf_counter()

        raw = list(detections)

        if any(d.image_id != image_id for d in raw):
            raise DetectionError("Mixed image IDs in one pipeline batch.")

        self.metrics["input_detections"] += len(raw)

        # Remove boxes outside the image and clip partially visible boxes.
        valid = []

        for detection in raw:
            clipped = detection.box.clip(image_width, image_height)

            if clipped is None:
                continue

            valid.append(
                Detection(
                    clipped,
                    detection.confidence,
                    detection.class_id,
                    detection.label,
                    detection.image_id,
                )
            )

        filtered = confidence_filter(
            valid,
            self.config.confidence_threshold,
        )

        self.metrics["filtered_detections"] += len(filtered)

        if self.config.class_agnostic_nms:
            results = non_maximum_suppression(
                filtered,
                self.config.iou_threshold,
                True,
                self.config.max_detections,
            )
        else:
            results = class_aware_nms(
                filtered,
                self.config.iou_threshold,
                self.config.max_detections,
            )

        self.metrics["output_detections"] += len(results)
        self.metrics["processed_images"] += 1
        self.metrics["elapsed_seconds"].append(
            time.perf_counter() - start
        )

        return results

    def report(self) -> dict:
        timings = self.metrics["elapsed_seconds"]

        return {
            "processed_images": self.metrics["processed_images"],
            "input_detections": self.metrics["input_detections"],
            "filtered_detections": self.metrics["filtered_detections"],
            "output_detections": self.metrics["output_detections"],
            "average_pipeline_seconds": (
                statistics.mean(timings) if timings else 0.0
            ),
        }


def calculate_detection_metrics(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Calculate single-threshold precision, recall and F1.

    Matching is performed independently per class and image.
    Each ground-truth object can match at most one prediction.
    """

    if not 0 <= iou_threshold <= 1:
        raise DetectionError("Invalid evaluation IoU threshold.")

    predictions = sorted(
        predictions,
        key=lambda d: d.confidence,
        reverse=True,
    )

    ground_truth = list(ground_truth)

    matched = set()
    true_positives = 0
    false_positives = 0

    for prediction in predictions:
        candidates = []

        for index, target in enumerate(ground_truth):
            if index in matched:
                continue

            if (
                target.image_id != prediction.image_id
                or target.class_id != prediction.class_id
            ):
                continue

            overlap = intersection_over_union(
                prediction.box,
                target.box,
            )

            if overlap >= iou_threshold:
                candidates.append((overlap, index))

        if candidates:
            _, best_index = max(candidates)
            matched.add(best_index)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def average_precision(
    predictions: Iterable[Detection],
    ground_truth: Iterable[Detection],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate interpolated AP for a single class.

    Predictions and ground truth must belong to the same class.
    Uses the precision envelope and recall-change integration.
    """

    predictions = sorted(
        predictions,
        key=lambda d: d.confidence,
        reverse=True,
    )
    ground_truth = list(ground_truth)

    if not predictions and not ground_truth:
        return 0.0

    if not ground_truth:
        return 0.0

    target_count = len(ground_truth)
    matched = set()

    true_positive_flags = []
    false_positive_flags = []

    for prediction in predictions:
        candidates = []

        for index, target in enumerate(ground_truth):
            if index in matched:
                continue

            if prediction.image_id != target.image_id:
                continue

            overlap = intersection_over_union(
                prediction.box,
                target.box,
            )

            if overlap >= iou_threshold:
                candidates.append((overlap, index))

        if candidates:
            _, index = max(candidates)
            matched.add(index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)
        else:
            true_positive_flags.append(0)
            false_positive_flags.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = [0.0]
    precisions = [0.0]

    for tp, fp in zip(true_positive_flags, false_positive_flags):
        cumulative_tp += tp
        cumulative_fp += fp

        recalls.append(cumulative_tp / target_count)
        precisions.append(
            cumulative_tp / (cumulative_tp + cumulative_fp)
        )

    recalls.append(1.0)
    precisions.append(0.0)

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(
            precisions[index],
            precisions[index + 1],
        )

    ap = 0.0

    for index in range(1, len(recalls)):
        ap += (
            recalls[index] - recalls[index - 1]
        ) * precisions[index]

    return ap


def save_results(
    detections: Iterable[Detection],
    destination: str,
) -> None:
    """Write validated detection results as JSON."""

    path = Path(destination)

    records = []

    for detection in detections:
        records.append({
            "image_id": detection.image_id,
            "class_id": detection.class_id,
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox_xyxy": [
                detection.box.x1,
                detection.box.y1,
                detection.box.x2,
                detection.box.y2,
            ],
        })

    # Prevent accidental path traversal and write to a local destination.
    if path.name in ("", ".", ".."):
        raise DetectionError("Invalid output filename.")

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, allow_nan=False)


def run_tests() -> None:
    a = BoundingBox(10, 10, 110, 110)
    b = BoundingBox(60, 60, 160, 160)

    expected = 2500 / 17500
    assert abs(intersection_over_union(a, b) - expected) < 1e-9

    assert intersection_over_union(
        BoundingBox(0, 0, 10, 10),
        BoundingBox(10, 0, 20, 10),
    ) == 0.0

    assert a.area == 10000

    try:
        BoundingBox(10, 10, 10, 20)
        raise AssertionError("Invalid box was accepted.")
    except DetectionError:
        pass

    print("All unit assertions passed.")


def main() -> None:
    run_tests()

    config = PipelineConfig(
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100,
        class_agnostic_nms=False,
    )

    pipeline = DetectionPipeline(config)

    predictions = [
        Detection(
            BoundingBox(50, 40, 200, 220),
            0.96,
            0,
            "person",
            "image-001",
        ),
        Detection(
            BoundingBox(55, 45, 198, 218),
            0.89,
            0,
            "person",
            "image-001",
        ),
        Detection(
            BoundingBox(300, 100, 430, 260),
            0.91,
            2,
            "car",
            "image-001",
        ),
        Detection(
            BoundingBox(500, 50, 580, 130),
            0.18,
            0,
            "person",
            "image-001",
        ),
        Detection(
            BoundingBox(610, 100, 710, 220),
            0.82,
            0,
            "person",
            "image-001",
        ),
    ]

    results = pipeline.process(
        "image-001",
        predictions,
        image_width=640,
        image_height=480,
    )

    ground_truth = [
        Detection(
            BoundingBox(52, 42, 202, 222),
            1.0,
            0,
            "person",
            "image-001",
        ),
        Detection(
            BoundingBox(305, 105, 435, 265),
            1.0,
            2,
            "car",
            "image-001",
        ),
    ]

    print("\nFINAL DETECTIONS")

    for detection in results:
        print(
            f"{detection.label:10s} "
            f"confidence={detection.confidence:.3f} "
            f"box={detection.box}"
        )

    print("\nEVALUATION")
    print(json.dumps(
        calculate_detection_metrics(results, ground_truth),
        indent=2,
    ))

    print("\nPIPELINE REPORT")
    print(json.dumps(pipeline.report(), indent=2))

    save_results(results, "detections.json")
    print("\nResults saved to detections.json")


if __name__ == "__main__":
    main()
