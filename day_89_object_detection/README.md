# Object Detection: Bounding Boxes, IoU, NMS and Detection Pipelines

## 1. Object Detection Architecture

Object detection is a computer vision task that identifies objects in an image and estimates their spatial locations.

An object detector produces a collection of predictions. Each prediction contains:

- An object category.
- A confidence score.
- A bounding box.
- An image identifier.

A typical detection pipeline follows this sequence:

    Input Image
         |
         v
    Image Preprocessing
         |
         v
    Detection Model
         |
         v
    Raw Predictions
         |
         v
    Bounding Box Validation
         |
         v
    Confidence Filtering
         |
         v
    Non-Maximum Suppression
         |
         v
    Final Detections
         |
         v
    Evaluation and Reporting

The implementations cover geometric calculations, prediction filtering, duplicate suppression, detection evaluation, pipeline management and database persistence.

## 2. Bounding Boxes

A bounding box represents the rectangular region containing a detected object.

The implementations use the continuous-coordinate representation:

| Coordinate | Meaning |
|---|---|
| x1 | Left boundary |
| y1 | Top boundary |
| x2 | Right boundary |
| y2 | Bottom boundary |

The dimensions are:

- Width = x2 - x1
- Height = y2 - y1
- Area = Width × Height

The center coordinates are:

- Center X = (x1 + x2) / 2
- Center Y = (y1 + y2) / 2

### 2.1 Coordinate representations

Two common representations are:

| Format | Parameters |
|---|---|
| XYXY | x1, y1, x2, y2 |
| XYWH | x, y, width, height |

XYXY represents the two corners of a rectangle. XYWH represents its upper-left corner and dimensions.

Conversion between them is important when integrating detection models, annotation systems and visualization applications.

### 2.2 Bounding-box validation

A valid bounding box must satisfy:

- Every coordinate is finite.
- x2 is greater than x1.
- y2 is greater than y1.
- Width and height are positive.

A box can also extend beyond image boundaries.

Clipping restricts its coordinates to the image dimensions:

    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

A box with no remaining visible area is discarded.

The Python, JavaScript, C++17 and Java implementations perform this validation before suppression.

The PostgreSQL implementation enforces image-boundary constraints through triggers.

## 3. Intersection over Union

Intersection over Union (IoU) measures the spatial overlap between two bounding boxes.

It is used to determine whether two predictions represent substantially overlapping regions.

The mathematical definition is:

IoU = Intersection Area / Union Area

Where:

- Intersection Area represents the common region.
- Union Area represents the combined region without double-counting the intersection.

The calculation is:

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)

    intersection_area =
        intersection_width * intersection_height

    union_area =
        area_a + area_b - intersection_area

    iou = intersection_area / union_area

### 3.1 IoU interpretation

| IoU | Interpretation |
|---:|---|
| 0.00 | No overlap |
| 0.25 | Limited overlap |
| 0.50 | Moderate overlap |
| 0.75 | Strong overlap |
| 1.00 | Identical boxes |

The interpretation depends on the evaluation or suppression threshold.

IoU is not a confidence score. It measures geometric overlap rather than the probability that an object exists.

### 3.2 Edge cases

The implementations account for:

- Identical bounding boxes.
- Completely disjoint boxes.
- Boxes touching at their boundaries.
- Invalid dimensions.
- Non-finite coordinates.
- Very small intersections.

Two boxes sharing only an edge have an intersection area of zero under the continuous-coordinate convention used here.

## 4. Generalized Intersection over Union

Generalized IoU (GIoU) extends IoU by considering the smallest enclosing rectangle.

Its definition is:

GIoU = IoU - (C - U) / C

Where:

- C is the area of the smallest enclosing rectangle.
- U is the union area of the two boxes.

Unlike ordinary IoU, GIoU can become negative when boxes are separated.

This provides additional information about geometric separation.

GIoU is particularly useful as a bounding-box regression loss component during model training.

The four implementations calculate GIoU independently of their detection-suppression mechanisms.

## 5. Confidence Scores and Filtering

A detection confidence score represents the detector's confidence in a predicted object.

Confidence filtering removes predictions below a configured threshold.

For example:

    confidence_threshold = 0.25

A prediction with confidence 0.18 is rejected, while predictions with confidence 0.25 or higher remain candidates.

The threshold controls the trade-off between:

- Retaining potentially valid objects.
- Reducing false-positive predictions.
- Reducing the number of boxes requiring suppression.

A high confidence threshold can improve precision while reducing recall.

A low threshold can preserve more potential detections but increase computational requirements and false positives.

Confidence scores should not automatically be interpreted as perfectly calibrated probabilities.

## 6. Non-Maximum Suppression

Non-Maximum Suppression (NMS) removes redundant bounding-box predictions.

Detection models can produce multiple boxes around the same object. NMS retains a higher-confidence candidate and suppresses excessive overlaps.

### 6.1 Greedy NMS algorithm

The implemented algorithm follows these operations:

1. Sort predictions by descending confidence.
2. Select the highest-confidence prediction.
3. Add it to the retained set.
4. Calculate IoU against remaining candidates.
5. Suppress candidates whose IoU exceeds the configured threshold.
6. Repeat until no candidates remain or the detection limit is reached.

The suppression condition is:

    IoU(selected_box, candidate_box) > threshold

The implementations deliberately preserve candidates whose IoU equals the threshold.

### 6.2 Class-aware NMS

Class-aware NMS processes different object categories independently.

For example:

- Two overlapping person detections can suppress each other.
- A person detection does not suppress a car detection.

This is the default behavior in the four source implementations.

### 6.3 Class-agnostic NMS

Class-agnostic NMS compares all categories.

An overlapping prediction can suppress another prediction even when their class labels differ.

This mode can be useful in particular detection applications but may incorrectly eliminate valid overlapping objects from different categories.

### 6.4 NMS complexity

Sorting requires O(n log n) time.

The greedy overlap comparisons require O(n²) time in the worst case.

Memory usage is O(n).

For large candidate sets, optimized implementations can use vectorized operations, spatial indexing, compiled kernels or GPU acceleration.

The supplied implementations prioritize transparent algorithmic behavior.

## 7. Soft-NMS

Soft-NMS modifies candidate confidence scores rather than immediately removing overlapping predictions.

This can preserve legitimate detections in crowded scenes.

The Gaussian score-decay equation is:

    new_score = old_score * exp(-(IoU² / sigma))

Where:

- IoU measures overlap.
- Sigma controls the rate of score reduction.

A candidate with high overlap receives a larger confidence reduction.

A candidate with low overlap retains most of its original score.

Soft-NMS is useful when objects frequently appear close together, such as pedestrians in crowded streets.

The implementations preserve the original detection objects and generate updated predictions with modified confidence values.

Soft-NMS is not identical to hard NMS. It changes the scoring behavior and can produce a different final detection ranking.

## 8. Detection Evaluation

Detection evaluation compares predicted objects against ground-truth annotations.

A prediction is considered a true positive when:

- Its class matches the ground-truth class.
- Its image matches the ground-truth image.
- Its IoU satisfies the evaluation threshold.
- The ground-truth object has not already been matched.

Each ground-truth object can match at most one prediction.

### 8.1 Evaluation outcomes

| Outcome | Definition |
|---|---|
| True Positive | Correctly matched prediction |
| False Positive | Prediction without a valid match |
| False Negative | Ground-truth object without a matched prediction |

### 8.2 Precision

Precision measures the proportion of predictions that are correct.

Precision = TP / (TP + FP)

Higher precision indicates fewer false-positive detections.

### 8.3 Recall

Recall measures the proportion of ground-truth objects successfully detected.

Recall = TP / (TP + FN)

Higher recall indicates fewer missed objects.

### 8.4 F1-score

F1-score is the harmonic mean of precision and recall.

F1 = 2 × Precision × Recall / (Precision + Recall)

It provides a combined measure when both precision and recall matter.

### 8.5 Average Precision

Average Precision (AP) summarizes the precision-recall relationship across confidence-ranked predictions.

The Python and JavaScript implementations calculate interpolated AP for a single class and evaluation threshold.

The implementation:

1. Sorts predictions by confidence.
2. Matches predictions against available ground-truth objects.
3. Calculates cumulative precision and recall.
4. Constructs the interpolated precision envelope.
5. Integrates precision over recall changes.

This implementation is not a complete COCO evaluation implementation.

COCO-style evaluation additionally uses multiple IoU thresholds, area ranges, maximum detection limits and class-level aggregation.

Mean Average Precision (mAP) is obtained by averaging AP across the relevant categories and, depending on the evaluation protocol, IoU thresholds.

## 9. Detection Pipeline Design

A detection pipeline coordinates individual processing operations.

The implementation uses the following processing stages:

    Detection Input
         |
         v
    Metadata Validation
         |
         v
    Bounding Box Clipping
         |
         v
    Confidence Filtering
         |
         v
    NMS / Soft-NMS
         |
         v
    Detection Output
         |
         v
    Evaluation and Metrics

The stages have separate responsibilities.

| Stage | Responsibility |
|---|---|
| Metadata validation | Validate image identifiers and dimensions |
| Clipping | Restrict boxes to image boundaries |
| Confidence filtering | Remove low-confidence predictions |
| NMS | Eliminate redundant predictions |
| Evaluation | Compare predictions against annotations |
| Reporting | Record processing statistics |

### 9.1 Failure handling

The implementations reject:

- Invalid coordinates.
- NaN and infinite values.
- Invalid confidence scores.
- Invalid class identifiers.
- Missing image identifiers.
- Invalid image dimensions.
- Invalid suppression thresholds.
- Invalid maximum detection counts.
- Mixed image identifiers within one batch.

The JavaScript implementation also includes asynchronous processing, bounded concurrency, task queuing and pipeline closure.

The Java implementation uses explicit pipeline states and immutable result records.

The C++ implementation separates geometry, suppression and evaluation into domain-oriented structures and services.

The Python implementation emphasizes simulation, assertions, file output and readable workflow orchestration.

## 10. Python Implementation

The Python file uses standard-library features to construct a complete detection-processing simulation.

### Main components

| Component | Responsibility |
|---|---|
| BoundingBox | Bounding-box geometry and clipping |
| Detection | Immutable prediction representation |
| PipelineConfig | Processing configuration |
| DetectionPipeline | End-to-end processing |
| intersection_over_union | IoU calculation |
| generalized_iou | GIoU calculation |
| confidence_filter | Confidence filtering |
| non_maximum_suppression | Greedy NMS |
| soft_nms | Gaussian score decay |
| calculate_detection_metrics | Precision, recall and F1 |
| average_precision | Single-class AP |
| save_results | JSON persistence |

### Implementation decisions

Dataclasses represent detection records and configuration.

Frozen records prevent accidental modification of prediction metadata.

Dictionaries group predictions by class during class-aware suppression.

Assertions validate geometric calculations and invalid-input handling.

The pipeline records input counts, filtered counts, output counts and processing duration.

JSON output provides a structured representation suitable for subsequent analysis.

## 11. JavaScript Implementation

The JavaScript file provides an asynchronous and event-driven processing architecture.

### Main components

| Component | Responsibility |
|---|---|
| BoundingBox | Immutable geometric representation |
| Detection | Validated prediction object |
| DetectionPipeline | Asynchronous task processing |
| EventEmitter | Processing and failure notifications |
| confidenceFilter | Candidate filtering |
| nms | Greedy suppression |
| softNMS | Gaussian score reduction |
| averagePrecision | Detection evaluation |
| saveResults | Asynchronous JSON persistence |

### Event-driven processing

The pipeline extends Node.js EventEmitter.

It emits:

- `processed` when an image finishes processing.
- `pipelineError` when a processing operation fails.

The queue limits the number of simultaneously executing tasks.

The concurrency setting prevents unrestricted processing of incoming work.

Closing the pipeline rejects tasks that have not started.

### Persistence and security

The JSON writer:

- Validates detection objects.
- Resolves the destination path.
- Restricts output to a filename in the current directory.
- Uses a unique temporary filename.
- Creates the temporary file exclusively.
- Uses restrictive file permissions.
- Renames the completed temporary file into place.

Production deployments should also consider filesystem permissions, storage quotas, access controls and retention policies.

## 12. C++17 Implementation

The C++ implementation uses strongly structured domain objects and STL algorithms.

### Main components

| Component | Responsibility |
|---|---|
| BoundingBox | Geometric operations |
| Detection | Prediction representation |
| DetectionConfig | Processing policy |
| SuppressionEngine | Hard NMS and Soft-NMS |
| EvaluationResult | Evaluation metrics |
| RepositoryGovernanceEngine | Processing orchestration |

The suppression mode is represented by an enum.

The configuration validates thresholds and maximum detection counts before processing.

The implementation uses:

- `std::vector` for prediction collections.
- `std::sort` for confidence ordering.
- `std::copy_if` for confidence filtering.
- `std::set` for ground-truth matching.
- `std::move` for efficient ownership transfer.
- Exceptions for invalid domain states.

The engine reports input and output detection counts and the percentage reduction produced by the pipeline.

This design is appropriate for understanding how detection-processing components can be organized in a compiled systems application.

## 13. Java 17 Implementation

The Java implementation represents an enterprise-oriented detection service.

### Domain model

Java records represent immutable data:

- BoundingBox
- Detection
- PipelinePolicy
- PipelineResult
- EvaluationResult

Enums represent controlled states:

- SuppressionMode
- PipelineState

Service classes separate processing responsibilities.

| Service | Responsibility |
|---|---|
| GeometryService | IoU and GIoU |
| SuppressionService | Confidence filtering and NMS |
| DetectionEvaluationService | Prediction evaluation |
| DetectionPipeline | Processing orchestration |

### State management

The pipeline uses four states:

| State | Meaning |
|---|---|
| CREATED | Pipeline initialized |
| RUNNING | Processing is active |
| COMPLETED | Processing completed successfully |
| FAILED | Processing failed |

A failed pipeline cannot accept another processing operation.

Configuration objects are immutable, and output collections are copied before being exposed.

The evaluation service uses a set to prevent multiple predictions from matching the same ground-truth object.

These decisions support predictable state management and reduce accidental data modification.

## 14. PostgreSQL Database Architecture

The PostgreSQL implementation provides a relational persistence layer.

### Entity relationships

    Detection Model
          |
          v
      Detection Run
          |
          +-------------------+
          |                   |
          v                   v
       Image              Predictions
          |                   |
          v                   v
    Ground Truth          Suppression Events
          |
          v
      Evaluation
          |
          v
       Metrics

### Main tables

| Table | Purpose |
|---|---|
| detection_models | Model registry |
| object_classes | Object category registry |
| images | Image metadata |
| ground_truth_boxes | Annotated bounding boxes |
| detection_runs | Processing configuration and state |
| predictions | Model-generated predictions |
| suppression_events | Suppression audit records |
| evaluations | Detection evaluation results |

### Referential integrity

Primary keys identify records.

Foreign keys maintain relationships between models, images, runs, predictions and evaluation records.

Unique constraints prevent duplicate model versions and duplicate image identifiers.

Check constraints validate coordinates, thresholds, confidence scores and dimensions.

Triggers validate prediction and annotation coordinates against the corresponding image dimensions.

### Database functions

`calculate_iou` calculates bounding-box intersection over union.

`calculate_giou` calculates generalized intersection over union.

`filter_predictions` applies the configured confidence threshold.

`execute_hard_nms` processes candidate predictions and records suppression decisions.

`evaluate_detection_run` performs one-to-one matching against ground-truth annotations.

### Database views

`v_prediction_summary` aggregates detection counts and confidence information.

`v_evaluation_report` combines model metadata, image identifiers and evaluation metrics.

### Transactions

The processing workflow uses transactions to coordinate:

- Run state changes.
- Confidence filtering.
- NMS execution.
- Suppression audit insertion.
- Final run completion.
- Evaluation persistence.

A transaction ensures that related operations can be rolled back together if an operation fails.

The demonstration includes an explicit rollback operation.

### Performance considerations

The database defines indexes for:

- Dataset retrieval.
- Model status filtering.
- Run status and image lookup.
- Model processing history.
- Confidence-ranked predictions.
- Retained prediction retrieval.
- Ground-truth image and class matching.
- Suppression audit retrieval.

The procedural NMS implementation prioritizes transparency. Its nested comparisons can be expensive for large prediction sets.

Production workloads with large candidate volumes should perform NMS in an optimized application or compiled processing layer and persist the results in batches.

## 15. Important Technical Distinctions

| Concept | Primary purpose |
|---|---|
| Bounding Box | Represents object location |
| IoU | Measures overlap between two boxes |
| GIoU | Measures overlap and enclosing-region geometry |
| Confidence Filtering | Removes low-confidence predictions |
| Hard NMS | Removes redundant overlapping boxes |
| Soft-NMS | Reduces scores of overlapping boxes |
| Precision | Measures prediction correctness |
| Recall | Measures ground-truth detection coverage |
| F1-score | Combines precision and recall |
| AP | Summarizes precision-recall performance |
| mAP | Aggregates AP across classes and evaluation conditions |

These mechanisms solve different problems and should not be treated as interchangeable.

A bounding box describes geometry. IoU compares geometry. Confidence filtering evaluates prediction scores. NMS resolves redundant predictions. Evaluation measures detection quality against annotated data.

## 16. Execution Requirements

| Implementation | Requirement |
|---|---|
| Python | Python 3.10 or newer |
| JavaScript | Node.js 18 or newer |
| C++17 | C++ compiler supporting C++17 |
| Java | Java 17 JDK |
| PostgreSQL | PostgreSQL 15 or newer |

All four application implementations use standard language facilities without third-party dependencies.

The SQL script creates its own schema, tables, constraints, functions, procedures, triggers, indexes, sample data and reports.

## 17. Practical Limitations

The implementations focus on post-processing and evaluation rather than neural-network inference.

They do not implement:

- CNN or transformer-based feature extraction.
- Image decoding and resizing.
- Neural-network training.
- GPU inference.
- Anchor generation.
- Feature pyramid networks.
- Model quantization.
- Multi-scale inference.
- Mask segmentation.
- Keypoint estimation.
- Object tracking across video frames.

The implementations assume that raw model predictions have already been produced.

The evaluation examples use greedy one-to-one matching and a configurable IoU threshold. They do not constitute a complete COCO or Pascal VOC benchmark implementation.

The database NMS procedure is a transparent relational demonstration rather than a high-throughput inference engine.

## 18. Production Engineering Considerations

A production object detection system requires more than geometric correctness.

Important engineering considerations include:

- Consistent coordinate conventions between models and annotation systems.
- Versioned model artifacts.
- Reproducible preprocessing configurations.
- Confidence threshold calibration.
- Per-class suppression policies.
- Dataset quality validation.
- Secure image storage.
- Restricted access to model artifacts.
- Input size and resource limits.
- Processing timeout management.
- Detection provenance and audit records.
- Monitoring of precision, recall and latency.
- Evaluation against independent validation datasets.
- Detection drift monitoring.

For crowded scenes, suppression thresholds and Soft-NMS behavior require careful validation.

For real-time applications, end-to-end latency must include image decoding, preprocessing, model inference, post-processing and output delivery.

A reliable object detection pipeline therefore combines correct geometry, controlled prediction filtering, appropriate suppression, consistent evaluation and operational monitoring.
