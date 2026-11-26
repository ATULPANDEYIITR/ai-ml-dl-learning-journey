# Computer Vision Pipeline: Image Preprocessing, Augmentation and Classification

## 1. Project Scope

A computer vision pipeline transforms raw digital images into structured representations that can be evaluated by classification algorithms.

This project implements three connected stages:

1. **Image preprocessing:** Converts images into a consistent representation suitable for analysis.
2. **Data augmentation:** Generates controlled variations of existing training images to improve dataset diversity.
3. **Image classification:** Extracts numerical features and assigns images to predefined categories.

The implementations also address dataset management, model persistence, validation, batch processing, performance measurement, and operational governance.

The five source files provide complementary implementations:

| File | Engineering perspective | Main responsibility |
|---|---|---|
| Python | Pipeline simulation | End-to-end image processing and classification |
| JavaScript | Event-driven processing | Asynchronous training and batch inference |
| C++17 | Computational engineering | Image transformations and classification engine |
| Java 17 | Enterprise architecture | Model lifecycle, registry, and classification services |
| PostgreSQL | Data management | Dataset, processing, augmentation, and model records |

All four application implementations use the P3 ASCII Portable Pixmap format for image ingestion. This provides a dependency-free way to demonstrate actual image operations.

The classification algorithm is a nearest-centroid baseline using image statistics. It is intentionally transparent and does not represent a deep learning architecture.

---

## 2. Computer Vision Pipeline Architecture

A typical image classification system follows this processing sequence:

                  IMAGE SOURCES
                       |
                       v
                IMAGE INGESTION
                       |
                       v
                INPUT VALIDATION
                       |
                       v
                IMAGE PREPROCESSING
                       |
             +---------+---------+
             |                   |
             v                   v
        TRAINING DATA         TEST DATA
             |
             v
       DATA AUGMENTATION
             |
             v
       FEATURE EXTRACTION
             |
             v
       MODEL TRAINING
             |
             v
       MODEL VALIDATION
             |
             v
       CLASSIFICATION MODEL
             |
             v
       IMAGE PREDICTION
             |
             v
       EVALUATION AND REPORTING


Training and inference have different responsibilities.

Training uses labeled images to construct a classification model. Inference processes previously unseen images using an existing model.

Test data must remain separate from training and augmentation operations. This separation provides a more reliable measurement of generalization.

3. Image Representation and Ingestion
3.1 Digital image structure
A digital image is represented as a collection of pixels.

An RGB image contains three channels:
- Red
- Green
- Blue
An image can be represented as a three-dimensional array:
\[
I \in \mathbb{R}^{H \times W \times 3}
\]

Where:
- \(H\) represents image height.
- \(W\) represents image width.
- \(3\) represents the number of color channels.
For an image of dimensions 256 × 256:
\[
256 \times 256 \times 3 = 196608
\]

The image therefore contains 196,608 individual channel values.

An 8-bit RGB image normally stores channel values between 0 and 255.

A 16-bit image supports a larger intensity range.
3.2 PPM image format
The implementations support P3 Portable Pixmap images.

The format contains:
Component	Description
P3	ASCII RGB image format identifier
Width	Number of horizontal pixels
Height	Number of vertical pixels
Maximum value	Maximum permitted channel intensity
Pixel data	Red, green, and blue values

The format is useful for understanding image processing because pixel values remain directly accessible.

The implementations validate:
- Image dimensions.
- Pixel count.
- Channel count.
- Maximum channel intensity.
- Numeric values.
- File availability.
- Unexpected or incomplete pixel data.
The JavaScript implementation also uses asynchronous file operations, while the C++ and Java implementations use their respective standard file-handling facilities.
3.3 Image integrity
An image should not enter the processing pipeline until its basic properties have been validated.

Important ingestion checks include:
- File existence.
- Supported image format.
- Valid image dimensions.
- Valid channel values.
- Maximum image size.
- Successful file reading.
- Content integrity.
A production system should also verify file signatures, malware scanning results, image decoder behavior, and cryptographic checksums.

4. Image Preprocessing
Image preprocessing converts heterogeneous source images into a consistent representation.

It reduces unnecessary variation in image dimensions, intensity ranges, and color representation.
4.1 Image resizing
Image resizing changes the spatial dimensions of an image.

For example:
\[
256 \times 256 \rightarrow 32 \times 32
\]

The target image contains fewer pixels and therefore requires less memory and processing.

The source implementations use nearest-neighbor resizing.

For a target coordinate \((x,y)\), the corresponding source coordinate is calculated as:
\[
x_s =
\left\lfloor
\frac{xW_s}{W_t}
\right\rfloor
\]

\[
y_s =
\left\lfloor
\frac{yH_s}{H_t}
\right\rfloor
\]

Where:
- \(W_s\) and \(H_s\) are source dimensions.
- \(W_t\) and \(H_t\) are target dimensions.
Nearest-neighbor interpolation selects an existing source pixel without calculating intermediate intensities.

Advantages
- Low computational cost.
- Simple implementation.
- No interpolation arithmetic.
- Suitable for categorical pixel data.
Limitations
- Can produce blocky images.
- May discard important image details.
- Can introduce aliasing when reducing image dimensions.
Other interpolation methods include:
Method	Mechanism	Typical consideration
Nearest neighbor	Selects the closest source pixel	Fastest, but less smooth
Bilinear	Uses four neighboring pixels	Moderate computational cost
Bicubic	Uses a larger neighborhood	Smoother interpolation
Lanczos	Uses a windowed sinc filter	Often preserves more detail

The PostgreSQL preprocessing profile records the selected interpolation method.
4.2 Pixel normalization
Normalization changes the numerical range of pixel values.

For 8-bit images, zero-to-one normalization is:
\[
I_{normalized} = \frac{I}{255}
\]

The resulting values belong to:
\[
[0,1]
\]

Another commonly used transformation is:
\[
I_{normalized} =
\frac{I}{255}\times 2-1
\]

Its output range is:
\[
[-1,1]
\]

The implementations support both approaches.

Normalization is useful because numerical algorithms often behave more consistently when input values occupy controlled ranges.

The appropriate normalization method depends on the model architecture and training configuration.
4.3 Grayscale conversion
Grayscale conversion reduces three RGB channels to one intensity value.

The implementations use perceptual luminance weighting:
\[
Y = 0.2126R + 0.7152G + 0.0722B
\]

Green receives the largest weight because human visual perception is more sensitive to green intensity than to equal changes in red or blue.

Grayscale conversion can reduce computation when color information is not essential.

It should not be used when color is an important classification feature.
4.4 Standardization
Standardization transforms numerical observations using their mean and standard deviation.
\[
z = \frac{x-\mu}{\sigma}
\]

Where:
- \(x\) is the original value.
- \(\mu\) is the mean.
- \(\sigma\) is the standard deviation.
When standard deviation is zero, the implementations return zero values instead of dividing by zero.

For production machine learning, standardization parameters must be calculated from training data and reused during validation and inference.

Calculating them independently for test data can introduce data leakage.

5. Data Augmentation
5.1 Concept
Data augmentation creates additional training examples through controlled image transformations.

It increases the diversity of training observations without requiring an equivalent number of newly collected images.

Original image:
          ORIGINAL IMAGE
                |
       +--------+--------+
       |        |        |
       v        v        v
     FLIP    ROTATION  BRIGHTNESS
       |        |        |
       v        v        v
   VARIANT  VARIANT   VARIANT


The transformed images normally retain the original class label when the transformation does not change the underlying semantic category.
5.2 Horizontal flipping
Horizontal flipping reverses the horizontal pixel order.
\[
I'(x,y) = I(W-1-x,y)
\]

This transformation can improve robustness when object orientation is not important.

It is inappropriate for some directional classification problems, such as reading text or identifying left-versus-right anatomical structures.
5.3 Vertical flipping
Vertical flipping reverses the vertical pixel order.
\[
I'(x,y) = I(x,H-1-y)
\]

It is appropriate only when vertically inverted images remain semantically valid.
5.4 Rotation
Rotation changes image orientation.

A 90-degree clockwise rotation transforms:
\[
W \times H \rightarrow H \times W
\]

The implementations preserve pixel values while exchanging image dimensions.

Arbitrary-angle rotation requires interpolation and may introduce empty regions around image boundaries.

The PostgreSQL augmentation policy records a maximum rotation angle for downstream processing.
5.5 Brightness adjustment
Brightness adjustment multiplies each channel by a factor:
\[
I' = \alpha I
\]

Where \(\alpha\) controls brightness.
- \(\alpha < 1\): Darker image.
- \(\alpha = 1\): Original brightness.
- \(\alpha > 1\): Brighter image.
Pixel values are clipped to the permitted intensity range.

Clipping prevents invalid channel values.
5.6 Augmentation controls
The implementations use deterministic random seeds.

A fixed seed makes transformation sequences reproducible, which is useful for debugging and controlled experiments.

The Python, JavaScript, C++, and Java implementations use different random-number mechanisms.

A matching numerical seed does not guarantee identical transformations across languages.
5.7 Augmentation risks
Incorrect augmentation can reduce model quality.

Important risks include:
- Changing the semantic meaning of an image.
- Introducing unrealistic visual patterns.
- Increasing duplication without adding meaningful diversity.
- Creating different distributions between training and deployment.
- Applying transformations to validation or test images.
- Generating augmented images with incorrect labels.
Augmentation must occur after the training split has been established.

The SQL implementation enforces that recorded augmentation sources and generated images belong to the training split.

6. Feature Extraction
Feature extraction converts images into numerical representations.

The implementations calculate nine RGB statistical features.
Feature	Description
red_mean	Mean normalized red intensity
green_mean	Mean normalized green intensity
blue_mean	Mean normalized blue intensity
red_std	Red-channel standard deviation
green_std	Green-channel standard deviation
blue_std	Blue-channel standard deviation
red_ratio	Red intensity relative to total RGB intensity
green_ratio	Green intensity relative to total RGB intensity
blue_ratio	Blue intensity relative to total RGB intensity

6.1 Channel mean
For a channel containing \(N\) pixel values:
\[
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
\]

The mean represents the average intensity of that channel.
6.2 Channel standard deviation
\[
\sigma =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\mu)^2
}
\]

Standard deviation measures the dispersion of pixel intensities.

Higher values indicate greater variation across the channel.
6.3 Color proportions
The relative contribution of an RGB channel is:
\[
R_{ratio} =
\frac{\mu_R}
{\mu_R+\mu_G+\mu_B}
\]

Equivalent calculations are performed for green and blue.

A small numerical safeguard prevents division by zero.
6.4 Feature representation
The resulting feature vector is:
\[
X =
[
R_m,G_m,B_m,
R_s,G_s,B_s,
R_r,G_r,B_r
]
\]

The classifier compares these vectors using Euclidean distance.

This representation is computationally inexpensive, but it does not preserve detailed spatial arrangements of objects.

Two visually different images can have similar channel statistics.

7. Image Classification
7.1 Nearest-centroid classification
The implementations use a nearest-centroid classifier.

It calculates one representative feature vector for every class.

For class \(k\), its centroid is:
\[
C_k =
\frac{1}{N_k}
\sum_{i=1}^{N_k}X_i
\]

Where \(N_k\) represents the number of training observations belonging to class \(k\).

For an unseen image, the classifier calculates its distance from each centroid.
7.2 Euclidean distance
\[
d(X,C_k) =
\sqrt{
\sum_{j=1}^{n}
(X_j-C_{kj})^2
}
\]

The predicted class is the class with the smallest distance.
\[
\hat{y} =
\underset{k}{\operatorname{argmin}}
\ d(X,C_k)
\]

7.3 Confidence scores
The implementations convert negative distances into normalized exponential scores.
\[
P(k|X) =
\frac{
e^{-(d_k-d_{min})}
}{
\sum_j e^{-(d_j-d_{min})}
}
\]

Subtracting the minimum distance improves numerical stability.

The predicted class is associated with the largest score.

These values are similarity-derived scores, not statistically calibrated probabilities.

A production classifier should undergo probability calibration before these scores are interpreted as reliable confidence estimates.
7.4 Classification example
The demonstration dataset contains three classes:
Class	Dominant color	Expected prediction
Red	Red channel	Red
Green	Green channel	Green
Blue	Blue channel	Blue

The training pipeline uses six synthetic images.

Augmentation creates additional training observations.

Three separate synthetic test images evaluate the classifier.

The executable examples validate the expected class predictions and model persistence.
7.5 Classification limitations
The nearest-centroid classifier is suitable for demonstrating:
- Feature extraction.
- Numerical distance calculations.
- Model training.
- Prediction.
- Model serialization.
- Classification workflows.
It does not provide the representational capability of a convolutional neural network.

It cannot reliably recognize complex objects, textures, shapes, or spatial arrangements using only nine global color statistics.

8. Python Implementation
The Python implementation provides the main end-to-end pipeline.
Main components
Component	Responsibility
Image	Validated image representation
PipelineConfig	Processing configuration
PPMReader	Image file ingestion and export
ImagePreprocessor	Resizing, normalization, grayscale conversion, and standardization
ImageAugmentor	Image transformations
FeatureExtractor	RGB statistical features
NearestCentroidClassifier	Training and classification
ComputerVisionPipeline	End-to-end orchestration

Important implementation decisions
The implementation uses dataclasses for configuration and result structures.

Enums represent classification-related categories.

Dictionaries store feature vectors and model centroids.

Lists store pixel data and training observations.

Custom exceptions distinguish image-format failures from validation and processing errors.

The classifier supports JSON model persistence.

The model-loading mechanism checks feature dimensions and numerical validity before accepting stored centroids.
Execution
The default demonstration creates synthetic RGB images, trains the classifier, performs predictions, tests transformations, and saves the model.

The image-processing mode accepts an existing P3 PPM file and a saved model.

9. JavaScript Implementation
The JavaScript implementation emphasizes event-driven processing and asynchronous operations.
Main components
Component	Responsibility
PPMCodec	Asynchronous image reading and writing
ImagePreprocessor	Image preparation
ImageAugmentor	Deterministic transformations
FeatureExtractor	Statistical feature generation
NearestCentroidClassifier	Classification and model persistence
ComputerVisionPipeline	Event-driven orchestration

Event-driven architecture
The pipeline extends Node.js EventEmitter.

It exposes events for:
- Training completion.
- Individual sample processing.
- Sample processing errors.
- Prediction completion.
This approach separates processing operations from event consumers.

An application can use these events to update monitoring systems, progress indicators, or operational logs.
Asynchronous processing
Image file operations use Node.js asynchronous filesystem functions.

Batch inference uses asynchronous worker functions.

The concurrency limit prevents an unrestricted number of simultaneous prediction operations.

The results are stored according to their original input positions.

This preserves the relationship between input images and prediction outputs even when workers complete at different times.
Model integrity
The JavaScript classifier calculates a SHA-256 fingerprint from its model metadata and centroid values.

This supports model identity checks after serialization and restoration.

A fingerprint does not independently establish model provenance or authenticity. Production systems should also use signed artifacts and controlled model storage.

10. C++17 Implementation
The C++ implementation emphasizes computational efficiency and explicit memory management.
Main components
Component	Responsibility
Pixel	RGB channel representation
Image	Image dimensions and pixel storage
PPMCodec	Image ingestion and export
ImagePreprocessor	Resizing and numerical transformations
ImageAugmentor	Geometric and brightness transformations
FeatureExtractor	Fixed-size feature vectors
NearestCentroidClassifier	Classification engine
ComputerVisionPipeline	Pipeline coordination

Engineering decisions
The image uses a contiguous std::vector<Pixel> container.

This provides predictable memory layout and avoids the allocation overhead associated with nested dynamic containers.

Feature vectors use std::array<double, 9> because the feature dimension is fixed.

std::map provides deterministic ordering for class labels and centroid storage.

The implementation validates dimensions and pixel values before processing.
Computational characteristics
For an image containing \(N\) pixels:
Operation	Time complexity
Image validation	\(O(N)\)
Resizing	\(O(W_tH_t)\)
Horizontal flipping	\(O(N)\)
Vertical flipping	\(O(N)\)
90-degree rotation	\(O(N)\)
Feature extraction	\(O(N)\)
Centroid training	\(O(MF)\)
Prediction	\(O(KF)\)

Where:
- \(N\) is the number of image pixels.
- \(M\) is the number of training observations.
- \(F\) is the feature dimension.
- \(K\) is the number of classes.
The fixed feature dimension makes classification inexpensive relative to image decoding and preprocessing.

11. Java 17 Enterprise Implementation
The Java implementation introduces explicit enterprise-oriented domain modeling.
Main architectural components
Component	Responsibility
PipelineConfig	Immutable processing configuration
Image	Validated image record
TrainingRecord	Labeled training input
FeatureVector	Immutable numerical representation
ClassificationModel	Model lifecycle and classification
ModelRegistry	Model registration and activation
ClassificationService	Prediction orchestration
Classifier	Classification contract

Model lifecycle
The implementation defines the following states:
State	Meaning
CREATED	Model metadata has been initialized
TRAINING	Training is in progress
TRAINED	Model parameters have been calculated
ACTIVE	Model is available for inference
FAILED	Training or processing has failed
RETIRED	Model has been withdrawn

State transitions prevent invalid lifecycle operations.

For example, a newly created model cannot immediately become active.

It must first complete training.

The model registry controls activation and identifies the model used for inference.
Enterprise design considerations
The implementation uses Java records for immutable data structures.

The Classifier interface separates model behavior from service orchestration.

The registry prevents duplicate model identifiers.

The classification service provides batch prediction through a bounded executor.

Model persistence records metadata and class centroids.

The model lifecycle and registry provide an application-level governance example. They are not substitutes for distributed model registries, access-control systems, artifact signing, or production deployment approvals.

12. PostgreSQL Data Architecture
The PostgreSQL implementation models the relational lifecycle of image classification.

It separates image metadata, processing operations, augmentation records, feature vectors, model versions, evaluations, and predictions.
12.1 Entity relationships
DATASET
   |
   v
IMAGE_ASSET
   |
   +----------------------+
   |                      |
   v                      v
DATASET_ASSIGNMENT    PREPROCESSING_RUN
                          |
                          v
                  PREPROCESSING_PROFILE
                          |
                          v
                    FEATURE_VECTOR

IMAGE_ASSET
   |
   v
AUGMENTATION_RUN
   |
   v
AUGMENTATION_POLICY

DATASET
   |
   v
CLASSIFICATION_MODEL
   |
   +-----------------------+
   |                       |
   v                       v
MODEL_EVALUATION        PREDICTION
                           |
                           v
                       IMAGE_CLASS


12.2 Core tables
Table	Purpose
dataset	Dataset identity and ownership
image_class	Classification categories
image_asset	Image metadata and validation state
dataset_assignment	Training, validation, and testing allocation
preprocessing_profile	Image preprocessing configuration
preprocessing_run	Processing execution records
augmentation_policy	Transformation configuration
augmentation_run	Generated image relationships
feature_vector	Extracted numerical representations
classification_model	Model metadata and lifecycle
model_evaluation	Classification evaluation metrics
prediction	Prediction results and inference metadata
model_status_audit	Model lifecycle history

12.3 Relational integrity
Primary keys identify individual records.

Foreign keys prevent references to nonexistent datasets, images, models, and classes.

Unique constraints prevent duplicate dataset names, model versions, and image content within a dataset.

Check constraints validate image dimensions, pixel metadata, confidence values, evaluation metrics, and processing states.

The indexes support common retrieval operations.

The JSONB feature representation provides flexibility for feature metadata, while the relational columns preserve important identifiers and processing relationships.
12.4 Dataset split management
Dataset assignments distinguish:
- Training images.
- Validation images.
- Testing images.
The dataset_assignment table ensures that an image has one recorded split assignment.

The augmentation trigger checks that both the original and generated images belong to the training split.

This prevents a recorded augmentation operation from using validation or testing images.

It is important to distinguish this database-level validation from complete dataset-leakage prevention.

A production system must also consider duplicate content, related image sequences, common subjects, acquisition sessions, and multiple versions of the same source image.
12.5 Preprocessing workflow
The preprocessing workflow records:
1. Image selection.
2. Preprocessing profile selection.
3. Execution initiation.
4. Processing completion or failure.
5. Output location.
6. Execution duration.
The complete_preprocessing function updates a running operation and records its output.

The function rejects attempts to complete operations that are not running.
12.6 Model lifecycle enforcement
The database defines explicit model states.

A trigger prevents invalid state transitions.

Model activation requires:
- A trained model.
- A model artifact.
- A validation evaluation.
- Validation accuracy of at least 0.80.
These are illustrative governance thresholds.

Actual activation criteria should reflect the classification task, dataset characteristics, class distribution, error costs, and deployment risks.

The audit trigger records status changes with timestamps and database-user information.
12.7 Evaluation reporting
The SQL implementation includes reporting for:
- Dataset distribution.
- Class proportions.
- Preprocessing duration.
- Processing failures.
- Model evaluation.
- Prediction accuracy.
- Misclassification counts.
- Inference latency.
- Model activation eligibility.
- Model lifecycle history.
Common Table Expressions simplify multi-stage reporting queries.

Window functions calculate class proportions.

Aggregate filtering separates successful and failed processing operations.

The PERCENTILE_CONT function supports inference-latency percentile reporting.

13. Model Evaluation
Classification performance must be measured using data that was not used to fit the model.
13.1 Accuracy
\[
Accuracy =
\frac{Correct\ Predictions}
{Total\ Predictions}
\]

Accuracy is useful when class distributions and error costs are reasonably balanced.

It can be misleading for imbalanced datasets.
13.2 Precision
\[
Precision =
\frac{TP}{TP+FP}
\]

Precision measures how many positive predictions are correct.
13.3 Recall
\[
Recall =
\frac{TP}{TP+FN}
\]

Recall measures how many actual positive observations are identified.
13.4 F1-score
\[
F1 =
2 \times
\frac{Precision \times Recall}
{Precision+Recall}
\]

F1 balances precision and recall.

Macro-F1 calculates F1 independently for each class and then averages the results.

It gives equal weight to each class, making it useful when class representation differs substantially.
13.5 Confusion matrix
A confusion matrix provides a detailed view of classification errors.

For three classes:
Actual / Predicted	Red	Green	Blue
Red	TP	Error	Error
Green	Error	TP	Error
Blue	Error	Error	TP

The PostgreSQL error report aggregates misclassifications by actual and predicted class.
13.6 Inference latency
Inference latency measures the time required to produce a prediction.

Operational evaluation should consider:
- Average latency.
- Median latency.
- 95th-percentile latency.
- Maximum latency.
- Throughput.
- Resource consumption.
A model with high accuracy may still be unsuitable for applications with strict response-time requirements.

14. Data Leakage and Reproducibility
Data leakage occurs when information that should be unavailable during training influences model development or evaluation.

Common causes include:
- Training on test images.
- Applying augmentation before dataset splitting.
- Calculating normalization statistics using the entire dataset.
- Duplicating related images across different splits.
- Tuning model parameters using test results.
- Reusing evaluation observations during training.
A safer sequence is:
SOURCE DATASET
      |
      v
IMAGE VALIDATION
      |
      v
DATASET SPLITTING
      |
      +----------------------+
      |          |           |
      v          v           v
   TRAINING   VALIDATION   TESTING
      |
      v
 AUGMENTATION
      |
      v
 MODEL TRAINING
      |
      v
 VALIDATION
      |
      v
 FINAL TEST EVALUATION


Reproducibility requires recording:
- Dataset version.
- Image identifiers.
- Split assignments.
- Preprocessing configuration.
- Augmentation parameters.
- Random seed.
- Feature extraction version.
- Model architecture.
- Training parameters.
- Model artifact identifier.
- Evaluation results.
The PostgreSQL schema provides relational structures for recording these details.

15. Security and Production Considerations
15.1 Image ingestion security
Image files are untrusted inputs.

A production ingestion service should enforce:
- Maximum file size.
- Maximum dimensions.
- Supported image formats.
- Decoder resource limits.
- Content validation.
- Malware scanning.
- Storage access controls.
- Content checksums.
File extensions must not be treated as proof of file type.
15.2 Model security
Model artifacts should be stored in controlled locations.

Recommended controls within an operational deployment include:
- Role-based access control.
- Artifact integrity verification.
- Signed model releases.
- Model version tracking.
- Controlled activation.
- Audit logging.
- Restricted model replacement.
- Secure credential management.
15.3 Privacy
Images may contain personal, confidential, or commercially sensitive information.

Processing systems should apply appropriate access restrictions and retention policies.

Sensitive image metadata should not be exposed through unrestricted reporting interfaces.
15.4 Processing reliability
Production pipelines should distinguish between:
- Invalid input.
- Unsupported image format.
- Preprocessing failure.
- Augmentation failure.
- Feature extraction failure.
- Model loading failure.
- Classification failure.
- Storage failure.
Failures should be recorded with sufficient diagnostic information without unnecessarily exposing confidential image contents.
15.5 Scalability
The provided implementations use in-memory processing for educational clarity.

Large-scale deployments require additional engineering considerations:
- Streaming image ingestion.
- Batch-oriented processing.
- Parallel decoding.
- GPU acceleration.
- Distributed object storage.
- Worker queues.
- Dataset sharding.
- Model-serving infrastructure.
- Processing retries.
- Monitoring and alerting.
The JavaScript and Java examples demonstrate bounded concurrent batch processing.

The C++ implementation demonstrates contiguous pixel storage and fixed-size feature vectors.

The PostgreSQL implementation separates operational metadata from image storage.

16. Implementation Comparison
Capability	Python	JavaScript	C++17	Java 17	PostgreSQL
Image ingestion	Yes	Yes	Yes	Yes	Metadata
Image validation	Yes	Yes	Yes	Yes	Constraints
Resizing	Yes	Yes	Yes	Yes	Configuration
Normalization	Yes	Yes	Yes	Yes	Configuration
Grayscale conversion	Yes	Yes	Yes	Yes	Not implemented
Standardization	Yes	Yes	Yes	Not implemented	Not implemented
Horizontal flipping	Yes	Yes	Yes	Yes	Policy
Vertical flipping	Yes	Yes	Yes	Yes	Policy
Rotation	90 degrees	90 degrees	90 degrees	90 degrees	Policy
Brightness	Yes	Yes	Yes	Yes	Policy
Feature extraction	Yes	Yes	Yes	Yes	Storage
Classification	Yes	Yes	Yes	Yes	Reporting
Model persistence	JSON	JSON	Text	Text	Metadata
Batch processing	Dataset workflow	Async workers	Sequential	Concurrent	Set-based queries
Audit trail	Logging	Events	Error handling	Model states	Audit table


17. Execution and Validation
Python
Requirements:
- Python 3.10 or later.
- Standard library only.
Default execution performs training, prediction, transformation tests, and model persistence.

Image classification mode accepts a P3 PPM file and a saved JSON model.
JavaScript
Requirements:
- Node.js 18 or later.
- Standard library only.
Default execution performs asynchronous training, event reporting, batch prediction, and model persistence.

The prediction mode loads a PPM image and a saved JSON model.
C++17
Requirements:
- A C++17-compatible compiler.
- Standard library only.
The program performs image preprocessing, augmentation, training, classification, and model persistence.

The prediction mode loads an existing PPM image and a saved model.
Java 17
Requirements:
- JDK 17.
- Standard Java libraries only.
The application demonstrates immutable data structures, model registration, activation, batch prediction, and model persistence.
PostgreSQL
Requirements:
- PostgreSQL 15 or later.
- Permission to create a schema, tables, functions, and triggers.
The SQL script creates the schema, domain types, tables, indexes, functions, triggers, views, and sample records.

It executes the initial workflow in a transaction and provides separate retrieval, validation, evaluation, and reporting queries.

18. Technical Limitations
The implementations intentionally use a small statistical classification model and a basic image format.

Their limitations should be understood when interpreting the results.
Area	Current implementation	Production alternative
Image formats	P3 PPM	JPEG, PNG, TIFF, WebP and other required formats
Resizing	Nearest neighbor	Bilinear, bicubic or Lanczos
Rotation	90-degree rotation	Arbitrary-angle rotation with interpolation
Features	RGB statistics	Spatial, texture, edge and learned features
Classification	Nearest centroid	CNN, vision transformer or task-specific model
Training	Centroid calculation	Gradient-based optimization
Confidence	Distance-derived scores	Calibrated predictive probabilities
Evaluation	Demonstration predictions and SQL metrics	Independent benchmark datasets
Storage	Local model files and relational metadata	Controlled artifact and object storage
Processing	In-memory	Distributed or accelerator-based processing

The implementations are executable baseline systems for studying the mechanics of a computer vision pipeline.

They do not claim to provide deep learning performance, production image decoding, or validated performance on real-world object recognition datasets.
