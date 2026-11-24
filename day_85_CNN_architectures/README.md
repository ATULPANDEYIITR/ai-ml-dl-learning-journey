# CNN Architectures: LeNet, AlexNet, VGG16 and Inception

## 1. Technical Scope

Convolutional Neural Networks (CNNs) are deep learning architectures designed to process structured spatial data, particularly images.

They learn hierarchical representations through trainable convolution filters. Early layers generally identify simple patterns such as edges and textures. Deeper layers combine these patterns into more complex structures.

This implementation examines four influential CNN architectures:

| Architecture | Year | Main contribution | Architectural characteristic |
|---|---:|---|---|
| LeNet-5 | 1998 | Early practical CNN design | Convolution and average pooling |
| AlexNet | 2012 | Large-scale deep image classification | ReLU, dropout and GPU training |
| VGG16 | 2014 | Systematic deep convolutional design | Repeated 3x3 filters |
| Inception v1 | 2014 | Efficient multi-scale feature extraction | Parallel convolution branches |

The five source implementations provide different engineering perspectives:

| File | Engineering purpose |
|---|---|
| Python | Executable CNN architectures and training demonstration |
| JavaScript | Architecture registry and asynchronous experiment management |
| C++17 | Tensor and model analysis with deployment governance |
| Java 17 | Enterprise model registration, validation and approval |
| PostgreSQL | Relational architecture catalog and experiment governance |

## 2. Fundamental CNN Concepts

### 2.1 Convolution

Convolution applies a learnable filter to a local region of an input tensor.

For an input feature map and a filter, the operation calculates weighted sums across spatial locations and channels.

The output dimension is:

\[
O = \left\lfloor
\frac{I+2P-K}{S}
\right\rfloor+1
\]

Where:

- \(I\): input dimension
- \(P\): padding
- \(K\): kernel size
- \(S\): stride
- \(O\): output dimension

For example, a 32 × 32 input with a 5 × 5 filter, stride 1 and no padding produces a 28 × 28 output.

### 2.2 Filters and Feature Maps

A filter identifies patterns within an image.

Examples include:

- Horizontal edges.
- Vertical edges.
- Corners.
- Textures.
- Shapes.
- Object parts.

A convolutional layer typically learns multiple filters. Each filter produces a feature map.

### 2.3 Pooling

Pooling reduces the spatial dimensions of feature maps.

**Max pooling** selects the maximum value in a local region.

**Average pooling** calculates the average value in a local region.

Pooling reduces computational requirements and changes the spatial resolution of extracted features.

It does not introduce trainable convolution parameters.

### 2.4 Activation Functions

Activation functions introduce nonlinearity.

| Function | Mathematical behavior | Application |
|---|---|---|
| Tanh | Maps values between -1 and 1 | Historical CNN designs |
| ReLU | \(f(x)=\max(0,x)\) | AlexNet, VGG and Inception |
| Softmax | Converts logits into class probabilities | Multi-class classification |

ReLU is computationally simple and generally improves optimization compared with saturating activations in deep networks.

### 2.5 Fully Connected Layers

Fully connected layers combine extracted features to produce classification outputs.

For a dense layer:

\[
P = (I \times O)+O
\]

Where:

- \(I\): number of input features.
- \(O\): number of output features.
- \(P\): number of trainable parameters.

The additional \(O\) represents bias parameters.

### 2.6 Parameter Calculation

For a standard convolutional layer:

\[
P=(K_h \times K_w \times C_{in}\times C_{out})+C_{out}
\]

Where:

- \(K_h\): filter height.
- \(K_w\): filter width.
- \(C_{in}\): input channels.
- \(C_{out}\): output channels.

For a 3 × 3 convolution with 64 input channels and 128 output channels:

\[
P=(3\times3\times64\times128)+128
\]

\[
P=73,856
\]

The parameter count depends on filter dimensions and channel counts, not image height and width.

---

## 3. LeNet-5

<PRIVATE_PERSON> LeCun and collaborators introduced LeNet-5 in the context of handwritten character recognition.

LeNet demonstrated that convolutional networks could learn useful visual representations through hierarchical feature extraction.

### Architecture

The original LeNet-5 was designed around 32 × 32 grayscale images.

Its principal stages are:

1. First convolutional layer.
2. Average pooling.
3. Second convolutional layer.
4. Average pooling.
5. Final convolutional feature transformation.
6. Fully connected classification layers.

The simplified implementation uses the following structure:

| Stage | Operation | Output channels | Spatial dimensions |
|---|---|---:|---:|
| Input | Grayscale image | 1 | 32 × 32 |
| C1 | 5 × 5 convolution | 6 | 28 × 28 |
| S2 | 2 × 2 average pooling | 6 | 14 × 14 |
| C3 | 5 × 5 convolution | 16 | 10 × 10 |
| S4 | 2 × 2 average pooling | 16 | 5 × 5 |
| C5 | 5 × 5 convolution | 120 | 1 × 1 |
| F6 | Fully connected | 84 | Vector |
| Output | Classification | Number of classes | Vector |

The implementation uses Tanh activations and average pooling.

### Important characteristics

- Relatively small architecture.
- Low computational requirements.
- Suitable for simple grayscale classification.
- Limited feature capacity compared with modern deep networks.
- Historically important for practical CNN development.

### Practical limitations

LeNet is not designed for large-scale natural image classification without architectural modification.

Its small feature extraction capacity makes it unsuitable for many high-resolution visual recognition tasks.

---

## 4. AlexNet

AlexNet was introduced by Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton in 2012.

It demonstrated the effectiveness of deep convolutional networks trained using GPUs for large-scale image classification.

### Architecture

AlexNet contains five convolutional layers followed by three fully connected layers.

| Stage | Operation | Main characteristics |
|---|---|---|
| 1 | Convolution | 11 × 11 filters, stride 4 |
| 2 | Max pooling | Spatial reduction |
| 3 | Convolution | 5 × 5 filters |
| 4 | Max pooling | Spatial reduction |
| 5 | Convolution | 3 × 3 filters |
| 6 | Convolution | 3 × 3 filters |
| 7 | Convolution | 3 × 3 filters |
| 8 | Max pooling | Final feature reduction |
| 9 | Fully connected | 4096 neurons |
| 10 | Fully connected | 4096 neurons |
| 11 | Classification | Output classes |

The implementation uses adaptive average pooling before its classifier so that the feature dimensions remain compatible with its fully connected layers.

### Important innovations

**ReLU activation**

AlexNet used ReLU to improve the efficiency of network training.

**Dropout**

Dropout randomly disables selected neurons during training to reduce co-adaptation and overfitting.

**Data augmentation**

Image transformations increase training diversity.

Examples include:

- Cropping.
- Horizontal flipping.
- Color perturbation.

**GPU computation**

GPU acceleration made training a much larger CNN practical.

### Practical limitations

AlexNet contains large fully connected layers and requires substantial memory.

Its early convolutional layer uses relatively large filters and aggressive spatial downsampling.

Modern architectures often use more parameter-efficient feature extraction.

---

## 5. VGG16

VGG16 was developed by the Visual Geometry Group at the University of Oxford.

It demonstrated the effectiveness of increasing network depth while using small convolution filters.

### Architecture

VGG16 contains:

- Thirteen convolutional layers.
- Five max-pooling operations.
- Three fully connected layers.
- Sixteen weighted layers in total.

Its convolutional stages follow a consistent structure.

| Block | Convolution layers | Channels | Pooling |
|---|---:|---:|---|
| Block 1 | 2 | 64 | 2 × 2 |
| Block 2 | 2 | 128 | 2 × 2 |
| Block 3 | 3 | 256 | 2 × 2 |
| Block 4 | 3 | 512 | 2 × 2 |
| Block 5 | 3 | 512 | 2 × 2 |

Each convolution generally uses:

- 3 × 3 filters.
- Stride 1.
- Padding 1.
- ReLU activation.

### Why 3 × 3 filters?

Multiple small filters can provide a larger effective receptive field.

Two consecutive 3 × 3 convolutions have an effective receptive field of 5 × 5.

Three consecutive 3 × 3 convolutions have an effective receptive field of 7 × 7.

This allows deeper nonlinear feature transformations while using small kernels.

### Parameter considerations

VGG16 has a large number of parameters, particularly in its original fully connected classifier.

Its memory and computational requirements can be substantial.

The Python implementation uses a configurable classifier width, defaulting to 4096, while preserving the thirteen-convolution VGG16 feature extractor.

The default classifier corresponds to the original large-width design.

### Practical limitations

- High parameter count.
- Significant memory consumption.
- Expensive inference relative to more efficient designs.
- Large intermediate feature maps.

VGG16 remains useful for studying deep convolutional feature extraction and transfer learning.

---

## 6. Inception v1

Inception v1, also known as GoogLeNet, was introduced by researchers at Google in 2014.

Its central idea is to process multiple receptive field sizes in parallel.

Rather than selecting only one convolution size for a particular stage, an Inception module combines several branches.

### Inception module

<PRIVATE_PERSON> The four main branches are:

| Branch | Operation | Purpose |
|---|---|---|
| 1 | 1 × 1 convolution | Channel-wise feature transformation |
| 2 | 1 × 1 reduction followed by 3 × 3 convolution | Medium-scale spatial features |
| 3 | 1 × 1 reduction followed by 5 × 5 convolution | Larger receptive field |
| 4 | 3 × 3 max pooling followed by 1 × 1 projection | Pooled features |

The outputs are concatenated along the channel dimension.

### Parallel processing

All branches receive the same input tensor.

Their spatial dimensions must match before concatenation.

For example, four branches producing:

- 64 channels.
- 128 channels.
- 32 channels.
- 32 channels.

produce:

\[
64+128+32+32=256
\]

The concatenated output contains 256 channels.

### Role of 1 × 1 convolutions

A 1 × 1 convolution changes the channel dimension without directly combining neighboring spatial positions.

It is particularly useful for reducing channel counts before expensive 3 × 3 and 5 × 5 operations.

For an input containing 192 channels, a 1 × 1 reduction to 16 channels substantially reduces the subsequent 5 × 5 convolution's computational cost.

### Global average pooling

The final feature maps are reduced to 1 × 1 spatial dimensions through global average pooling.

This reduces dependence on large fully connected layers.

### Practical limitations

Inception requires careful management of parallel branches and channel dimensions.

Its implementation is more complex than the sequential VGG design.

The Python implementation provides a compact Inception-v1-style classifier with multiple Inception modules. It does not implement every original GoogLeNet component, such as the original auxiliary classifiers.

---

## 7. Architectural Comparison

| Characteristic | LeNet | AlexNet | VGG16 | Inception v1 |
|---|---|---|---|---|
| Historical period | 1998 | 2012 | 2014 | 2014 |
| Main design | Sequential | Sequential | Sequential | Multi-branch |
| Convolution depth | 3 | 5 | 13 | Multiple modules |
| Typical input | 32 × 32 | 224 × 224 | 224 × 224 | 224 × 224 |
| Main activation | Tanh | ReLU | ReLU | ReLU |
| Pooling | Average | Max | Max | Max and average |
| Filter strategy | Small filters | Mixed filter sizes | Repeated 3 × 3 | Multiple receptive fields |
| Fully connected classifier | Small | Large | Large | Compact |
| Main advantage | Simplicity | Large-scale learning | Uniform deep design | Multi-scale processing |
| Main limitation | Limited capacity | Large dense layers | High parameter count | Complex topology |

These architectures represent different design choices rather than interchangeable implementations.

---

## 8. Source Implementation Architecture

### Python: Executable Neural Networks

The Python implementation uses PyTorch.

It contains actual neural-network modules rather than only architecture metadata.

| Component | Responsibility |
|---|---|
| `LeNet` | Small convolutional classifier |
| `AlexNet` | Five convolutional stages and dense classifier |
| `VGG16` | Thirteen convolutional layers |
| `InceptionBlock` | Four parallel convolution branches |
| `InceptionV1` | Multi-stage Inception network |
| `count_parameters` | Trainable parameter calculation |
| `calculate_model_size` | Parameter storage estimation |
| `inspect_model` | Tensor shape and inference validation |
| `demonstrate_training_step` | Forward pass, loss, backward pass and optimization |

The training demonstration uses:

- Cross-entropy loss.
- SGD.
- Momentum.
- Weight decay.
- Gradient clipping.

Synthetic inputs are used for execution testing. They do not represent a meaningful trained image classifier.

The script supports CPU and CUDA execution.

Example execution:

- `python cnn_architectures.py`
- `python cnn_architectures.py --architecture all`
- `python cnn_architectures.py --architecture inception --train-step`

The `all` option evaluates models sequentially to avoid keeping all model instances in memory simultaneously.

### JavaScript: Architecture Registry and Experiment Events

The JavaScript implementation provides a structured experiment management model.

Its primary components are:

| Component | Responsibility |
|---|---|
| `TensorShape` | Validates tensor dimensions |
| `Layer` | Defines a layer's structural properties |
| `ArchitectureDefinition` | Stores architecture metadata |
| `ArchitectureRegistry` | Maintains architecture definitions |
| `ArchitectureAnalyzer` | Calculates convolution parameters |
| `Experiment` | Tracks experiment lifecycle |
| `ExperimentService` | Validates and processes experiments |

The `EventEmitter` mechanism exposes experiment events:

- `validated`
- `started`
- `completed`
- `failed`

Experiment transitions are controlled.

An experiment cannot move directly from `CREATED` to `COMPLETED`.

The implementation uses asynchronous execution to represent experiment orchestration. It is an architecture analysis and workflow system, not a JavaScript tensor-training framework.

### C++17: Model Analysis and Deployment Governance

The C++ implementation focuses on strongly typed domain modeling and validation.

Its major components include:

- `ArchitectureRegistry`
- `ModelAnalyzer`
- `DeploymentGovernance`
- `DeploymentRequest`
- `TensorShape`
- `AnalysisResult`

The model analyzer calculates convolution and dense parameter counts.

The deployment governance engine evaluates:

1. Architecture registration.
2. Input tensor compatibility.
3. Class count.
4. Model evaluation.
5. Security review.
6. Reproducibility verification.
7. Deployment authorization.

The implementation uses `std::map` for deterministic architecture lookup and `std::vector` for ordered layer definitions.

### Java 17: Enterprise Governance

The Java implementation uses records, enums, collections and service classes.

The major domain objects are:

- `ArchitectureDefinition`
- `ExperimentConfiguration`
- `AnalysisReport`
- `QualityGates`
- `DeploymentRecord`

The services separate architecture analysis from deployment authorization.

The deployment lifecycle is:

`DRAFT → VALIDATED → APPROVED → DEPLOYED`

A failed quality review changes the state to `REJECTED`.

The four quality gates are:

| Gate | Purpose |
|---|---|
| Accuracy | Verify classification performance |
| Security | Verify model and deployment security |
| Reproducibility | Verify experiment configuration and repeatability |
| Resource | Verify computational resource requirements |

A deployment cannot proceed unless all four gates pass.

### PostgreSQL: Relational Experiment Management

The SQL implementation models the architecture research lifecycle.

Its principal relationships are:

    Architectures
        |
        +---- Architecture Layers
        |
        +---- Model Configurations
                    |
                    +---- Experiments
                    |       |
                    |       +---- Experiment Metrics
                    |       |
                    |       +---- Datasets
                    |
                    +---- Model Reviews
                    |       |
                    |       +---- Reviewers
                    |
                    +---- Deployments
                            |
                            +---- Deployment Policies
                            |
                            +---- Audit Events

The database separates architecture definitions from individual trained model configurations.

This distinction allows multiple experiments to use the same architecture with different datasets, hyperparameters and training configurations.

---

## 9. Training Workflow

A typical CNN experiment follows this sequence:

    Image Dataset
          |
          v
    Data Validation
          |
          v
    Image Preprocessing
          |
          v
    Model Architecture Selection
          |
          v
    Weight Initialization
          |
          v
    Forward Propagation
          |
          v
    Loss Calculation
          |
          v
    Backpropagation
          |
          v
    Optimizer Update
          |
          v
    Validation
          |
          v
    Performance Evaluation
          |
          v
    Model Governance
          |
          v
    Deployment Approval

### Forward propagation

The input image passes through convolutional, activation and pooling layers.

The final feature representation is converted into classification logits.

### Loss calculation

For multi-class classification, categorical cross-entropy is commonly used.

For logits \(z\) and correct class \(y\):

\[
L=-\log\left(
\frac{e^{z_y}}{\sum_j e^{z_j}}
\right)
\]

### Backpropagation

Gradients are calculated from the loss and propagated backward through the network.

Each trainable parameter receives a gradient.

### Optimization

An optimizer updates the weights using calculated gradients.

For basic gradient descent:

\[
\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)
\]

Where:

- \(\theta\): model parameters.
- \(\eta\): learning rate.
- \(\nabla L\): loss gradient.

---

## 10. Model Evaluation

Accuracy alone is insufficient for evaluating image classifiers.

The implementations and database model support several evaluation measurements.

### Accuracy

\[
Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}
\]

### Precision

\[
Precision=\frac{TP}{TP+FP}
\]

### Recall

\[
Recall=\frac{TP}{TP+FN}
\]

### F1-score

\[
F1=2\times\frac{Precision\times Recall}
{Precision+Recall}
\]

### Computational measurements

| Measurement | Purpose |
|---|---|
| Parameter count | Measures trainable model size |
| Model storage | Estimates parameter memory |
| Training loss | Measures optimization objective |
| Validation loss | Measures performance on validation data |
| Inference latency | Measures prediction time |
| Peak memory | Measures memory consumption |
| Accuracy | Measures classification correctness |

For reliable comparison, architectures must be evaluated using consistent datasets, preprocessing, data splits and evaluation protocols.

Training settings must also be documented.

---

## 11. Dataset Compatibility

Different architectures have different input contracts.

| Dataset | Image dimensions | Channels | Classes | Suitable architecture |
|---|---|---:|---:|---|
| MNIST | 28 × 28 | 1 | 10 | LeNet |
| CIFAR-10 | 32 × 32 | 3 | 10 | Modified LeNet |
| ImageNet | Variable, commonly resized or cropped to 224 × 224 | 3 | 1000 | AlexNet, VGG16, Inception |

The Python implementation uses LeNet with 32 × 32 inputs.

MNIST images must therefore be resized or padded to the expected dimensions.

The architecture registry and SQL catalog use explicit input channel and dimension checks.

A configuration that violates its architecture's input contract is rejected.

Dataset licensing and access conditions must be checked before redistribution or commercial use.

---

## 12. Edge Cases and Failure Conditions

The implementations account for important engineering problems.

| Condition | Expected behavior |
|---|---|
| Unknown architecture | Reject the configuration |
| Invalid input dimensions | Raise validation error |
| Incorrect input channels | Reject tensor contract |
| Fewer than two classes | Reject classification configuration |
| Invalid learning rate | Reject experiment |
| Invalid batch size | Reject experiment |
| Invalid tensor dimensions | Reject analysis |
| Incompatible layer channels | Reject sequential model specification |
| Non-finite training loss | Stop the training step |
| Missing evaluation approval | Reject deployment |
| Failed security review | Reject deployment |
| Failed reproducibility review | Reject deployment |
| Incomplete training | Reject deployment eligibility |

The Inception implementation also requires all parallel branches to produce identical spatial dimensions before channel concatenation.

This is enforced by matching padding and stride configurations.

---

## 13. Performance and Memory Considerations

CNN computational requirements depend on:

- Input resolution.
- Number of channels.
- Number of filters.
- Kernel dimensions.
- Network depth.
- Batch size.
- Intermediate feature-map dimensions.

A rough convolution computation estimate is:

\[
FLOPs \approx
2HWC_{out}K_hK_wC_{in}
\]

This assumes one multiply and one addition per operation.

The estimate excludes additional operations such as activation functions, normalization and pooling.

### Architecture-specific considerations

**LeNet**

Small models can execute efficiently on CPUs.

**AlexNet**

Large fully connected layers increase parameter storage and memory requirements.

**VGG16**

Repeated convolutional operations and large dense layers produce substantial computational and memory costs.

**Inception**

Parallel branches enable multi-scale feature extraction, but require simultaneous intermediate feature maps and more complex execution scheduling.

### Memory optimization

Common techniques include:

- Smaller batch sizes.
- Mixed-precision training.
- Gradient checkpointing.
- Model quantization.
- Efficient data loading.
- Optimized convolution kernels.
- Removing unused intermediate tensors.

The appropriate optimization depends on the hardware and the workload.

---

## 14. Security and Reproducibility

Model engineering requires more than successful training.

A production model should maintain traceability between:

- Architecture definition.
- Dataset version.
- Dataset license.
- Preprocessing configuration.
- Random seed.
- Training configuration.
- Model weights.
- Evaluation results.
- Deployment decision.

Important security controls include:

- Validate uploaded datasets and file formats.
- Restrict access to model artifacts.
- Protect training and evaluation datasets.
- Avoid logging sensitive image content.
- Verify model artifact checksums.
- Restrict model serialization inputs.
- Maintain deployment audit records.
- Separate training and production permissions.

PyTorch model loading should use safer weight-loading practices, such as `weights_only=True` where compatible, and artifacts should be obtained from trusted sources.

The SQL schema records model reviews and deployment decisions. In a production environment, database permissions should prevent unauthorized changes to approval records and audit history.

---

## 15. Database Design and Governance

The PostgreSQL implementation separates five major concerns.

### Architecture definitions

The `architectures` table stores the architecture family, publication year, input requirements and design characteristics.

### Layer definitions

The `architecture_layers` table stores ordered layers, kernel sizes, channel counts and branch membership.

Inception branch records are distinguished using `branch_name`.

### Experiments

The `experiments` table stores execution states and training progress.

The `experiment_metrics` table stores epoch-level measurements.

A unique constraint on experiment and epoch prevents duplicate epoch records.

### Quality reviews

The `model_reviews` table connects models, reviewers and review categories.

Review records distinguish:

- Accuracy approval.
- Security approval.
- Reproducibility approval.
- Resource approval.
- Data governance approval.

### Deployment policies

The `deployment_policies` table stores measurable acceptance thresholds.

The `check_deployment_eligibility` function checks:

- Experiment completion.
- Minimum completed epochs.
- Accuracy.
- Inference latency.
- Memory consumption.
- Required security approval.
- Required reproducibility approval.
- Accuracy review.

The `approve_deployment` procedure locks the relevant deployment record while evaluating its eligibility and updating its state.

This prevents concurrent approval requests from independently changing the same deployment record without synchronization.

---

## 16. Limitations and Implementation Boundaries

The implementations serve different purposes and should not be interpreted as identical model implementations.

| File | Limitation |
|---|---|
| Python | Uses synthetic inputs unless a real dataset pipeline is supplied |
| JavaScript | Estimates selected layer parameters rather than executing tensor operations |
| C++17 | Uses architecture specifications for governance and parameter analysis rather than training neural networks |
| Java 17 | Provides enterprise lifecycle simulation rather than distributed training |
| PostgreSQL | Stores experiment and governance information rather than neural-network weights or executing CNN inference |

The Python Inception model is a compact Inception-v1-style implementation, not an exact reproduction of every original GoogLeNet training and auxiliary-classifier component.

The JavaScript and C++ architecture catalogs use representative layer specifications for analysis. Their reported convolution parameter estimates must not be treated as complete parameter counts for the executable Python models.

The SQL layer catalog stores selected representative architectural layers and is not intended to reconstruct every computational operation in every complete network.

The SQL deployment policy evaluates recorded experiment measurements. These measurements are only as reliable as the evaluation pipeline that produced them.

For scientific comparisons, all implementations must use consistent architecture definitions and documented experimental configurations.

---

## 17. Execution Requirements

### Python

Requires Python 3.10 or newer and PyTorch 2.0 or newer.

Run:

`python cnn_architectures.py`

Evaluate all architectures:

`python cnn_architectures.py --architecture all`

Run a training demonstration:

`python cnn_architectures.py --architecture lenet --train-step`

### JavaScript

Requires Node.js 18 or newer.

Run:

`node cnn_architectures.js`

No third-party npm packages are required.

### C++17

Requires a C++17-compatible compiler.

Compile:

`g++ -std=c++17 -O2 -Wall -Wextra -pedantic cnn_architectures.cpp -o cnn`

Run:

`./cnn`

### Java 17

Requires Java Development Kit 17.

Compile:

`javac CnnArchitectureGovernance.java`

Run:

`java CnnArchitectureGovernance`

### PostgreSQL

Requires PostgreSQL 14 or newer.

Execute the SQL file using `psql`:

`psql -U postgres -d postgres -f cnn_architectures.sql`

The script creates a dedicated `cnn_lab` schema and commits the catalog, sample experiments, evaluation records and deployment policies in a transaction.
