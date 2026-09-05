# Machine Learning Paradigms

## Overview

Machine learning can be approached in several different ways depending on how data, feedback, and learning objectives are defined.

The five major paradigms covered in this guide are:

1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Self-Supervised Learning
5. Reinforcement Learning

Each paradigm solves a different class of problems. Understanding the distinction between them is important when selecting an appropriate machine learning approach for a real-world problem.

---

## 1. Supervised Learning

Supervised learning uses labeled data.

Each training example contains:

- Input features
- A known target or label

The objective is to learn a relationship between the input and the target so that predictions can be made for previously unseen data.

### Example

Suppose a company has historical information about houses:

- Area
- Number of bedrooms
- Location
- Age of the property
- Selling price

The house characteristics are the input features, while the selling price is the target.

The model learns from historical examples and predicts prices for new houses.

### Main Types

Supervised learning is primarily divided into:

- Classification
- Regression

### Classification

Classification predicts a discrete category.

Examples include:

- Spam or not spam
- Fraudulent or legitimate transaction
- Disease or no disease
- Customer churn or no churn
- Image category
- Sentiment category

Common classification algorithms include:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- K-Nearest Neighbors
- Naive Bayes
- Gradient Boosting
- Neural Networks

### Regression

Regression predicts a continuous numerical value.

Examples include:

- House prices
- Sales revenue
- Temperature
- Demand forecasting
- Delivery time
- Stock-related numerical predictions

Common regression algorithms include:

- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression
- Neural Networks

### Training Process

A typical supervised learning workflow consists of:

1. Collect labeled data.
2. Clean the data.
3. Select useful features.
4. Split the data into training, validation, and test sets.
5. Train a model.
6. Measure its performance.
7. Tune the model.
8. Evaluate it on unseen test data.
9. Deploy the model if performance is acceptable.

### Loss Function

A supervised model needs a way to measure prediction error.

For regression, common loss functions include:

- Mean Squared Error
- Mean Absolute Error
- Huber Loss

For classification, common loss functions include:

- Binary Cross-Entropy
- Categorical Cross-Entropy
- Hinge Loss

The learning algorithm attempts to minimize the selected loss function.

### Overfitting

Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data.

A model can have:

- High training performance
- Poor validation or test performance

Common techniques for reducing overfitting include:

- Regularization
- Cross-validation
- Early stopping
- Dropout
- Data augmentation
- Simplifying the model
- Increasing training data

### Underfitting

Underfitting occurs when a model is too simple to capture important patterns in the data.

It generally produces poor performance on both training and unseen data.

---

## 2. Unsupervised Learning

Unsupervised learning works with data that does not have explicitly provided target labels.

The objective is to discover useful patterns, structures, relationships, or representations within the data.

Instead of asking:

"What is the correct answer?"

the problem often asks:

"What structure exists in this data?"

### Common Applications

Unsupervised learning is used for:

- Customer segmentation
- Document grouping
- Anomaly detection
- Data exploration
- Dimensionality reduction
- Pattern discovery
- Recommendation systems
- Market segmentation

### Clustering

Clustering groups similar observations together.

For example, a retailer could analyze customers using:

- Age
- Purchase frequency
- Average spending
- Product preferences

A clustering algorithm may discover groups such as:

- High-value frequent customers
- Occasional customers
- Discount-focused customers
- Inactive customers

The groups are discovered from the data rather than manually assigned beforehand.

### K-Means

K-Means is one of the most widely known clustering algorithms.

Its general process is:

1. Select the number of clusters.
2. Initialize cluster centers.
3. Assign each observation to its nearest center.
4. Recalculate cluster centers.
5. Repeat the assignment and recalculation process.
6. Stop when the clusters stabilize or another stopping condition is reached.

A major limitation of K-Means is that the number of clusters generally needs to be selected beforehand.

### Hierarchical Clustering

Hierarchical clustering creates a hierarchy of groups.

It can be useful when the analyst wants to understand relationships at different levels of similarity.

The results are commonly represented using a dendrogram.

### DBSCAN

DBSCAN is a density-based clustering algorithm.

It can identify:

- Dense groups
- Noise
- Outliers

Unlike K-Means, DBSCAN does not require the number of clusters to be specified directly.

It can also identify clusters with more irregular shapes.

### Dimensionality Reduction

Datasets can contain hundreds or thousands of features.

Dimensionality reduction attempts to represent the data using fewer dimensions while retaining important information.

Common techniques include:

- Principal Component Analysis
- t-SNE
- UMAP
- Autoencoders

### Principal Component Analysis

PCA transforms the original features into a smaller number of new components.

The first components attempt to capture the greatest amount of variance in the data.

PCA can be used for:

- Visualization
- Noise reduction
- Feature compression
- Faster modeling
- Exploratory analysis

### Anomaly Detection

Unsupervised methods can identify observations that behave differently from the majority of the dataset.

Applications include:

- Fraud detection
- Network intrusion detection
- Equipment monitoring
- Manufacturing quality control
- Unusual financial transactions

---

## 3. Semi-Supervised Learning

Semi-supervised learning combines:

- A relatively small amount of labeled data
- A larger amount of unlabeled data

This is useful when obtaining labels is expensive, slow, or requires human expertise.

### Example

Imagine a company has:

- 1,000 manually labeled images
- 100,000 unlabeled images

Training only on the 1,000 labeled images may limit performance.

Semi-supervised learning attempts to extract useful information from the much larger unlabeled dataset.

### Why Use It?

Labeling data can be expensive.

Examples:

- Medical images require experts.
- Legal documents require domain specialists.
- Scientific datasets may require researchers.
- Audio may require trained annotators.
- Large image collections may require extensive manual review.

Unlabeled data is often much easier to collect.

### Pseudo-Labeling

One common strategy is pseudo-labeling.

A model first learns from labeled examples.

It then generates predictions for unlabeled examples.

Predictions with sufficiently high confidence may be treated as temporary labels.

The model can then train using both:

- Original labeled examples
- High-confidence pseudo-labeled examples

The process can be repeated carefully.

### Risks

Pseudo-labeling can propagate incorrect predictions.

If the initial model makes confident mistakes, those mistakes may become training labels.

Important considerations include:

- Confidence thresholds
- Model calibration
- Quality of labeled data
- Distribution differences
- Confirmation bias

### Consistency-Based Learning

Another approach is to encourage a model to produce similar predictions when the same example is subjected to reasonable transformations or perturbations.

The assumption is that small changes that do not alter the underlying meaning should not drastically change the prediction.

---

## 4. Self-Supervised Learning

Self-supervised learning creates training signals from the data itself.

There is no requirement for humans to manually provide labels for every training example.

The system constructs a learning task from the available data.

This approach is especially useful when enormous amounts of unlabeled data are available.

### Core Idea

Instead of manually saying:

"This image belongs to category A."

the learning system creates an internal prediction task.

For example:

- Hide part of an input and predict the missing information.
- Predict the next element in a sequence.
- Determine whether two transformed examples represent the same underlying content.
- Predict relationships between different parts of an observation.

The generated target is derived from the original data.

### Masked Prediction

A portion of an input is hidden.

The model attempts to reconstruct or predict the missing information.

This idea can be applied to:

- Text
- Images
- Audio
- Video
- Other structured sequences

### Contrastive Learning

Contrastive learning teaches representations by comparing examples.

The model attempts to bring related representations closer together while separating unrelated representations.

For example, two different transformations of the same image may be treated as related examples.

### Autoregressive Learning

An autoregressive objective predicts future elements from previous elements.

For sequential data, the general concept is:

Given previous information, predict what comes next.

This can be applied to:

- Text
- Audio
- Time series
- Other sequential data

### Representation Learning

A major purpose of self-supervised learning is learning useful representations.

Instead of directly solving one narrow task, the model learns a representation that can later be reused for multiple tasks.

For example, a representation learned from a large collection of images may later support:

- Image classification
- Object detection
- Image retrieval
- Similarity search

### Pretraining and Fine-Tuning

A common workflow is:

1. Pretrain a model using a large unlabeled dataset.
2. Learn general-purpose representations.
3. Adapt the model to a specific downstream task.
4. Fine-tune using task-specific data.

This approach can reduce the amount of labeled data required for downstream applications.

---

## 5. Reinforcement Learning

Reinforcement learning is based on interaction between an agent and an environment.

The agent takes actions and receives feedback in the form of rewards or penalties.

The objective is to learn a strategy that maximizes cumulative reward.

### Core Components

A reinforcement learning system commonly contains:

- Agent
- Environment
- State
- Action
- Reward
- Policy

### Agent

The agent is the decision-making system.

Examples include:

- A game-playing program
- A robot
- A recommendation system
- An autonomous control system

### Environment

The environment is the world in which the agent operates.

It provides observations or states and responds to actions.

### State

A state represents relevant information about the current situation.

For a game, the state may contain:

- Player position
- Opponent position
- Available resources
- Current score

### Action

An action is a decision available to the agent.

Examples include:

- Move left
- Move right
- Accelerate
- Stop
- Select an item
- Recommend an item

### Reward

A reward provides feedback about an action.

Positive rewards encourage desirable outcomes.

Negative rewards discourage undesirable outcomes.

The reward design is extremely important because the agent optimizes the objective represented by the reward signal.

### Policy

A policy determines how the agent selects actions based on its current situation.

A policy can be deterministic or stochastic.

### Return

The agent generally cares about cumulative future rewards rather than only the immediate reward.

The discounted return is commonly represented conceptually as:

Return = current reward + discounted future rewards

The discount factor controls how strongly future rewards influence the current decision.

---

## Markov Decision Processes

Many reinforcement learning problems are modeled as Markov Decision Processes.

An MDP is commonly described using:

- States
- Actions
- Transition dynamics
- Rewards
- Discount factor

The Markov assumption means that the relevant information needed to determine future behavior is captured by the current state.

This assumption does not always hold perfectly in real-world environments.

---

## Exploration and Exploitation

A reinforcement learning agent faces a fundamental trade-off.

### Exploitation

The agent chooses actions that it already believes will produce good results.

### Exploration

The agent tries actions that may be less certain but could reveal better strategies.

A system that only exploits may never discover better behavior.

A system that only explores may fail to consistently use what it has already learned.

Effective reinforcement learning balances both.

---

## Q-Learning

Q-learning estimates the value of taking an action in a particular state.

The Q-value represents the expected usefulness of taking an action and continuing according to the learning process.

The algorithm updates its estimates based on:

- Current state
- Selected action
- Reward
- Next state
- Estimated future value

Q-learning is an off-policy temporal-difference learning method.

---

## Temporal-Difference Learning

Temporal-difference learning updates value estimates using other learned estimates.

It does not necessarily need to wait until the end of an entire episode before learning.

This makes temporal-difference methods useful for continuing decision-making problems.

---

## Monte Carlo Learning vs Temporal-Difference Learning

Monte Carlo methods generally learn from complete episodes.

Temporal-difference methods can update estimates using intermediate experience.

### Monte Carlo

Advantages:

- Simple conceptual interpretation
- Uses actual observed returns

Limitations:

- Usually requires complete episodes
- Updates can be delayed

### Temporal Difference

Advantages:

- Can learn before an episode finishes
- Suitable for continuing tasks
- Often supports faster incremental updates

Limitations:

- Uses estimated values
- Can inherit errors from existing estimates

---

## Model-Free and Model-Based Reinforcement Learning

### Model-Free

The agent learns how to act without explicitly learning a complete model of the environment.

Examples include:

- Q-learning
- SARSA
- Policy gradient methods

### Model-Based

The agent uses or learns information about how the environment behaves.

It can use that model to plan future actions.

Model-based approaches can be more sample-efficient in some situations but may require an accurate environment model.

---

## Policy-Based Methods

Instead of learning only action values, policy-based methods directly optimize the policy.

Policy gradient methods are an important family of algorithms.

They are useful for problems involving:

- Continuous actions
- Stochastic policies
- Complex decision spaces

---

## Actor-Critic Methods

Actor-critic methods combine two ideas.

### Actor

The actor selects actions according to the current policy.

### Critic

The critic evaluates the quality of states or actions.

The actor improves its behavior using feedback from the critic.

This architecture forms the basis of several modern reinforcement learning algorithms.

---

## Multi-Armed Bandits

Multi-armed bandits are related to reinforcement learning but represent a simplified decision problem.

An agent chooses among several actions and receives rewards.

The primary challenge is balancing:

- Exploration
- Exploitation

Applications include:

- Recommendation selection
- Advertising
- Online experiments
- Content ranking

---

# Comparing the Five Paradigms

| Paradigm | Typical Data | Main Feedback | Main Objective |
|---|---|---|---|
| Supervised | Labeled data | Correct target | Predict target |
| Unsupervised | Unlabeled data | No explicit target | Discover structure |
| Semi-Supervised | Labeled + unlabeled | Partial labels | Improve learning with limited labels |
| Self-Supervised | Unlabeled data | Automatically generated target | Learn useful representations |
| Reinforcement | Interaction data | Rewards and penalties | Maximize cumulative reward |

---

# Key Differences

## Supervised vs Unsupervised

Supervised learning has known targets.

Unsupervised learning does not require explicitly provided targets.

A classification model may learn:

"Is this transaction fraudulent?"

A clustering model may instead discover:

"Which transactions appear similar?"

---

## Semi-Supervised vs Self-Supervised

These approaches are related but not identical.

Semi-supervised learning uses some human-provided labels together with unlabeled data.

Self-supervised learning generates its learning signal from the data itself.

Self-supervised learning can also be used as part of a larger supervised or semi-supervised workflow.

---

## Self-Supervised vs Supervised

Supervised learning typically depends on manually defined labels.

Self-supervised learning constructs a prediction task from the input data.

A self-supervised model can later be fine-tuned using supervised labels.

---

## Reinforcement Learning vs Supervised Learning

Supervised learning learns from examples with known targets.

Reinforcement learning learns from interactions and rewards.

In supervised learning, the system may be told the correct answer.

In reinforcement learning, the system may only receive feedback about how good the outcome was.

---

# Choosing the Right Paradigm

Use supervised learning when:

- Reliable labels are available.
- The target variable is clearly defined.
- Prediction is the main objective.

Use unsupervised learning when:

- Labels are unavailable.
- You want to discover groups or patterns.
- Exploratory analysis is important.

Use semi-supervised learning when:

- A small labeled dataset exists.
- A much larger unlabeled dataset is available.
- Labeling additional examples is expensive.

Use self-supervised learning when:

- Large amounts of unlabeled data are available.
- Useful representations need to be learned.
- Manual labeling is expensive.
- Pretraining is beneficial.

Use reinforcement learning when:

- Decisions occur sequentially.
- Actions influence future states.
- Feedback comes through rewards.
- The objective involves cumulative outcomes.

---

# Evaluation

Different paradigms require different evaluation strategies.

## Supervised Learning

Classification metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Log Loss

Regression metrics include:

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R-squared

The correct metric depends on the business or operational objective.

For example, accuracy may be inappropriate for a highly imbalanced fraud detection problem.

---

## Unsupervised Learning

Evaluation can be more difficult because there may be no ground-truth labels.

Possible approaches include:

- Silhouette score
- Davies-Bouldin index
- Cluster stability
- Reconstruction error
- Downstream task performance
- Human interpretation

A mathematically strong clustering result is not automatically useful from a business perspective.

---

## Semi-Supervised Learning

Evaluation should primarily be performed on a reliable labeled validation or test set.

Care must be taken to prevent pseudo-labels from contaminating evaluation data.

---

## Self-Supervised Learning

The quality of learned representations is often evaluated through downstream tasks.

A representation may be tested using:

- Classification
- Retrieval
- Detection
- Similarity measurement
- Fine-tuning performance

The pretraining objective alone does not necessarily tell you how useful the representation will be for every downstream application.

---

## Reinforcement Learning

Common evaluation considerations include:

- Average episode reward
- Success rate
- Sample efficiency
- Stability
- Safety
- Generalization
- Performance under environmental changes

A high reward does not necessarily mean the behavior is desirable if the reward function is poorly designed.

---

# Common Machine Learning Problems

## Data Leakage

Data leakage occurs when information that would not legitimately be available during prediction is used during training.

Examples include:

- Using future information
- Performing preprocessing using the complete dataset before splitting
- Including target-derived features
- Allowing test information to influence model selection

Data leakage can produce misleadingly high evaluation results.

---

## Distribution Shift

A model may be trained on one distribution and deployed on another.

Examples include:

- Changing customer behavior
- New products
- Changing economic conditions
- Different sensors
- Different geographic regions

A model should be evaluated against realistic deployment conditions.

---

## Concept Drift

Concept drift occurs when the relationship between inputs and targets changes over time.

A model that performed well historically may gradually become less accurate.

Monitoring and retraining strategies may therefore be required.

---

## Class Imbalance

Some classification problems contain far more examples of one class than another.

For example:

- 99.5% legitimate transactions
- 0.5% fraudulent transactions

A model predicting "legitimate" for every transaction could achieve high accuracy while being useless for fraud detection.

Metrics such as precision, recall, F1 score, and PR-AUC can provide more useful information.

---

## Calibration

A model can be accurate while producing poorly calibrated probabilities.

If a model says that an event has a 70% probability, calibration asks whether events predicted at approximately 70% probability actually occur roughly 70% of the time.

Calibration is particularly important when predictions are used for decision-making.

---

# Practical Machine Learning Workflow

A practical project often follows these stages:

## 1. Problem Definition

Clearly define:

- What needs to be predicted or optimized?
- Who will use the result?
- What constitutes success?
- What constraints exist?

## 2. Data Collection

Identify:

- Data sources
- Data quality
- Data volume
- Data availability
- Collection frequency

## 3. Data Preparation

Perform appropriate:

- Cleaning
- Transformation
- Missing-value handling
- Feature engineering
- Encoding
- Scaling

## 4. Data Splitting

Create appropriate training, validation, and test datasets.

The split strategy should reflect how the model will be used in production.

For time-dependent problems, random splitting may be inappropriate.

## 5. Baseline

Create a simple baseline before developing a complicated model.

A baseline helps determine whether additional complexity produces meaningful improvement.

## 6. Model Selection

Select algorithms based on:

- Data characteristics
- Objective
- Computational requirements
- Interpretability requirements
- Latency requirements
- Available training data

## 7. Training

Train the model using the appropriate learning objective.

## 8. Evaluation

Evaluate using metrics that correspond to the real-world objective.

## 9. Error Analysis

Study incorrect predictions and failure cases.

Error analysis can reveal:

- Data quality problems
- Missing features
- Incorrect labels
- Model limitations
- Distribution differences

## 10. Deployment

Deploy the model into the target environment.

## 11. Monitoring

Monitor:

- Prediction quality
- Input distributions
- Output distributions
- Latency
- Resource consumption
- Data drift
- Concept drift
- Failure rates

## 12. Retraining

Retrain when the model becomes outdated or when new high-quality data becomes available.

---

# Hybrid Approaches

Real-world systems do not always use a single paradigm.

Multiple approaches can be combined.

Examples include:

- Self-supervised pretraining followed by supervised fine-tuning
- Unsupervised clustering followed by supervised classification
- Semi-supervised learning combined with pseudo-labeling
- Reinforcement learning combined with supervised pretraining
- Self-supervised representations used in reinforcement learning
- Human feedback combined with automated learning

The best architecture depends on the problem, available data, and operational constraints.

---

# Human-in-the-Loop Learning

Human involvement can remain important even when automated learning methods are used.

Humans may:

- Label difficult examples
- Review uncertain predictions
- Correct model errors
- Validate clusters
- Define reward functions
- Identify unsafe behavior
- Provide domain expertise

Human review is particularly valuable for high-impact or high-risk applications.

---

# Active Learning

Active learning attempts to identify which examples would be most valuable for human labeling.

Instead of randomly labeling thousands of examples, the system may prioritize:

- Uncertain examples
- Diverse examples
- Rare examples
- Representative examples
- Examples near decision boundaries

This can reduce labeling effort when expert annotation is expensive.

---

# Weak Supervision

Weak supervision uses imperfect sources to generate training signals.

Possible sources include:

- Heuristic rules
- Existing databases
- External knowledge
- User behavior
- Programmatic labeling functions

Weak supervision can increase the amount of usable training data, but the generated labels may contain noise.

---

# Security Considerations

Machine learning systems can face security threats such as:

- Adversarial examples
- Data poisoning
- Model theft
- Membership inference
- Privacy leakage
- Training-data manipulation
- Prompt or input manipulation
- Reward manipulation

Security should be considered throughout the machine learning lifecycle rather than only after deployment.

---

# Production Considerations

A model that performs well in a notebook may still fail in production.

Production systems should consider:

- Latency
- Throughput
- Memory requirements
- Availability
- Scalability
- Monitoring
- Versioning
- Reproducibility
- Data pipelines
- Model rollback
- Security
- Cost

Machine learning is therefore not only a modeling problem. It is also a data, software, infrastructure, and operational problem.

---

# Summary

The five major machine learning paradigms address different learning situations.

### Supervised Learning

Learns from labeled examples to predict known targets.

### Unsupervised Learning

Discovers hidden structures and patterns in unlabeled data.

### Semi-Supervised Learning

Combines a limited amount of labeled data with a larger amount of unlabeled data.

### Self-Supervised Learning

Creates learning signals automatically from the input data and is widely useful for representation learning and pretraining.

### Reinforcement Learning

Learns sequential decision-making through interaction with an environment and feedback from rewards.

Understanding these distinctions helps determine:

- What type of data is required
- What learning objective should be used
- Which algorithms are appropriate
- How the system should be evaluated
- What limitations may arise
- How the resulting model can be deployed effectively

A strong machine learning practitioner should not begin by selecting an algorithm. The process should begin by understanding the problem, the available data, the feedback mechanism, the deployment environment, and the desired outcome.

The algorithm should follow from those requirements.
