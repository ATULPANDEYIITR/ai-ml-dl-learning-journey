"""
====================================================================
INTRODUCTION TO ARTIFICIAL INTELLIGENCE
====================================================================

Topic:
    Introduction to AI

Coverage:
    1. What is Artificial Intelligence?
    2. Intelligence and machine intelligence
    3. AI terminology
    4. History of AI
    5. AI vs ML vs DL
    6. AI taxonomy
    7. Narrow AI / ANI
    8. Artificial General Intelligence / AGI
    9. Artificial Superintelligence / ASI
    10. Symbolic AI
    11. Machine Learning
    12. Deep Learning
    13. Generative AI
    14. Discriminative vs generative systems
    15. Reactive vs learning systems
    16. Rule-based AI
    17. Search and planning
    18. Knowledge representation
    19. Reasoning
    20. Perception
    21. Natural Language Processing
    22. Computer Vision
    23. Speech AI
    24. Robotics
    25. Recommendation systems
    26. Autonomous systems
    27. AI applications
    28. AI capabilities
    29. AI limitations
    30. AI failure modes
    31. AI evaluation
    32. Human intelligence vs AI
    33. AI myths
    34. AI terminology quiz
    35. Small Python demonstrations
    36. Mini AI decision system
    37. Rule-based chatbot
    38. Search algorithm
    39. Simple classifier
    40. AI learning roadmap

Requirements:
    Python 3.x

Works in:
    - Jupyter Notebook
    - Google Colab
    - VS Code
    - Standard Python interpreter

====================================================================
"""


# ==================================================================
# SECTION 0: ENVIRONMENT
# ==================================================================

print("=" * 70)
print("INTRODUCTION TO ARTIFICIAL INTELLIGENCE")
print("=" * 70)

print("""
This program is a learning laboratory for understanding Artificial
Intelligence from beginner concepts to advanced conceptual ideas.
""")


# ==================================================================
# SECTION 1: WHAT IS ARTIFICIAL INTELLIGENCE?
# ==================================================================

print("\n" + "=" * 70)
print("1. WHAT IS ARTIFICIAL INTELLIGENCE?")
print("=" * 70)

definition = """
Artificial Intelligence (AI) is the field of computing concerned with
building systems that can perform tasks that normally require some
forms of human intelligence.

Examples include:

- recognizing images
- understanding language
- making predictions
- solving problems
- planning actions
- recommending products
- detecting anomalies
- generating text
- generating images
- controlling robots
- playing games
"""

print(definition)


# Important conceptual distinction
print("""
AI is NOT simply:

    "A computer that does something complicated."

A better conceptualization is:

    AI = computational systems designed to perform
         intelligent-seeming tasks.

The word "intelligence" is difficult to define precisely.

Human intelligence includes capabilities such as:

- perception
- memory
- learning
- reasoning
- planning
- problem solving
- language
- adaptation
- decision making
- creativity
""")


# ==================================================================
# SECTION 2: INTELLIGENCE
# ==================================================================

print("\n" + "=" * 70)
print("2. WHAT DO WE MEAN BY INTELLIGENCE?")
print("=" * 70)

intelligence_components = [
    "Perception",
    "Learning",
    "Memory",
    "Reasoning",
    "Planning",
    "Problem solving",
    "Language understanding",
    "Decision making",
    "Adaptation",
    "Generalization",
]

for i, component in enumerate(intelligence_components, 1):
    print(f"{i:2}. {component}")


print("""
A key AI question is:

    Can intelligence be implemented computationally?

This question motivated decades of research in:

- mathematics
- computer science
- neuroscience
- psychology
- linguistics
- statistics
- philosophy
- robotics
""")


# ==================================================================
# SECTION 3: AI TERMINOLOGY
# ==================================================================

print("\n" + "=" * 70)
print("3. IMPORTANT AI TERMINOLOGY")
print("=" * 70)

ai_terms = {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "NLP": "Natural Language Processing",
    "CV": "Computer Vision",
    "RL": "Reinforcement Learning",
    "AGI": "Artificial General Intelligence",
    "ASI": "Artificial Superintelligence",
    "ANI": "Artificial Narrow Intelligence",
    "LLM": "Large Language Model",
}

for acronym, meaning in ai_terms.items():
    print(f"{acronym:5} -> {meaning}")


# ==================================================================
# SECTION 4: AI VS ML VS DL
# ==================================================================

print("\n" + "=" * 70)
print("4. AI VS ML VS DL")
print("=" * 70)

print("""
A useful hierarchy is:

                    ARTIFICIAL INTELLIGENCE
                             |
             +---------------+---------------+
             |                               |
        Symbolic AI                    Machine Learning
                                             |
                                  +----------+----------+
                                  |                     |
                            Traditional ML        Deep Learning
                                                        |
                                             Neural Networks
                                                        |
                                      CNNs / RNNs / Transformers
""")


print("""
Important:

    AI is the broadest concept.

    ML is one approach to AI.

    DL is a specialized family of ML methods.

Therefore:

    Deep Learning ⊂ Machine Learning ⊂ Artificial Intelligence

But not every AI system uses machine learning.
""")


# ==================================================================
# SECTION 5: EXAMPLE OF AI WITHOUT ML
# ==================================================================

print("\n" + "=" * 70)
print("5. AI WITHOUT MACHINE LEARNING")
print("=" * 70)


def temperature_controller(temperature):
    """
    Simple rule-based intelligent system.
    """

    if temperature < 18:
        return "Turn heater ON"

    elif temperature > 25:
        return "Turn air conditioner ON"

    else:
        return "Maintain current state"


temperatures = [12, 17, 21, 24, 28, 35]

for temperature in temperatures:
    decision = temperature_controller(temperature)
    print(f"Temperature: {temperature}°C -> {decision}")


print("""
This system demonstrates a basic form of automated decision making.

It does not learn from data.

Its behavior comes from explicitly written rules.

This is often associated with symbolic or rule-based AI.
""")


# ==================================================================
# SECTION 6: AI HISTORY
# ==================================================================

print("\n" + "=" * 70)
print("6. HISTORY OF ARTIFICIAL INTELLIGENCE")
print("=" * 70)

history = [
    ("1940s", "Early computational models and neural ideas"),
    ("1950", "Alan Turing publishes ideas about machine intelligence"),
    ("1956", "Dartmouth workshop popularizes the term Artificial Intelligence"),
    ("1950s-60s", "Early symbolic AI, search, theorem proving"),
    ("1960s", "Early natural language and robotics experiments"),
    ("1970s", "Knowledge-based systems and expert systems"),
    ("1970s-80s", "AI funding and expectations fluctuate"),
    ("1980s", "Expert systems become commercially important"),
    ("Late 1980s-90s", "AI winter and shift toward statistical approaches"),
    ("1990s", "Machine learning becomes increasingly important"),
    ("1997", "IBM Deep Blue defeats chess champion Garry Kasparov"),
    ("2000s", "Large datasets and computational power accelerate ML"),
    ("2012", "Deep learning breakthrough in image recognition"),
    ("2016", "AlphaGo defeats Lee Sedol"),
    ("2017", "Transformer architecture is introduced"),
    ("2020s", "Large language models and generative AI become mainstream"),
]

for period, event in history:
    print(f"{period:12} | {event}")


# ==================================================================
# SECTION 7: ALAN TURING
# ==================================================================

print("\n" + "=" * 70)
print("7. ALAN TURING AND MACHINE INTELLIGENCE")
print("=" * 70)

print("""
Alan Turing was one of the foundational figures in theoretical
computer science and machine intelligence.

In 1950, he published:

    "Computing Machinery and Intelligence"

One famous idea from this work is the imitation game, later known
as the Turing Test.

The central question was essentially:

    "Can machines think?"

Turing suggested replacing the difficult question "Can machines think?"
with an operational test involving communication between humans
and machines.
""")


# ==================================================================
# SECTION 8: DARTMOUTH WORKSHOP
# ==================================================================

print("\n" + "=" * 70)
print("8. DARTMOUTH WORKSHOP")
print("=" * 70)

print("""
The 1956 Dartmouth Summer Research Project on Artificial Intelligence
is commonly regarded as a foundational event in AI as a formal field.

Researchers associated with the event included:

- John McCarthy
- Marvin Minsky
- Nathaniel Rochester
- Claude Shannon

The term "Artificial Intelligence" became strongly associated with
this research program.
""")


# ==================================================================
# SECTION 9: SYMBOLIC AI
# ==================================================================

print("\n" + "=" * 70)
print("9. SYMBOLIC AI")
print("=" * 70)

print("""
Symbolic AI represents knowledge explicitly.

For example:

    Human -> mammal
    Dog -> mammal
    Cat -> mammal

Rules can then be written:

    IF animal is a dog
    THEN animal is a mammal

The machine reasons over symbolic representations.

Typical techniques include:

- logic
- rules
- knowledge bases
- theorem proving
- search
- planning
- ontologies
- expert systems
""")


# ==================================================================
# SECTION 10: RULE-BASED REASONING
# ==================================================================

print("\n" + "=" * 70)
print("10. RULE-BASED REASONING")
print("=" * 70)


def diagnose_weather(temperature, raining, humidity):

    if raining:
        return "Carry an umbrella"

    if temperature > 35:
        return "Very hot: stay hydrated"

    if humidity > 80:
        return "High humidity"

    return "Normal outdoor conditions"


examples = [
    (38, False, 50),
    (28, True, 60),
    (30, False, 90),
    (22, False, 40),
]

for temp, rain, humidity in examples:
    print(
        f"Temperature={temp}, "
        f"Rain={rain}, "
        f"Humidity={humidity}"
        f" -> {diagnose_weather(temp, rain, humidity)}"
    )


# ==================================================================
# SECTION 11: MACHINE LEARNING
# ==================================================================

print("\n" + "=" * 70)
print("11. MACHINE LEARNING")
print("=" * 70)

print("""
Machine Learning is a major approach within AI.

Instead of explicitly writing every rule, we provide data and allow
an algorithm to learn patterns or parameters from that data.

Traditional programming:

    Data + Rules -> Output

Machine Learning:

    Data + Desired learning objective -> Model

Then:

    New Data + Model -> Prediction / Decision
""")


# ==================================================================
# SECTION 12: SIMPLE MACHINE LEARNING IDEA
# ==================================================================

print("\n" + "=" * 70)
print("12. SIMPLE LEARNING EXAMPLE")
print("=" * 70)

print("""
Imagine a system that predicts whether a student passes.

Input features:

    hours studied
    attendance
    previous score

Target:

    pass / fail

Example:

    8 hours + 90% attendance + 75 previous score -> PASS

A machine learning algorithm tries to learn the relationship between
the input features and the target.
""")


# ==================================================================
# SECTION 13: DEEP LEARNING
# ==================================================================

print("\n" + "=" * 70)
print("13. DEEP LEARNING")
print("=" * 70)

print("""
Deep Learning uses neural networks with multiple computational layers.

Conceptually:

Input
  |
  v
Layer 1
  |
  v
Layer 2
  |
  v
Layer 3
  |
  v
Output

Deep neural networks can learn increasingly complex representations.

For example, in image recognition:

Pixels
  ↓
Edges
  ↓
Shapes
  ↓
Parts
  ↓
Objects
  ↓
Class prediction
""")


# ==================================================================
# SECTION 14: NEURAL NETWORK CONCEPT
# ==================================================================

print("\n" + "=" * 70)
print("14. SIMPLE NEURAL NETWORK CALCULATION")
print("=" * 70)


def neuron(inputs, weights, bias):
    """
    Simplified neuron.

    z = weighted sum + bias
    """

    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    return weighted_sum + bias


inputs = [2, 3, 4]
weights = [0.5, 0.2, 0.1]
bias = 1

output = neuron(inputs, weights, bias)

print("Inputs :", inputs)
print("Weights:", weights)
print("Bias   :", bias)
print("Neuron output:", output)


print("""
A real neural network contains many such computational units,
activation functions, layers, parameters, and a training process.
""")


# ==================================================================
# SECTION 15: AI TAXONOMY
# ==================================================================

print("\n" + "=" * 70)
print("15. AI TAXONOMY")
print("=" * 70)

taxonomy = {
    "Artificial Narrow Intelligence": "Designed for specific tasks",
    "Artificial General Intelligence": "Hypothetical broad human-level intelligence",
    "Artificial Superintelligence": "Hypothetical intelligence surpassing humans broadly",
}

for category, explanation in taxonomy.items():
    print(f"\n{category}")
    print("-" * len(category))
    print(explanation)


# ==================================================================
# SECTION 16: NARROW AI
# ==================================================================

print("\n" + "=" * 70)
print("16. NARROW AI / ANI")
print("=" * 70)

print("""
Artificial Narrow Intelligence refers to AI designed for a limited
domain or task.

Examples:

- spam detection
- face recognition
- recommendation systems
- fraud detection
- speech recognition
- chess engines
- route planning
- image classification
- machine translation
- medical image analysis

A narrow system can be extremely powerful within its domain while
remaining incapable of performing unrelated tasks.
""")


# ==================================================================
# SECTION 17: GENERAL AI
# ==================================================================

print("\n" + "=" * 70)
print("17. ARTIFICIAL GENERAL INTELLIGENCE")
print("=" * 70)

print("""
AGI is generally used to describe a hypothetical AI system capable
of performing a broad range of intellectual tasks with substantial
generality and adaptability.

Potential characteristics might include:

- learning new tasks
- transferring knowledge
- reasoning across domains
- adapting to unfamiliar situations
- planning
- understanding complex environments
- solving novel problems

There is no universally accepted operational definition of AGI.

Therefore, claims that a system is or is not AGI depend partly on
the definition and evaluation criteria being used.
""")


# ==================================================================
# SECTION 18: SUPERINTELLIGENCE
# ==================================================================

print("\n" + "=" * 70)
print("18. ARTIFICIAL SUPERINTELLIGENCE")
print("=" * 70)

print("""
Artificial Superintelligence, or ASI, is a hypothetical concept.

It refers to an artificial intelligence whose intellectual capabilities
would substantially exceed those of humans across many domains.

ASI is primarily a theoretical and speculative concept.

It should not be confused with:

    "A very good narrow AI."

A chess engine can be superhuman at chess without being
superintelligent in the broad sense.
""")


# ==================================================================
# SECTION 19: AI CAPABILITY DIMENSIONS
# ==================================================================

print("\n" + "=" * 70)
print("19. AI CAPABILITY DIMENSIONS")
print("=" * 70)

capabilities = [
    "Perception",
    "Prediction",
    "Classification",
    "Generation",
    "Reasoning",
    "Planning",
    "Optimization",
    "Learning",
    "Decision making",
    "Interaction",
    "Control",
]

for i, capability in enumerate(capabilities, 1):
    print(f"{i:2}. {capability}")


# ==================================================================
# SECTION 20: AI SUBFIELDS
# ==================================================================

print("\n" + "=" * 70)
print("20. MAJOR AI SUBFIELDS")
print("=" * 70)

subfields = {
    "Machine Learning":
        "Learning patterns from data",
    "Deep Learning":
        "Multi-layer neural network learning",
    "Natural Language Processing":
        "Computational processing of human language",
    "Computer Vision":
        "Understanding visual information",
    "Robotics":
        "Intelligent physical agents",
    "Speech AI":
        "Speech recognition, synthesis and understanding",
    "Knowledge Representation":
        "Representing facts, concepts and relationships",
    "Reasoning":
        "Drawing conclusions from information",
    "Planning":
        "Selecting actions to reach goals",
    "Search":
        "Exploring possible states or solutions",
    "Reinforcement Learning":
        "Learning through interaction and reward",
    "Generative AI":
        "Generating new content from learned patterns",
}

for field, explanation in subfields.items():
    print(f"\n{field}")
    print(f"  {explanation}")


# ==================================================================
# SECTION 21: COMPUTER VISION
# ==================================================================

print("\n" + "=" * 70)
print("21. COMPUTER VISION")
print("=" * 70)

print("""
Computer Vision deals with computational understanding of visual data.

Tasks include:

- image classification
- object detection
- image segmentation
- facial recognition
- optical character recognition
- pose estimation
- image generation
- video understanding

Example:

Input:
    photograph of a road

Possible outputs:
    car
    pedestrian
    traffic light
    lane
    road
""")


# ==================================================================
# SECTION 22: NLP
# ==================================================================

print("\n" + "=" * 70)
print("22. NATURAL LANGUAGE PROCESSING")
print("=" * 70)

print("""
NLP concerns computational processing of human language.

Tasks include:

- text classification
- sentiment analysis
- translation
- summarization
- question answering
- information extraction
- speech-to-text
- text generation
- conversational systems
- language modeling

Modern NLP is heavily influenced by neural networks and
Transformer architectures.
""")


# ==================================================================
# SECTION 23: SPEECH AI
# ==================================================================

print("\n" + "=" * 70)
print("23. SPEECH AI")
print("=" * 70)

speech_tasks = [
    "Automatic Speech Recognition",
    "Speaker Identification",
    "Speaker Diarization",
    "Speech Synthesis",
    "Voice Activity Detection",
    "Emotion / paralinguistic analysis",
]

for task in speech_tasks:
    print("-", task)


# ==================================================================
# SECTION 24: GENERATIVE AI
# ==================================================================

print("\n" + "=" * 70)
print("24. GENERATIVE AI")
print("=" * 70)

print("""
Generative AI refers to systems capable of generating new content.

Examples:

    Text
    Images
    Audio
    Video
    Code
    3D assets

Generative models learn statistical patterns from data and can
produce new samples or sequences.

Examples of model families include:

    autoregressive models
    diffusion models
    variational autoencoders
    generative adversarial networks
    Transformer-based models
""")


# ==================================================================
# SECTION 25: DISCRIMINATIVE VS GENERATIVE
# ==================================================================

print("\n" + "=" * 70)
print("25. DISCRIMINATIVE VS GENERATIVE MODELS")
print("=" * 70)

print("""
A simplified distinction:

Discriminative:
    Learn how to distinguish or predict labels.

Example:
    Is this email spam?

Generative:
    Learn patterns that can support generation or modeling of data.

Example:
    Generate a new email-like sequence.

This distinction has many technical nuances, but it is useful
as an introductory mental model.
""")


# ==================================================================
# SECTION 26: SEARCH
# ==================================================================

print("\n" + "=" * 70)
print("26. SEARCH AS AN AI TECHNIQUE")
print("=" * 70)

print("""
Many AI problems can be expressed as:

    State
      |
      v
    Actions
      |
      v
    New states
      |
      v
    Goal

Examples:

- chess
- maze solving
- route planning
- puzzle solving
- scheduling
""")


# ==================================================================
# SECTION 27: BREADTH-FIRST SEARCH
# ==================================================================

print("\n" + "=" * 70)
print("27. BREADTH-FIRST SEARCH")
print("=" * 70)

from collections import deque


def breadth_first_search(graph, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:

        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbor in graph.get(node, []):
            new_path = path + [neighbor]
            queue.append(new_path)

    return None


graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": ["G"],
    "G": [],
}

path = breadth_first_search(graph, "A", "G")

print("Graph search path:", path)


# ==================================================================
# SECTION 28: PLANNING
# ==================================================================

print("\n" + "=" * 70)
print("28. AI PLANNING")
print("=" * 70)

print("""
Planning means selecting a sequence of actions that can move a
system from an initial state toward a desired goal.

Example:

Initial state:
    Hungry

Goal:
    Eat food

Possible actions:

    find restaurant
    travel to restaurant
    order food
    receive food
    eat food

A planning system must consider:

    states
    actions
    constraints
    costs
    goals
    consequences
""")


# ==================================================================
# SECTION 29: KNOWLEDGE REPRESENTATION
# ==================================================================

print("\n" + "=" * 70)
print("29. KNOWLEDGE REPRESENTATION")
print("=" * 70)

knowledge_base = {
    "Earth": {
        "type": "planet",
        "orbits": "Sun",
    },
    "Moon": {
        "type": "natural satellite",
        "orbits": "Earth",
    },
}

for entity, properties in knowledge_base.items():
    print(entity, "->", properties)


print("""
Knowledge representation allows AI systems to organize information
about entities, attributes, relationships and rules.
""")


# ==================================================================
# SECTION 30: REASONING
# ==================================================================

print("\n" + "=" * 70)
print("30. REASONING")
print("=" * 70)


facts = {
    "Socrates is human": True,
    "Humans are mortal": True,
}

print("Facts:")
for fact, value in facts.items():
    print(f"- {fact}: {value}")

print("""
Classic logical structure:

    All humans are mortal.
    Socrates is human.
    Therefore Socrates is mortal.

AI reasoning can involve:

- deductive reasoning
- inductive reasoning
- probabilistic reasoning
- abductive reasoning
- constraint reasoning
""")


# ==================================================================
# SECTION 31: RECOMMENDATION SYSTEMS
# ==================================================================

print("\n" + "=" * 70)
print("31. RECOMMENDATION SYSTEMS")
print("=" * 70)

print("""
Recommendation systems predict what a user might prefer.

Examples:

- movies
- products
- songs
- videos
- news
- courses

Inputs might include:

    previous interactions
    ratings
    clicks
    purchases
    search behavior
    item characteristics

Output:

    ranked recommendations
""")


# ==================================================================
# SECTION 32: SIMPLE RECOMMENDER
# ==================================================================

print("\n" + "=" * 70)
print("32. SIMPLE RECOMMENDATION SYSTEM")
print("=" * 70)


items = {
    "Python Course": 4.8,
    "SQL Course": 4.6,
    "AI Course": 4.9,
    "Excel Course": 4.4,
    "Java Course": 4.2,
}

recommendations = sorted(
    items.items(),
    key=lambda x: x[1],
    reverse=True
)

for item, rating in recommendations:
    print(f"{item:20} -> {rating}")


# ==================================================================
# SECTION 33: AI IN DIFFERENT INDUSTRIES
# ==================================================================

print("\n" + "=" * 70)
print("33. AI APPLICATIONS")
print("=" * 70)

applications = {
    "Healthcare": [
        "medical imaging",
        "clinical decision support",
        "drug discovery",
        "patient monitoring",
    ],

    "Finance": [
        "fraud detection",
        "credit risk",
        "algorithmic trading",
        "customer support",
    ],

    "Retail": [
        "recommendations",
        "demand forecasting",
        "inventory optimization",
        "personalization",
    ],

    "Manufacturing": [
        "predictive maintenance",
        "quality inspection",
        "robotics",
        "process optimization",
    ],

    "Transportation": [
        "route optimization",
        "traffic prediction",
        "driver assistance",
        "autonomous systems",
    ],

    "Education": [
        "adaptive learning",
        "automated feedback",
        "content generation",
        "learning analytics",
    ],

    "Cybersecurity": [
        "anomaly detection",
        "malware classification",
        "phishing detection",
        "threat intelligence",
    ],

    "Agriculture": [
        "crop monitoring",
        "yield prediction",
        "disease detection",
        "precision agriculture",
    ],
}

for industry, use_cases in applications.items():

    print(f"\n{industry}")

    for use_case in use_cases:
        print(f"  - {use_case}")


# ==================================================================
# SECTION 34: AI SYSTEM PIPELINE
# ==================================================================

print("\n" + "=" * 70)
print("34. GENERIC AI SYSTEM PIPELINE")
print("=" * 70)

print("""
A simplified AI system may look like:

                DATA
                  |
                  v
          DATA PREPARATION
                  |
                  v
        REPRESENTATION / FEATURES
                  |
                  v
             MODEL
                  |
                  v
             TRAINING
                  |
                  v
            EVALUATION
                  |
                  v
           DEPLOYMENT
                  |
                  v
             MONITORING
                  |
                  v
              FEEDBACK
                  |
                  +-------> improvement
""")


# ==================================================================
# SECTION 35: TRAINING VS INFERENCE
# ==================================================================

print("\n" + "=" * 70)
print("35. TRAINING VS INFERENCE")
print("=" * 70)

print("""
TRAINING
--------

The model learns parameters or patterns from data.

Example:

    historical emails
        ↓
    learning algorithm
        ↓
    trained spam classifier


INFERENCE
---------

The trained model is used to produce an output.

Example:

    new email
        ↓
    trained spam classifier
        ↓
    spam / not spam


Training and inference are fundamentally different computational
phases.
""")


# ==================================================================
# SECTION 36: AI MODEL
# ==================================================================

print("\n" + "=" * 70)
print("36. WHAT IS AN AI MODEL?")
print("=" * 70)

print("""
A model is a computational representation that captures patterns,
relationships or decision behavior useful for a task.

Depending on the approach, a model might contain:

- parameters
- weights
- rules
- decision structures
- embeddings
- probability distributions
- neural network layers

Training adjusts model parameters according to a learning objective.
""")


# ==================================================================
# SECTION 37: DATA
# ==================================================================

print("\n" + "=" * 70)
print("37. DATA IN AI")
print("=" * 70)

data_types = [
    "Structured data",
    "Tabular data",
    "Text",
    "Images",
    "Audio",
    "Video",
    "Time series",
    "Graphs",
    "Sensor data",
    "Multimodal data",
]

for data_type in data_types:
    print("-", data_type)


print("""
AI quality is strongly influenced by:

    data quality
    data quantity
    data diversity
    labeling quality
    sampling
    representativeness
    distribution
    preprocessing
""")


# ==================================================================
# SECTION 38: FEATURES
# ==================================================================

print("\n" + "=" * 70)
print("38. FEATURES")
print("=" * 70)

print("""
A feature is an input variable or representation used by a model.

Example:

House price prediction:

    area
    bedrooms
    location
    age
    distance from city center

These can serve as features.

Modern deep learning systems can learn useful representations
directly from raw or minimally processed data.
""")


# ==================================================================
# SECTION 39: LABELS
# ==================================================================

print("\n" + "=" * 70)
print("39. LABELS")
print("=" * 70)

print("""
In supervised learning, a label represents the target output
associated with a training example.

Example:

    Image -> Cat

Here:

    Image = input
    Cat   = label
""")


# ==================================================================
# SECTION 40: SUPERVISED / UNSUPERVISED / REINFORCEMENT
# ==================================================================

print("\n" + "=" * 70)
print("40. LEARNING PARADIGMS")
print("=" * 70)

learning_paradigms = {
    "Supervised Learning":
        "Learn from input-output examples",

    "Unsupervised Learning":
        "Discover patterns or structure without explicit labels",

    "Self-Supervised Learning":
        "Create learning signals from the data itself",

    "Reinforcement Learning":
        "Learn through actions, states and rewards",
}

for paradigm, explanation in learning_paradigms.items():
    print(f"\n{paradigm}")
    print(f"  {explanation}")


# ==================================================================
# SECTION 41: CLASSIFICATION
# ==================================================================

print("\n" + "=" * 70)
print("41. CLASSIFICATION")
print("=" * 70)


def classify_score(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 50:
        return "Pass"

    return "Fail"


scores = [35, 49, 50, 68, 76, 91]

for score in scores:
    print(f"Score {score} -> {classify_score(score)}")


# ==================================================================
# SECTION 42: PREDICTION
# ==================================================================

print("\n" + "=" * 70)
print("42. PREDICTION")
print("=" * 70)

print("""
AI systems can produce predictions such as:

    probability of fraud
    expected demand
    likelihood of churn
    predicted price
    estimated risk
    classification label

Prediction does not necessarily mean certainty.

A model might output:

    Fraud probability = 0.82

This means the model estimates a high probability according to
its learned representation and calibration.
""")


# ==================================================================
# SECTION 43: PROBABILITY
# ==================================================================

print("\n" + "=" * 70)
print("43. AI AND PROBABILITY")
print("=" * 70)

probabilities = {
    "Spam": 0.91,
    "Not Spam": 0.09,
}

for label, probability in probabilities.items():
    print(f"{label:12}: {probability:.2%}")


print("""
Many AI systems operate probabilistically.

This introduces an important distinction:

    Prediction != certainty

AI outputs can be:

    uncertain
    noisy
    probabilistic
    context-dependent
""")


# ==================================================================
# SECTION 44: HALLUCINATION
# ==================================================================

print("\n" + "=" * 70)
print("44. AI HALLUCINATION")
print("=" * 70)

print("""
In generative AI, a hallucination generally refers to generated
content that is unsupported, incorrect or fabricated despite being
presented as plausible.

Example:

User:
    Who invented a fictional technology?

Model:
    It might confidently invent a person, date and paper.

The important lesson:

    Fluency does not guarantee truth.

Therefore AI systems may need:

    retrieval
    verification
    citations
    tool use
    human review
    grounding
""")


# ==================================================================
# SECTION 45: BIAS
# ==================================================================

print("\n" + "=" * 70)
print("45. AI BIAS")
print("=" * 70)

print("""
AI systems can reproduce or amplify biases present in:

- training data
- labels
- sampling
- feature selection
- system design
- evaluation methodology
- deployment environment

Therefore:

    Model accuracy alone is not enough.

A responsible evaluation may need to consider:

    fairness
    robustness
    calibration
    subgroup performance
    safety
    privacy
""")


# ==================================================================
# SECTION 46: OVERFITTING CONCEPT
# ==================================================================

print("\n" + "=" * 70)
print("46. OVERFITTING")
print("=" * 70)

print("""
Overfitting occurs when a model learns training data too specifically
and performs poorly on unseen data.

Conceptually:

Training performance:
    Very high

Test performance:
    Poor

The model has learned patterns that do not generalize well.

This leads to one of the most important concepts in ML:

    GENERALIZATION
""")


# ==================================================================
# SECTION 47: GENERALIZATION
# ==================================================================

print("\n" + "=" * 70)
print("47. GENERALIZATION")
print("=" * 70)

print("""
Generalization means the ability of a model to perform effectively
on new examples that were not part of its training data.

A powerful AI system must not simply memorize.

It must capture useful patterns that transfer to relevant
unseen situations.
""")


# ==================================================================
# SECTION 48: ROBUSTNESS
# ==================================================================

print("\n" + "=" * 70)
print("48. ROBUSTNESS")
print("=" * 70)

print("""
Robustness is the ability of an AI system to maintain useful
performance when inputs change, become noisy, or differ from
expected conditions.

Examples:

    lighting changes
    spelling mistakes
    background noise
    sensor noise
    unusual phrasing
    distribution shifts
""")


# ==================================================================
# SECTION 49: DISTRIBUTION SHIFT
# ==================================================================

print("\n" + "=" * 70)
print("49. DISTRIBUTION SHIFT")
print("=" * 70)

print("""
AI systems often assume that future data resembles training data.

But real-world distributions can change.

Example:

Training:
    historical customer behavior

Deployment:
    customer behavior after a major market change

If the underlying distribution changes substantially,
model performance can deteriorate.
""")


# ==================================================================
# SECTION 50: AI EVALUATION
# ==================================================================

print("\n" + "=" * 70)
print("50. AI EVALUATION")
print("=" * 70)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 score",
    "ROC-AUC",
    "Mean Absolute Error",
    "Mean Squared Error",
    "BLEU",
    "ROUGE",
    "Perplexity",
    "Latency",
    "Throughput",
    "Robustness",
    "Fairness",
]

for metric in metrics:
    print("-", metric)


print("""
Different AI tasks require different evaluation strategies.

A single metric rarely captures the complete quality of an AI system.
""")


# ==================================================================
# SECTION 51: AI VS HUMAN INTELLIGENCE
# ==================================================================

print("\n" + "=" * 70)
print("51. AI VS HUMAN INTELLIGENCE")
print("=" * 70)

comparison = {
    "Speed": "Machines can process certain computations extremely quickly",
    "Memory": "Machines can store and retrieve enormous amounts of data",
    "Adaptability": "Humans are generally highly adaptable across diverse contexts",
    "Embodiment": "Humans experience the physical world directly",
    "Learning": "Both humans and machines can learn, but mechanisms differ",
    "Common sense": "Human common-sense reasoning remains difficult to reproduce reliably",
    "Creativity": "Both can produce novel outputs, but the underlying processes differ",
}

for dimension, explanation in comparison.items():
    print(f"\n{dimension}:")
    print(f"  {explanation}")


# ==================================================================
# SECTION 52: AI AGENT CONCEPT
# ==================================================================

print("\n" + "=" * 70)
print("52. AI AGENTS")
print("=" * 70)

print("""
An AI agent can be conceptualized as a system that:

    perceives
       ↓
    interprets
       ↓
    reasons
       ↓
    decides
       ↓
    acts
       ↓
    observes consequences
       ↓
    repeats

Examples:

- robot
- game-playing agent
- autonomous vehicle controller
- software agent
- task-oriented AI system
""")


# ==================================================================
# SECTION 53: ENVIRONMENT
# ==================================================================

print("\n" + "=" * 70)
print("53. AGENT + ENVIRONMENT")
print("=" * 70)

print("""
Agent <---------------- Environment
  |                         ^
  |                         |
  +-------- Action ---------+
            |
        Observation

The environment produces observations.

The agent selects actions.

This interaction is fundamental to reinforcement learning
and intelligent-agent design.
""")


# ==================================================================
# SECTION 54: SIMPLE AI AGENT
# ==================================================================

print("\n" + "=" * 70)
print("54. SIMPLE AI AGENT")
print("=" * 70)


def thermostat_agent(temperature):

    if temperature < 20:
        action = "HEAT"

    elif temperature > 24:
        action = "COOL"

    else:
        action = "DO NOTHING"

    return action


environment_temperatures = [16, 19, 21, 23, 27]

for temp in environment_temperatures:
    action = thermostat_agent(temp)

    print(
        f"Environment temperature={temp}°C "
        f"-> Agent action={action}"
    )


# ==================================================================
# SECTION 55: OPTIMIZATION
# ==================================================================

print("\n" + "=" * 70)
print("55. OPTIMIZATION")
print("=" * 70)

print("""
Many AI problems can be expressed as optimization problems.

General form:

    minimize or maximize:

        Objective Function

subject to:

        Constraints

Examples:

    minimize delivery time
    minimize prediction error
    maximize recommendation relevance
    maximize reward
    minimize cost
""")


# ==================================================================
# SECTION 56: SIMPLE OPTIMIZATION
# ==================================================================

print("\n" + "=" * 70)
print("56. SIMPLE OPTIMIZATION EXAMPLE")
print("=" * 70)


options = {
    "Route A": 45,
    "Route B": 32,
    "Route C": 51,
    "Route D": 39,
}

best_route = min(options, key=options.get)

print("Available routes:")

for route, time in options.items():
    print(f"{route}: {time} minutes")

print("\nBest route:", best_route)
print("Travel time:", options[best_route], "minutes")


# ==================================================================
# SECTION 57: AI DECISION SYSTEM
# ==================================================================

print("\n" + "=" * 70)
print("57. MINI AI DECISION SYSTEM")
print("=" * 70)


def loan_risk_system(income, debt, credit_score):

    risk_score = 0

    if income < 30000:
        risk_score += 30

    if debt > 50000:
        risk_score += 30

    if credit_score < 600:
        risk_score += 40

    if risk_score >= 70:
        decision = "HIGH RISK"

    elif risk_score >= 40:
        decision = "MEDIUM RISK"

    else:
        decision = "LOW RISK"

    return risk_score, decision


customers = [
    {
        "name": "Customer A",
        "income": 70000,
        "debt": 20000,
        "credit_score": 750,
    },

    {
        "name": "Customer B",
        "income": 25000,
        "debt": 70000,
        "credit_score": 550,
    },

    {
        "name": "Customer C",
        "income": 45000,
        "debt": 40000,
        "credit_score": 650,
    },
]


for customer in customers:

    score, decision = loan_risk_system(
        customer["income"],
        customer["debt"],
        customer["credit_score"],
    )

    print(
        f"{customer['name']}: "
        f"risk_score={score}, "
        f"decision={decision}"
    )


print("""
IMPORTANT:

This is NOT a production-grade credit model.

It is a conceptual demonstration of:

    inputs
      ↓
    rules
      ↓
    score
      ↓
    decision

Real financial AI requires rigorous validation, fairness analysis,
regulatory compliance, explainability, security and monitoring.
""")


# ==================================================================
# SECTION 58: SIMPLE CHATBOT
# ==================================================================

print("\n" + "=" * 70)
print("58. SIMPLE RULE-BASED CHATBOT")
print("=" * 70)


def chatbot(message):

    message = message.lower().strip()

    if "hello" in message or "hi" in message:
        return "Hello! How can I help you?"

    elif "ai" in message:
        return "AI means Artificial Intelligence."

    elif "ml" in message:
        return "ML is a major approach within AI."

    elif "deep learning" in message:
        return "Deep Learning uses multi-layer neural networks."

    elif "bye" in message:
        return "Goodbye!"

    return "I do not understand that yet."


test_messages = [
    "Hello",
    "What is AI?",
    "Tell me about ML",
    "What is deep learning?",
    "Bye",
]

for message in test_messages:

    print("User :", message)
    print("Bot  :", chatbot(message))


print("""
This chatbot demonstrates:

    input
      ↓
    preprocessing
      ↓
    pattern matching
      ↓
    response

It may look intelligent but does not learn.
""")


# ==================================================================
# SECTION 59: AI SYSTEM MATURITY
# ==================================================================

print("\n" + "=" * 70)
print("59. AI SYSTEM MATURITY")
print("=" * 70)

maturity = [
    "Manual rules",
    "Rule-based automation",
    "Statistical models",
    "Machine learning",
    "Deep learning",
    "Foundation models",
    "Generative AI",
    "Agentic systems",
]

for i, stage in enumerate(maturity, 1):
    print(f"{i}. {stage}")


# ==================================================================
# SECTION 60: FOUNDATION MODELS
# ==================================================================

print("\n" + "=" * 70)
print("60. FOUNDATION MODELS")
print("=" * 70)

print("""
A foundation model is a broadly trained model that can serve as
a base for many downstream tasks.

The concept is particularly associated with large-scale models
trained on broad datasets.

Potential capabilities:

    language understanding
    generation
    reasoning-like behavior
    image understanding
    multimodal processing
    code generation

The same underlying model can potentially support many applications.
""")


# ==================================================================
# SECTION 61: LARGE LANGUAGE MODELS
# ==================================================================

print("\n" + "=" * 70)
print("61. LARGE LANGUAGE MODELS")
print("=" * 70)

print("""
A Large Language Model is a language model trained at large scale.

A simplified conceptual pipeline:

    Text
      ↓
    Tokenization
      ↓
    Numerical representations
      ↓
    Neural network
      ↓
    Probability distribution
      ↓
    Next-token prediction
      ↓
    Generated sequence

Modern LLMs commonly use Transformer-based architectures.
""")


# ==================================================================
# SECTION 62: MULTIMODAL AI
# ==================================================================

print("\n" + "=" * 70)
print("62. MULTIMODAL AI")
print("=" * 70)

print("""
Multimodal AI deals with multiple types of information.

Examples:

    text + image
    text + audio
    image + audio
    video + text
    speech + text
    sensor + vision

A multimodal system may need to:

    perceive
    align
    represent
    reason
    generate
""")


# ==================================================================
# SECTION 63: ROBOTICS
# ==================================================================

print("\n" + "=" * 70)
print("63. AI + ROBOTICS")
print("=" * 70)

print("""
Robotics combines AI with physical machines.

A robot may contain:

    sensors
    perception systems
    localization
    planning
    control
    actuators

Conceptual pipeline:

    Sensors
       ↓
    Perception
       ↓
    World model
       ↓
    Planning
       ↓
    Control
       ↓
    Actuators
""")


# ==================================================================
# SECTION 64: AUTONOMOUS SYSTEMS
# ==================================================================

print("\n" + "=" * 70)
print("64. AUTONOMOUS SYSTEMS")
print("=" * 70)

print("""
Autonomous systems operate with varying levels of independence.

Examples:

    autonomous vehicles
    drones
    warehouse robots
    robotic vacuum cleaners
    industrial robots

Autonomy does not necessarily mean complete independence.

Real systems often operate under:

    constraints
    safety rules
    supervision
    fallback mechanisms
    human intervention
""")


# ==================================================================
# SECTION 65: AI SAFETY
# ==================================================================

print("\n" + "=" * 70)
print("65. AI SAFETY")
print("=" * 70)

safety_topics = [
    "Reliability",
    "Robustness",
    "Security",
    "Privacy",
    "Fairness",
    "Interpretability",
    "Human oversight",
    "Monitoring",
    "Misuse prevention",
    "Fail-safe mechanisms",
]

for topic in safety_topics:
    print("-", topic)


# ==================================================================
# SECTION 66: AI SECURITY
# ==================================================================

print("\n" + "=" * 70)
print("66. AI SECURITY")
print("=" * 70)

print("""
AI systems can be attacked or manipulated.

Examples include:

    adversarial examples
    prompt injection
    data poisoning
    model theft
    membership inference
    model extraction
    malicious inputs
    supply-chain attacks

AI security combines:

    cybersecurity
    machine learning
    software security
    data security
""")


# ==================================================================
# SECTION 67: AI LIMITATIONS
# ==================================================================

print("\n" + "=" * 70)
print("67. LIMITATIONS OF AI")
print("=" * 70)

limitations = [
    "Data dependency",
    "Distribution shift",
    "Hallucination",
    "Bias",
    "Limited interpretability",
    "Computational cost",
    "Energy consumption",
    "Security vulnerabilities",
    "Privacy risks",
    "Evaluation difficulties",
    "Lack of reliable common sense",
    "Potential brittleness",
]

for limitation in limitations:
    print("-", limitation)


# ==================================================================
# SECTION 68: AI MYTHS
# ==================================================================

print("\n" + "=" * 70)
print("68. COMMON AI MYTHS")
print("=" * 70)

myths = {
    "AI always tells the truth":
        "False. AI systems can produce incorrect outputs.",

    "AI and ML are identical":
        "False. ML is one major approach within AI.",

    "Deep learning equals all AI":
        "False. AI includes many non-deep-learning approaches.",

    "More parameters automatically means intelligence":
        "False. Capability depends on architecture, data, training and evaluation.",

    "AI understands exactly like humans":
        "This is not established and depends strongly on what understanding means.",

    "A chatbot is automatically AGI":
        "False. Conversational ability alone does not establish general intelligence.",
}

for myth, correction in myths.items():

    print(f"\nClaim: {myth}")
    print(f"Explanation: {correction}")


# ==================================================================
# SECTION 69: AI PROBLEM FORMULATION
# ==================================================================

print("\n" + "=" * 70)
print("69. FORMULATING AN AI PROBLEM")
print("=" * 70)

print("""
When approaching an AI problem, ask:

1. What is the problem?
2. What is the desired output?
3. What data is available?
4. What type of task is this?
5. Is ML actually necessary?
6. What baseline should be used?
7. What metric matters?
8. What are the failure modes?
9. What constraints exist?
10. How will the system be deployed?
11. How will it be monitored?
12. What happens when the system is wrong?
""")


# ==================================================================
# SECTION 70: AI TASK CLASSIFICATION
# ==================================================================

print("\n" + "=" * 70)
print("70. CLASSIFYING AI PROBLEMS")
print("=" * 70)


def classify_ai_problem(problem):

    problem = problem.lower()

    if "spam" in problem or "fraud" in problem:
        return "Classification"

    if "price" in problem or "demand" in problem:
        return "Prediction / Regression / Forecasting"

    if "recommend" in problem:
        return "Recommendation"

    if "generate" in problem or "write" in problem:
        return "Generative AI"

    if "image" in problem:
        return "Computer Vision"

    if "speech" in problem or "voice" in problem:
        return "Speech AI"

    if "route" in problem or "schedule" in problem:
        return "Search / Optimization / Planning"

    return "Requires further problem analysis"


problems = [
    "Detect fraudulent transactions",
    "Predict house price",
    "Recommend movies",
    "Generate a report",
    "Analyze image",
    "Recognize speech",
    "Find shortest route",
]

for problem in problems:

    print(
        f"{problem:35} -> "
        f"{classify_ai_problem(problem)}"
    )


# ==================================================================
# SECTION 71: AI PROJECT LIFECYCLE
# ==================================================================

print("\n" + "=" * 70)
print("71. AI PROJECT LIFECYCLE")
print("=" * 70)

lifecycle = [
    "Problem definition",
    "Data collection",
    "Data validation",
    "Data preparation",
    "Exploratory analysis",
    "Feature / representation design",
    "Baseline",
    "Model selection",
    "Training",
    "Validation",
    "Evaluation",
    "Deployment",
    "Monitoring",
    "Maintenance",
    "Continuous improvement",
]

for step_number, step in enumerate(lifecycle, 1):
    print(f"{step_number:2}. {step}")


# ==================================================================
# SECTION 72: BASELINE
# ==================================================================

print("\n" + "=" * 70)
print("72. WHY BASELINES MATTER")
print("=" * 70)

print("""
Before building a complex AI model, establish a baseline.

Example:

If 90% of transactions are legitimate, a naive classifier that
always predicts "legitimate" could achieve 90% accuracy.

That does NOT mean it is a useful fraud detection system.

Therefore metrics and baselines must be interpreted in context.
""")


# ==================================================================
# SECTION 73: ACCURACY
# ==================================================================

print("\n" + "=" * 70)
print("73. ACCURACY")
print("=" * 70)


def accuracy(correct, total):

    if total == 0:
        return 0

    return correct / total


correct_predictions = 92
total_predictions = 100

print(
    "Accuracy:",
    f"{accuracy(correct_predictions, total_predictions):.2%}"
)


# ==================================================================
# SECTION 74: PRECISION AND RECALL
# ==================================================================

print("\n" + "=" * 70)
print("74. PRECISION AND RECALL")
print("=" * 70)

print("""
Precision:

    Of the items predicted positive,
    how many were actually positive?

Recall:

    Of all actual positive items,
    how many did we successfully detect?

Different applications prioritize different errors.
""")


def precision(tp, fp):

    denominator = tp + fp

    if denominator == 0:
        return 0

    return tp / denominator


def recall(tp, fn):

    denominator = tp + fn

    if denominator == 0:
        return 0

    return tp / denominator


tp = 80
fp = 20
fn = 10

print("Precision:", f"{precision(tp, fp):.2%}")
print("Recall   :", f"{recall(tp, fn):.2%}")


# ==================================================================
# SECTION 75: F1 SCORE
# ==================================================================

print("\n" + "=" * 70)
print("75. F1 SCORE")
print("=" * 70)


def f1_score(p, r):

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)


p = precision(tp, fp)
r = recall(tp, fn)

print("F1 score:", f"{f1_score(p, r):.2%}")


# ==================================================================
# SECTION 76: HUMAN-IN-THE-LOOP
# ==================================================================

print("\n" + "=" * 70)
print("76. HUMAN-IN-THE-LOOP AI")
print("=" * 70)

print("""
Not every AI system should operate completely autonomously.

Human-in-the-loop systems allow humans to:

    review
    approve
    reject
    correct
    override
    escalate

Example:

    AI flags suspicious transaction
              ↓
        Human investigator
              ↓
        Final decision
""")


# ==================================================================
# SECTION 77: EXPLAINABILITY
# ==================================================================

print("\n" + "=" * 70)
print("77. EXPLAINABILITY")
print("=" * 70)

print("""
Explainability asks:

    Why did the system produce this output?

Interpretability can be particularly important in:

    healthcare
    finance
    law
    public services
    safety-critical systems

Not every model is equally easy to explain.

There is often a trade-off between:

    performance
    complexity
    interpretability
""")


# ==================================================================
# SECTION 78: AI GOVERNANCE
# ==================================================================

print("\n" + "=" * 70)
print("78. AI GOVERNANCE")
print("=" * 70)

governance = [
    "Data governance",
    "Model governance",
    "Risk management",
    "Documentation",
    "Auditability",
    "Access control",
    "Privacy",
    "Security",
    "Human oversight",
    "Monitoring",
]

for item in governance:
    print("-", item)


# ==================================================================
# SECTION 79: AI VS AUTOMATION
# ==================================================================

print("\n" + "=" * 70)
print("79. AI VS AUTOMATION")
print("=" * 70)

print("""
Automation:

    A predefined process executes automatically.

AI:

    The system may infer, predict, classify, generate,
    reason, optimize or adapt based on data/models.

Example automation:

    Every day at 9 AM -> send email.

Example AI:

    Analyze incoming customer message -> infer intent ->
    generate or select an appropriate response.

AI and automation overlap, but they are not synonyms.
""")


# ==================================================================
# SECTION 80: FINAL KNOWLEDGE MAP
# ==================================================================

print("\n" + "=" * 70)
print("80. FINAL AI KNOWLEDGE MAP")
print("=" * 70)

knowledge_map = """
ARTIFICIAL INTELLIGENCE
|
+-- Symbolic AI
|   |
|   +-- Rules
|   +-- Logic
|   +-- Search
|   +-- Planning
|   +-- Knowledge Representation
|
+-- Machine Learning
|   |
|   +-- Supervised Learning
|   +-- Unsupervised Learning
|   +-- Self-Supervised Learning
|   +-- Reinforcement Learning
|   |
|   +-- Deep Learning
|       |
|       +-- CNNs
|       +-- RNNs
|       +-- Transformers
|       +-- Foundation Models
|
+-- Applications
|   |
|   +-- NLP
|   +-- Computer Vision
|   +-- Speech
|   +-- Robotics
|   +-- Healthcare
|   +-- Finance
|   +-- Cybersecurity
|   +-- Manufacturing
|   +-- Agriculture
|
+-- Generative AI
|   |
|   +-- Text
|   +-- Images
|   +-- Audio
|   +-- Video
|   +-- Code
|
+-- Advanced Concepts
    |
    +-- Agents
    +-- Multimodal AI
    +-- Reasoning
    +-- Planning
    +-- AI Safety
    +-- AI Security
    +-- AI Governance
"""


print(knowledge_map)


# ==================================================================
# SECTION 81: KNOWLEDGE CHECK
# ==================================================================

print("\n" + "=" * 70)
print("81. KNOWLEDGE CHECK")
print("=" * 70)

questions = [
    (
        "What is the broadest concept: AI, ML or DL?",
        "AI"
    ),

    (
        "Is every AI system a machine learning system?",
        "No"
    ),

    (
        "Is every deep learning system a machine learning system?",
        "Yes"
    ),

    (
        "What does ANI stand for?",
        "Artificial Narrow Intelligence"
    ),

    (
        "What does AGI stand for?",
        "Artificial General Intelligence"
    ),

    (
        "What does ASI stand for?",
        "Artificial Superintelligence"
    ),

    (
        "What is the purpose of training?",
        "Learning model parameters/patterns from data"
    ),

    (
        "What is inference?",
        "Using a trained model to produce outputs"
    ),
]

for question, answer in questions:

    print("\nQuestion:")
    print(question)

    print("Answer:")
    print(answer)


# ==================================================================
# SECTION 82: FINAL SUMMARY
# ==================================================================

print("\n" + "=" * 70)
print("82. FINAL SUMMARY")
print("=" * 70)

summary = """
You should now understand that:

1. AI is a broad field concerned with intelligent computational systems.

2. AI can exist without machine learning.

3. Machine Learning is an important approach within AI.

4. Deep Learning is a subset of Machine Learning.

5. Narrow AI is designed for specific tasks.

6. AGI is a hypothetical concept involving broad general intelligence.

7. ASI is a hypothetical concept involving intelligence that
   substantially exceeds human intelligence across broad domains.

8. Symbolic AI relies heavily on explicit representations, rules,
   logic, search and reasoning.

9. Machine Learning learns patterns from data.

10. Deep Learning uses multi-layer neural networks.

11. Generative AI produces new content.

12. Computer Vision processes visual information.

13. NLP processes human language.

14. Speech AI processes spoken language and audio.

15. Robotics combines intelligent computation with physical machines.

16. AI agents perceive environments, make decisions and act.

17. AI systems can fail because of bias, hallucination, distribution
    shift, data problems, security issues and poor problem formulation.

18. AI evaluation must go beyond a single metric.

19. AI deployment requires monitoring and governance.

20. Building AI is not simply choosing a model.

A real AI system is:

    PROBLEM
       ↓
    DATA
       ↓
    REPRESENTATION
       ↓
    MODEL
       ↓
    TRAINING
       ↓
    EVALUATION
       ↓
    DEPLOYMENT
       ↓
    MONITORING
       ↓
    FEEDBACK
       ↓
    IMPROVEMENT
"""

print(summary)


# ==================================================================
# SECTION 83: LEARNING CHECKLIST
# ==================================================================

print("\n" + "=" * 70)
print("83. LEARNING CHECKLIST")
print("=" * 70)

checklist = [
    "I can define Artificial Intelligence.",
    "I understand why intelligence is difficult to define.",
    "I know the major historical milestones of AI.",
    "I understand AI vs ML vs DL.",
    "I understand symbolic AI.",
    "I understand machine learning conceptually.",
    "I understand deep learning conceptually.",
    "I understand ANI.",
    "I understand the concept of AGI.",
    "I understand the concept of ASI.",
    "I understand NLP.",
    "I understand Computer Vision.",
    "I understand Speech AI.",
    "I understand Generative AI.",
    "I understand AI agents.",
    "I understand search and planning.",
    "I understand recommendation systems.",
    "I understand AI evaluation.",
    "I understand hallucination.",
    "I understand bias.",
    "I understand robustness.",
    "I understand generalization.",
    "I understand training vs inference.",
    "I understand AI limitations.",
    "I understand AI safety and governance.",
]

for item in checklist:
    print("[ ]", item)


# ==================================================================
# SECTION 84: NEXT LEARNING STAGE
# ==================================================================

print("\n" + "=" * 70)
print("84. WHAT TO LEARN NEXT")
print("=" * 70)

next_topics = [
    "Python programming for AI",
    "Mathematics for AI",
    "Linear algebra",
    "Probability and statistics",
    "Calculus",
    "Data structures and algorithms",
    "NumPy",
    "Pandas",
    "Data visualization",
    "Machine Learning",
    "Scikit-learn",
    "Model evaluation",
    "Feature engineering",
    "Deep Learning",
    "PyTorch",
    "TensorFlow",
    "Computer Vision",
    "NLP",
    "Transformers",
    "Generative AI",
    "LLMs",
    "RAG",
    "AI Agents",
    "MLOps",
    "AI Safety",
    "AI Security",
]

for number, topic in enumerate(next_topics, 1):
    print(f"{number:2}. {topic}")


# ==================================================================
# END
# ==================================================================

print("\n" + "=" * 70)
print("END OF INTRODUCTION TO AI")
print("=" * 70)

print("""
Congratulations.

You have completed a conceptual and practical introduction to
Artificial Intelligence.

The next step is not to memorize definitions.

The next step is to implement.

Recommended sequence:

    Python
       ↓
    Mathematics
       ↓
    Data handling
       ↓
    Machine Learning
       ↓
    Deep Learning
       ↓
    NLP / Computer Vision
       ↓
    Generative AI
       ↓
    LLMs
       ↓
    AI Agents
       ↓
    MLOps / AI Systems
       ↓
    Advanced AI Research
""")
