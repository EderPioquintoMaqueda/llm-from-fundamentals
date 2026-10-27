
# Next-Token Prediction

## Objective

Explore the numerical process behind next-token prediction using a
small, controlled example implemented in Python.

## Concepts

- Token IDs and vocabulary mappings
- Logits and probability distributions
- Softmax normalization
- Greedy next-token selection
- Numerical stability

## Implementation

The experiment uses hypothetical logits to calculate a probability
distribution over four candidate tokens.

The softmax function converts the logits into non-negative probabilities
that sum to one. The program then selects the token with the highest
probability.

## Limitations

This is a numerical demonstration, not a trained language model.
The logits are manually defined rather than learned from data.

## Key Takeaway

Next-token prediction can be understood as a conditional probability
estimation problem. Training a language model involves learning
parameters that produce useful probability distributions from context.



## Experiment 2: Embeddings and Vector Similarity

### Objective

Understand how tokens can be represented as vectors and how vector
similarity can be calculated using NumPy.

### Concepts

- Token IDs versus vector representations
- Embedding dimensions
- Dot product and vector norms
- Cosine similarity
- Embedding lookup matrices

### Implementation

The Python script defines three illustrative vectors and calculates
their cosine similarity.

The vectors are manually specified to demonstrate the mathematics.
They are not learned embeddings and do not encode actual linguistic
knowledge.

### Key Takeaway

Token IDs identify vocabulary entries, whereas embeddings provide
numerical representations that a neural network can learn and process.

In a Transformer, these initial representations are subsequently
processed to incorporate information from the surrounding context.



## Experiment 3: Matrix Transformations

### Objective

Understand how matrix multiplication transforms numerical
representations from one feature space into another.

### Concepts

- Matrix dimensions and shape compatibility
- Matrix multiplication
- Weight matrices
- Bias vectors
- Linear transformations

### Implementation

The Python script transforms three illustrative vectors
using a manually defined weight matrix and bias vector.

The parameters are fixed and are not learned during training.

### Key Takeaway

A weight matrix combines input features to produce new
representations. Neural networks learn these parameters
during training to improve their predictions.


## Experiment 5: Dot Product

### Objective

Understand the dot product as both a vector operation
and a building block of neural network computations.

### Concepts

- Element-wise multiplication followed by summation
- Geometric interpretation of the dot product
- Relationship between vectors and angles
- Weighted sums in artificial neurons
- Dot products in Transformer attention

### Key Takeaway

The dot product combines corresponding vector components
into a scalar. Neural networks use it to combine input
features with learned weights, and Transformers use it
to calculate query-key attention scores.
