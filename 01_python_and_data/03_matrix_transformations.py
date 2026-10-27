
"""
Matrix operations and linear transformations.

Demonstrates how a weight matrix transforms vectors
from one feature space into another.
"""

import numpy as np


def main() -> None:
    # Three illustrative feature vectors.
    embeddings = np.array([
        [0.8, 0.2, 0.1],
        [0.7, 0.3, 0.2],
        [0.1, 0.2, 0.9],
    ])

    # Transform three input features into two output features.
    weights = np.array([
        [0.5, 0.1],
        [0.2, 0.7],
        [0.4, 0.3],
    ])

    bias = np.array([0.1, -0.1])

    print("Input matrix:")
    print(embeddings)
    print("Input shape:", embeddings.shape)
    print()

    print("Weight matrix:")
    print(weights)
    print("Weight shape:", weights.shape)
    print()

    # Matrix multiplication.
    linear_output = embeddings @ weights

    print("Linear output:")
    print(linear_output)
    print("Output shape:", linear_output.shape)
    print()

    # Add a bias to each row.
    transformed = linear_output + bias

    print("Output after adding bias:")
    print(transformed)
    print()

    # Show the first vector transformation.
    first_vector = embeddings[0]
    first_output = first_vector @ weights + bias

    print("First input vector:", first_vector)
    print("Its transformed vector:", first_output)


if __name__ == "__main__":
    main()
