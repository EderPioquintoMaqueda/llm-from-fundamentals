
"""
Embeddings and vector similarity.

This experiment uses manually defined vectors to demonstrate
vector operations. These are not embeddings learned by an LLM.
"""

import numpy as np


EMBEDDINGS = {
    "motor": np.array([0.8, 0.2, 0.1]),
    "maquina": np.array([0.7, 0.3, 0.2]),
    "manzana": np.array([0.1, 0.2, 0.9]),
}


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )


def main() -> None:
    motor = EMBEDDINGS["motor"]

    print("Motor embedding:", motor)
    print("Vector dimensions:", motor.shape[0])
    print()

    for word, vector in EMBEDDINGS.items():
        similarity = cosine_similarity(motor, vector)
        print(f"Similarity(motor, {word}): {similarity:.4f}")


if __name__ == "__main__":
    main()
