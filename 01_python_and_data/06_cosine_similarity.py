
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two nonzero vectors."""
    if vector_a.shape != vector_b.shape:
        raise ValueError("Vectors must have the same shape.")

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Cosine similarity is undefined for zero vectors."
        )

    dot_product = np.dot(vector_a, vector_b)

    return float(dot_product / (norm_a * norm_b))


def main() -> None:
    vector_a = np.array([1.0, 2.0])

    examples = {
        "Same direction": np.array([2.0, 4.0]),
        "Perpendicular": np.array([2.0, -1.0]),
        "Opposite direction": np.array([-2.0, -4.0]),
    }

    print("Reference vector:", vector_a)
    print("Magnitude of A:", np.linalg.norm(vector_a))
    print()

    for name, vector_b in examples.items():
        dot_product = np.dot(vector_a, vector_b)
        norm_b = np.linalg.norm(vector_b)
        similarity = cosine_similarity(vector_a, vector_b)

        print(name)
        print("  Vector B:", vector_b)
        print(f"  Dot product: {dot_product:.4f}")
        print(f"  Magnitude of B: {norm_b:.4f}")
        print(f"  Cosine similarity: {similarity:.4f}")
        print()

    # Visualize the vectors in two dimensions.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (name, vector_b) in zip(axes, examples.items()):
        ax.quiver(
            0, 0,
            vector_a[0], vector_a[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Vector A",
        )
        ax.quiver(
            0, 0,
            vector_b[0], vector_b[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Vector B",
        )

        ax.set_title(name)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.axhline(0, linewidth=0.7)
        ax.axvline(0, linewidth=0.7)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
