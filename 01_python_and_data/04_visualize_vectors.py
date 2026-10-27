
import numpy as np
import matplotlib.pyplot as plt


def vector_norm(vector: np.ndarray) -> float:
    """Calculate the Euclidean norm of a vector."""
    return float(np.sqrt(np.sum(vector ** 2)))


def main() -> None:
    vectors = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [-1.0, 1.0],
    ])

    # Stretch the horizontal dimension by a factor of two.
    transformation = np.array([
        [2.0, 0.0],
        [0.0, 1.0],
    ])

    transformed_vectors = vectors @ transformation

    print("Original vectors and their norms:")
    for vector in vectors:
        print(
            f"{vector} -> norm = {vector_norm(vector):.3f}"
        )

    print("\nTransformed vectors and their norms:")
    for vector in transformed_vectors:
        print(
            f"{vector} -> norm = {vector_norm(vector):.3f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    titles = ["Original vectors", "Transformed vectors"]
    datasets = [vectors, transformed_vectors]

    for ax, data, title in zip(axes, datasets, titles):
        for vector in data:
            ax.quiver(
                0, 0,
                vector[0], vector[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.008,
            )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-3, 5)
        ax.set_ylim(-1, 3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
