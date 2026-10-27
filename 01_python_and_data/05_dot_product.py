
import numpy as np


def dot_product(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate the dot product of two equal-length vectors."""
    if vector_a.shape != vector_b.shape:
        raise ValueError("Vectors must have the same shape.")

    return float(np.dot(vector_a, vector_b))


def main() -> None:
    vector_a = np.array([2.0, 3.0])
    vector_b = np.array([4.0, 1.0])

    result = dot_product(vector_a, vector_b)

    print("Vector A:", vector_a)
    print("Vector B:", vector_b)
    print("Dot product:", result)

    # Compare with the manual calculation.
    manual_result = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )
    print("Manual calculation:", manual_result)

    # Show how changing direction affects the result.
    examples = {
        "Similar direction": np.array([1.0, 1.0]),
        "Perpendicular": np.array([1.0, -1.0]),
        "Opposite direction": np.array([-1.0, -1.0]),
    }

    reference = np.array([1.0, 1.0])

    print("\nDirectional examples:")
    for name, vector in examples.items():
        print(
            f"{name}: "
            f"{np.dot(reference, vector):.2f}"
        )


if __name__ == "__main__":
    main()
