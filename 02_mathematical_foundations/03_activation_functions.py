
import numpy as np
import matplotlib.pyplot as plt


def relu(x: np.ndarray) -> np.ndarray:
    """Apply the ReLU activation function."""
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function."""
    # Clip inputs to avoid numerical overflow in exp().
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic tangent activation function."""
    return np.tanh(x)


def main() -> None:
    x = np.linspace(-6.0, 6.0, 400)

    activations = {
        "ReLU": relu(x),
        "Sigmoid": sigmoid(x),
        "Tanh": tanh(x),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (name, values) in zip(axes, activations.items()):
        ax.plot(x, values)
        ax.axhline(0.0, linewidth=0.7)
        ax.axvline(0.0, linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel("Input z")
        ax.set_ylabel("Activation f(z)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    sample = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    print("Activation values")
    print("Input:   ", sample)
    print("ReLU:    ", relu(sample))
    print("Sigmoid: ", np.round(sigmoid(sample), 4))
    print("Tanh:    ", np.round(tanh(sample), 4))


if __name__ == "__main__":
    main()
