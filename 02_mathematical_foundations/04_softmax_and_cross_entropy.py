
import numpy as np
import matplotlib.pyplot as plt


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits into a probability distribution."""
    # Subtracting the maximum improves numerical stability.
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits)


def cross_entropy(
    probabilities: np.ndarray,
    target_index: int,
) -> float:
    """Calculate cross-entropy for the correct class."""
    if not 0 <= target_index < len(probabilities):
        raise ValueError("target_index is outside the valid range.")

    # Avoid log(0), which is undefined.
    epsilon = 1e-12
    correct_probability = np.clip(
        probabilities[target_index],
        epsilon,
        1.0,
    )

    return float(-np.log(correct_probability))


def main() -> None:
    tokens = ["Paris", "London", "Berlin"]
    target_index = 0  # The correct token is Paris.

    # Example 1: a relatively good prediction.
    good_logits = np.array([2.5, 0.8, 0.2])
    good_probabilities = softmax(good_logits)
    good_loss = cross_entropy(good_probabilities, target_index)

    # Example 2: a prediction that favors the wrong token.
    bad_logits = np.array([0.2, 2.5, 0.8])
    bad_probabilities = softmax(bad_logits)
    bad_loss = cross_entropy(bad_probabilities, target_index)

    print("NEXT-TOKEN PREDICTION")
    print(f"Correct token: {tokens[target_index]}\n")

    for token, logit, probability in zip(
        tokens, good_logits, good_probabilities
    ):
        print(
            f"{token:>6} | "
            f"logit = {logit:>4.1f} | "
            f"probability = {probability:.4f}"
        )

    print(f"\nGood prediction loss: {good_loss:.4f}")
    print(f"Bad prediction loss:  {bad_loss:.4f}")

    # Visualize how cross-entropy changes with the
    # probability assigned to the correct token.
    correct_probabilities = np.linspace(0.01, 1.0, 300)
    losses = -np.log(correct_probabilities)

    plt.figure(figsize=(8, 5))
    plt.plot(correct_probabilities, losses)
    plt.xlabel("Probability assigned to the correct token")
    plt.ylabel("Cross-entropy loss")
    plt.title("Cross-Entropy: Lower Loss Means Better Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
