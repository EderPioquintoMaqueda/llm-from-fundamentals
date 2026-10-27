
"""
Next-token prediction: a minimal numerical demonstration.

This example illustrates token IDs, logits, softmax probabilities,
and next-token selection. It is not a trained language model.
"""

import math


VOCABULARY = {
    0: "movimiento",
    1: "calor",
    2: "luz",
    3: "agua",
}

# Hypothetical scores produced by a model for the next token.
# These values are illustrative, not learned parameters.
LOGITS = [0.5, 3.5, 0.7, -0.2]


def softmax(logits: list[float]) -> list[float]:
    """Convert logits into a probability distribution."""
    if not logits:
        raise ValueError("Logits cannot be empty.")

    # Subtract the maximum to improve numerical stability.
    max_logit = max(logits)
    exponentials = [
        math.exp(value - max_logit)
        for value in logits
    ]
    total = sum(exponentials)

    return [value / total for value in exponentials]


def predict_next_token(
    vocabulary: dict[int, str],
    logits: list[float],
) -> tuple[str, list[float]]:
    """Return the most probable token and all token probabilities."""
    if len(vocabulary) != len(logits):
        raise ValueError(
            "The vocabulary and logits must have the same length."
        )

    probabilities = softmax(logits)
    best_index = max(
        range(len(probabilities)),
        key=probabilities.__getitem__,
    )

    return vocabulary[best_index], probabilities


def main() -> None:
    predicted_token, probabilities = predict_next_token(
        VOCABULARY,
        LOGITS,
    )

    print("Context: The electric motor converts energy into...")
    print("\nNext-token probabilities:")

    for token_id, probability in enumerate(probabilities):
        token = VOCABULARY[token_id]
        print(f"{token:12} {probability:.2%}")

    print(f"\nMost probable next token: {predicted_token}")
    print(f"Probability sum: {sum(probabilities):.6f}")


if __name__ == "__main__":
    main()
