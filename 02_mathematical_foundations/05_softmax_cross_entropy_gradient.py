
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw model scores (logits) into probabilities.

    np.ndarray is a type annotation: it tells the reader that
    this function expects and returns a NumPy array.
    """

    # np.max(logits) returns the largest value in the array.
    # Subtracting it helps prevent numerical overflow.
    shifted_logits = logits - np.max(logits)

    # np.exp applies the exponential function to each array element.
    exp_logits = np.exp(shifted_logits)

    # np.sum adds every element in the array.
    # Dividing by the total makes the values sum to 1.
    probabilities = exp_logits / np.sum(exp_logits)

    return probabilities


def cross_entropy(
    probabilities: np.ndarray,
    target_index: int,
) -> float:
    """
    Calculate the loss for one correct class.

    target_index is the position of the correct class.
    float indicates that the function returns a single number.
    """

    # Array indexing starts at 0 in Python.
    # If target_index is 0, this selects the first probability.
    correct_probability = probabilities[target_index]

    # np.clip keeps the probability above a tiny positive value.
    # This avoids attempting to calculate log(0).
    safe_probability = np.clip(correct_probability, 1e-12, 1.0)

    # np.log calculates the natural logarithm.
    # The minus sign makes the loss positive or zero.
    loss = -np.log(safe_probability)

    # Convert the NumPy numerical value into a standard Python float.
    return float(loss)


def calculate_gradient(
    probabilities: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """
    Calculate the gradient of softmax + cross-entropy
    with respect to the logits.

    Mathematical result:
        gradient = probabilities - one_hot_target
    """

    # np.zeros_like creates an array of zeros with the same
    # shape as probabilities.
    one_hot_target = np.zeros_like(probabilities)

    # Set the correct class to 1.
    # All incorrect classes remain 0.
    one_hot_target[target_index] = 1.0

    # Subtract the target vector from the predicted probabilities.
    gradient = probabilities - one_hot_target

    return gradient


def numerical_gradient(
    logits: np.ndarray,
    target_index: int,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    Estimate each derivative using a small change in one logit.

    This is a numerical check, not the formula used for training.
    """

    # Create an array to store one approximate derivative per logit.
    gradient = np.zeros_like(logits)

    # enumerate returns both the index and the value of each element.
    # We need the index to modify one logit at a time.
    for index, _ in enumerate(logits):

        # .copy() creates a separate array so the original logits
        # are not modified by these experiments.
        logits_plus = logits.copy()
        logits_minus = logits.copy()

        # Increase this logit by a tiny amount.
        logits_plus[index] += epsilon

        # Decrease the same logit by the same amount.
        logits_minus[index] -= epsilon

        # Calculate the loss for each slightly modified array.
        loss_plus = cross_entropy(
            softmax(logits_plus),
            target_index,
        )

        loss_minus = cross_entropy(
            softmax(logits_minus),
            target_index,
        )

        # Central difference approximation:
        # [f(x + epsilon) - f(x - epsilon)] / (2 * epsilon)
        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    return gradient


def main() -> None:
    # Each number is a logit for a candidate next token.
    tokens = ["Paris", "London", "Berlin"]
    logits = np.array([2.5, 0.8, 0.2])

    # The correct answer is the first token, whose index is 0.
    target_index = 0

    # Step 1: transform logits into probabilities.
    probabilities = softmax(logits)

    # Step 2: calculate how wrong the prediction is.
    loss = cross_entropy(probabilities, target_index)

    # Step 3: calculate the exact gradient formula.
    analytical = calculate_gradient(probabilities, target_index)

    # Step 4: estimate the same gradient numerically.
    numerical = numerical_gradient(logits, target_index)

    print("PREDICTION AND GRADIENT")
    print("-" * 55)
    print(f"Correct token: {tokens[target_index]}")
    print(f"Cross-entropy loss: {loss:.6f}\n")

    # zip pairs the corresponding elements from multiple sequences.
    # Here it pairs each token with its logit, probability,
    # analytical gradient, and numerical gradient.
    for token, logit, probability, grad, num_grad in zip(
        tokens,
        logits,
        probabilities,
        analytical,
        numerical,
    ):
        print(
            f"{token:>6} | "
            f"logit={logit:>5.2f} | "
            f"probability={probability:.4f} | "
            f"gradient={grad:+.6f} | "
            f"numerical={num_grad:+.6f}"
        )

    # np.allclose checks whether two arrays are approximately equal.
    # Small differences are expected due to numerical precision.
    gradients_match = np.allclose(
        analytical,
        numerical,
        atol=1e-5,
    )

    print(f"\nDo the gradients match? {gradients_match}")


# This condition means main() runs only when this file is executed
# directly, not when it is imported as a module into another script.
if __name__ == "__main__":
    main()
