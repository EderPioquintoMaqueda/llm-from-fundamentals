
import numpy as np
import matplotlib.pyplot as plt


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of logits into probabilities.

    Expected shape:
        logits: (number_of_samples, number_of_classes)
    """

    # axis=1 means: find the maximum separately in each row.
    # keepdims=True preserves the shape as (samples, 1), so NumPy
    # can broadcast the subtraction across every class in that row.
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    # np.exp applies the exponential to every matrix element.
    exp_logits = np.exp(shifted_logits)

    # Sum the exponentials across classes, one sample at a time.
    # The resulting shape is (number_of_samples, 1).
    row_totals = np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    # Broadcasting divides every value in a row by that row's total.
    return exp_logits / row_totals


def cross_entropy(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Calculate the average cross-entropy over all samples.

    probabilities has shape (samples, classes).
    targets has shape (samples, classes) and uses one-hot encoding.
    """

    # Clip prevents log(0). It does not change ordinary probabilities
    # unless they are extremely close to zero or one.
    safe_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    # np.log calculates the natural logarithm element by element.
    log_probabilities = np.log(safe_probabilities)

    # For each sample, the one-hot target selects the correct class.
    # Incorrect classes contribute zero to the loss.
    sample_losses = -np.sum(
        targets * log_probabilities,
        axis=1,
    )

    # np.mean averages the loss across the training samples.
    return float(np.mean(sample_losses))


def predict(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Return the predicted class index for every sample."""

    # @ performs matrix multiplication.
    # features @ weights produces one logit per class and sample.
    logits = features @ weights + bias

    # Convert logits into probabilities.
    probabilities = softmax(logits)

    # np.argmax returns the index of the largest probability.
    # axis=1 means select the most likely class in each row.
    return np.argmax(probabilities, axis=1)


def main() -> None:
    # Each row is one sample.
    # Each sample has two numerical features: x1 and x2.
    features = np.array([
        [0.0, 0.2],
        [0.2, 0.1],
        [0.3, 0.4],
        [0.4, 0.2],
        [0.6, 0.5],
        [0.7, 0.3],
        [1.4, 1.2],
        [1.5, 1.7],
        [1.7, 1.4],
        [1.8, 1.9],
        [2.0, 1.6],
        [2.1, 2.0],
    ], dtype=float)

    # Integer labels:
    # class 0 = first group, class 1 = second group.
    # The label is the correct answer, not a model prediction.
    labels = np.array([
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
    ])

    # Convert integer labels to one-hot vectors.
    # np.eye(2) creates a 2-by-2 identity matrix:
    # [[1, 0],
    #  [0, 1]]
    # Indexing it with labels creates one target row per sample.
    targets = np.eye(2)[labels]

    # Get the number of samples and input features from the array shape.
    # features.shape[0] = number of rows.
    # features.shape[1] = number of columns.
    number_of_samples = features.shape[0]
    number_of_features = features.shape[1]
    number_of_classes = 2

    # A fixed seed makes the random initialization reproducible.
    # Re-running the script starts with the same initial weights.
    np.random.seed(42)

    # The weight matrix maps input features to output classes.
    # Shape: (number_of_features, number_of_classes) = (2, 2).
    # Small initial values are a common choice for this simple model.
    weights = 0.01 * np.random.randn(
        number_of_features,
        number_of_classes,
    )

    # One bias per class. Shape: (2,).
    # Start at zero.
    bias = np.zeros(number_of_classes)

    # Training settings.
    learning_rate = 0.1
    epochs = 1000

    # Store the loss at each epoch so we can plot training progress.
    loss_history = []

    for epoch in range(epochs):

        # ---------------------------------------------------------
        # 1. FORWARD PASS: calculate the model's predictions.
        # ---------------------------------------------------------

        # Matrix multiplication:
        # (12 samples, 2 features) @ (2 features, 2 classes)
        # gives a matrix with shape (12 samples, 2 classes).
        logits = features @ weights + bias

        # Convert the two logits for each sample into probabilities.
        probabilities = softmax(logits)

        # Measure the average prediction error.
        loss = cross_entropy(probabilities, targets)

        # Keep this loss value for the learning curve.
        loss_history.append(loss)

        # ---------------------------------------------------------
        # 2. BACKWARD PASS: calculate the gradients.
        # ---------------------------------------------------------

        # For softmax + cross-entropy, the gradient with respect
        # to each logit is probability minus the one-hot target.
        # Shape: (number_of_samples, number_of_classes).
        output_gradient = probabilities - targets

        # Gradient of the weights:
        # features.T has shape (2 features, 12 samples).
        # output_gradient has shape (12 samples, 2 classes).
        # Their product has shape (2 features, 2 classes).
        #
        # Divide by number_of_samples to obtain the average gradient.
        weight_gradient = (
            features.T @ output_gradient
        ) / number_of_samples

        # Gradient of the bias:
        # Sum the logit gradients over samples, leaving one value
        # for each class, then average across samples.
        bias_gradient = np.sum(
            output_gradient,
            axis=0,
        ) / number_of_samples

        # ---------------------------------------------------------
        # 3. UPDATE: move the parameters against their gradients.
        # ---------------------------------------------------------

        # Gradient descent update:
        # new parameter = old parameter - learning rate * gradient.
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    # Calculate final predictions after all training epochs.
    predicted_labels = predict(features, weights, bias)

    # Compare predictions with the correct labels.
    # Each True value counts as one correct prediction.
    correct_predictions = predicted_labels == labels

    # np.mean treats True as 1 and False as 0.
    # Multiplying by 100 converts accuracy into a percentage.
    accuracy = np.mean(correct_predictions) * 100.0

    print("LINEAR CLASSIFIER RESULTS")
    print("-" * 40)
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss:   {loss_history[-1]:.4f}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print("\nLearned weights:")
    print(weights)
    print("\nLearned bias:")
    print(bias)

    # -------------------------------------------------------------
    # 4. VISUALIZE THE TRAINING PROCESS.
    # -------------------------------------------------------------

    # Create a figure with two plots side by side.
    # figsize specifies its width and height in inches.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss versus training epoch.
    axes[0].plot(loss_history)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True)

    # Plot the original samples using their correct classes.
    axes[1].scatter(
        features[labels == 0, 0],
        features[labels == 0, 1],
        label="Class 0",
    )
    axes[1].scatter(
        features[labels == 1, 0],
        features[labels == 1, 1],
        label="Class 1",
    )

    # Create a grid of points covering the plotted feature space.
    # np.linspace generates evenly spaced values.
    x1_values = np.linspace(-0.2, 2.4, 150)
    x2_values = np.linspace(-0.2, 2.4, 150)

    # np.meshgrid builds coordinate matrices from the two vectors.
    grid_x1, grid_x2 = np.meshgrid(x1_values, x2_values)

    # Flatten each grid into a one-dimensional array with ravel().
    # column_stack combines them into (x1, x2) feature pairs.
    grid_features = np.column_stack((
        grid_x1.ravel(),
        grid_x2.ravel(),
    ))

    # Predict the class for every point on the grid.
    grid_predictions = predict(
        grid_features,
        weights,
        bias,
    )

    # Restore the flat predictions to the grid's two-dimensional shape.
    decision_regions = grid_predictions.reshape(grid_x1.shape)

    # Draw the boundary between the predicted classes.
    axes[1].contour(
        grid_x1,
        grid_x2,
        decision_regions,
        levels=[0.5],
    )

    axes[1].set_title("Learned Decision Boundary")
    axes[1].set_xlabel("Feature x1")
    axes[1].set_ylabel("Feature x2")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# Run main() only when executing this script directly.
if __name__ == "__main__":
    main()
