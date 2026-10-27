
def main() -> None:
    # Training example
    x = 2.0
    target = 3.0

    # Initial parameters
    w = 0.5
    b = 0.0

    # Forward pass: calculate the prediction and loss.
    prediction = w * x + b
    error = prediction - target
    loss = error ** 2

    # Backward pass: apply the chain rule.
    dloss_dprediction = 2.0 * error
    dprediction_dw = x
    dprediction_db = 1.0

    dloss_dw = dloss_dprediction * dprediction_dw
    dloss_db = dloss_dprediction * dprediction_db

    print("FORWARD PASS")
    print(f"Input: {x}")
    print(f"Weight: {w}")
    print(f"Bias: {b}")
    print(f"Prediction: {prediction:.4f}")
    print(f"Target: {target:.4f}")
    print(f"Loss: {loss:.4f}")

    print("\nBACKWARD PASS")
    print(
        "dLoss/dPrediction:",
        f"{dloss_dprediction:.4f}",
    )
    print(f"dLoss/dWeight: {dloss_dw:.4f}")
    print(f"dLoss/dBias: {dloss_db:.4f}")

    # Gradient descent update.
    learning_rate = 0.1

    new_w = w - learning_rate * dloss_dw
    new_b = b - learning_rate * dloss_db

    # Recalculate the loss with updated parameters.
    new_prediction = new_w * x + new_b
    new_loss = (new_prediction - target) ** 2

    print("\nPARAMETER UPDATE")
    print(f"New weight: {new_w:.4f}")
    print(f"New bias: {new_b:.4f}")
    print(f"New prediction: {new_prediction:.4f}")
    print(f"New loss: {new_loss:.4f}")


if __name__ == "__main__":
    main()
