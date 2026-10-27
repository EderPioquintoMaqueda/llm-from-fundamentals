
import numpy as np
import matplotlib.pyplot as plt


def loss(w: float) -> float:
    """Calculate the loss for a single parameter."""
    return (w - 3.0) ** 2


def loss_derivative(w: float) -> float:
    """Calculate the analytical derivative of the loss."""
    return 2.0 * (w - 3.0)


def gradient_descent(
    initial_w: float,
    learning_rate: float,
    iterations: int,
) -> list[tuple[int, float, float]]:
    """Minimize the loss by updating the parameter."""
    w = initial_w
    history = [(0, w, loss(w))]

    for step in range(1, iterations + 1):
        gradient = loss_derivative(w)
        w = w - learning_rate * gradient
        history.append((step, w, loss(w)))

    return history


def main() -> None:
    initial_w = 0.0
    learning_rate = 0.1
    iterations = 30

    history = gradient_descent(
        initial_w=initial_w,
        learning_rate=learning_rate,
        iterations=iterations,
    )

    print("Gradient descent")
    print(f"Initial parameter: {initial_w:.4f}")
    print(f"Learning rate: {learning_rate}")
    print()

    for step, w, current_loss in history:
        if step % 5 == 0 or step == 1:
            gradient = loss_derivative(w)
            print(
                f"Step {step:2d} | "
                f"w = {w:8.5f} | "
                f"loss = {current_loss:10.7f} | "
                f"gradient = {gradient:8.5f}"
            )

    final_w = history[-1][1]
    final_loss = history[-1][2]

    print()
    print(f"Final parameter: {final_w:.6f}")
    print(f"Final loss: {final_loss:.8f}")

    # Plot the loss function and optimization path.
    w_values = np.linspace(-1.0, 6.0, 300)
    loss_values = [loss(w) for w in w_values]

    path_w = [item[1] for item in history]
    path_loss = [item[2] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(w_values, loss_values, label="Loss function")
    axes[0].scatter(
        path_w,
        path_loss,
        label="Optimization steps",
    )
    axes[0].set_xlabel("Parameter w")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Gradient descent minimizes loss")
    axes[0].grid(True)
    axes[0].legend()

    steps = [item[0] for item in history]
    losses = [item[2] for item in history]

    axes[1].plot(steps, losses)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over training iterations")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
