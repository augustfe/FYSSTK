import autograd.numpy as np
from autograd import elementwise_grad, grad
from typing import Callable, Optional
from Activators import sigmoid, derivate
from CostFuncs import CostCrossEntropy, CostOLS
from Schedules import Scheduler
from sklearn.utils import resample
from copy import deepcopy


class NeuralNet:
    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = sigmoid,
        cost_func: Callable = CostOLS,
        seed: Optional[int] = None,
    ):
        if not isinstance(dimensions, tuple):
            raise TypeError(f"Dimensions must be tuple, not {type(dimensions)}")
        if not all(isinstance(layer, int) for layer in dimensions):
            raise TypeError(f"Values of dimensions must be ints, not {dimensions}")
        if not isinstance(seed, int) or seed is not None:
            raise TypeError(f"Seed must be either None or int, not {type(seed)}")
        if dimensions <= 0:
            raise ValueError(f"Number of dimensions must be positive, not {dimensions}")

        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func

        self.seed = seed

        self.z_layers = list()
        self.a_layers = list()
        self.classification = None

        self.reset_weights()

    def reset_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            # Weights
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )

            # Bias
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def fit(
        self,
        X_train: np.ndarray,
        target_train: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lmbda: float = 0,
        X_val: Optional[np.ndarray] = None,
        target_val: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        # Handle TypeErrors (arrays are iffy with jax etc.)
        if not isinstance(scheduler, Scheduler):
            raise TypeError("The scheduler must be of class Scheduler")
        if not isinstance(batches, int):
            raise TypeError(f"Number of batches must be int, not {type(batches)}")
        if not isinstance(epochs, int):
            raise TypeError(f"Number of epochs must be int, not {type(epochs)}")
        if not isinstance(lmbda, (int, float)):
            raise TypeError(f"lmbda must be a number, not {type(lmbda)}")

        # Handle ValueErrors
        if batches <= 0:
            raise ValueError(f"Number of batches must be positive, not {batches}")
        if epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, not {epochs}")
        if lmbda < 0:
            raise ValueError(f"lmbda cannot be negative, {lmbda=}")

        # Set random seed to exclude source of error
        if self.seed is not None:
            np.random.seed(self.seed)

        # Validate if available
        validate = False
        if X_val is not None and target_val is not None:
            validate = True

        # Training metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)

        train_accuracies = np.empty(epochs)
        train_accuracies.fill(np.nan)

        self.schedulers_weight: list[Scheduler] = list()
        self.schedulers_bias: list[Scheduler] = list()

        if X_train.shape[0] < batches:
            raise ValueError(
                f"Number of batches cannot exceed training points, {X_train.shape[0]} < {batches}"
            )

        batch_size = X_train.shape[0] // batches

        # One step of bootstrap
        X_train, target_train = resample(X_train, target_train)

        cost_function_train = self.cost_func(target_train)
        if validate:
            cost_function_validate = self.cost_func(target_val)

            # Validation metrics
            validation_errors = np.empty(epochs)
            validation_errors.fill(np.nan)

            validation_accuracies = np.empty(epochs)
            validation_accuracies.fill(np.nan)

        for i in range(len(self.weights)):
            # I believe deepcopy is necessary to ensure layers are not cross contaminated
            self.schedulers_weight.append(deepcopy(scheduler))
            self.schedulers_bias.append(deepcopy(scheduler))

        data_indices = np.arange(X_train.shape[0])

        for e in range(epochs):
            for i in range(batches):
                # Draw with replacement
                batch_idx = np.random.choice(data_indices, batch_size)

                X_batch = X_train[batch_idx, :]
                target_batch = target_train[batch_idx]

                self.feed_forward(X_batch)
                self.back_propagate(X_batch, target_batch, lmbda)

            # Reset schedulers
            for weight_scheduler, bias_scheduler in zip(
                self.schedulers_weight, self.schedulers_bias
            ):
                weight_scheduler.reset()
                bias_scheduler.reset()

            train_error = cost_function_train(X_train)
            train_errors[e] = train_error

            if validate:
                validation_error = cost_function_validate(X_val)
                validation_errors[e] = validation_error

            if self.classification:
                pred_train = self.predict(X_train)
                train_accuracy = self.accuracy(pred_train, target_train)
                train_accuracies[e] = train_accuracy

            if validate and self.classification:
                pred_validate = self.predict(X_val)
                pred_accuracy = self.accuracy(pred_validate, target_val)
                validation_accuracies[e] = pred_accuracy

        scores = {"train_errors": train_errors}
        if validate:
            scores["validation_errors"] = validation_errors
        if self.classification:
            scores["train_accuracy"] = train_accuracies
        if validate and self.classification:
            scores["validation_accuracy"] = validation_accuracies

        return scores

    def feed_forward(self, X_batch: np.ndarray) -> np.ndarray:
        self.a_layers = list()
        self.z_layers = list()

        # Make sure X is a matrix
        if len(X_batch.shape == 1):
            X_batch = X_batch.reshape((1, X_batch.size))

        # Add a bias
        bias = np.ones((X_batch.shape[0], 1))
        X_batch = np.hstack((bias, X_batch))

        a = X_batch
        self.a_layers.append(a)
        self.z_layers.append(a)

        # Feed forward for all but output layer
        for weight in self.weights[:-1]:
            # Calculate z for hidden layers
            z = a @ weight
            self.z_layers.append(z)
            # Activate layer
            a = self.hidden_func(z)

            # Add bias layer
            bias = np.ones((a.shape[0], 1)) * 0.01
            a = np.hstack((bias, a))
            self.a_layers.append(a)

        # Output layer
        z = a @ self.weights[-1]
        a = self.output_func(z)
        self.a_layers.append(a)
        self.z_layers.append(z)

        # Return the output layer
        return a

    def back_propagate(
        self, X_batch: np.ndarray, target_batch: np.ndarray, lmbda: float
    ):
        hidden_derivative = derivate(self.hidden_func)
        output_derivative = derivate(self.output_func)

        # Start with output layer
        if self.output_func.__name__ == "softmax":
            delta_matrix = self.a_layers[-1] - target_batch
        else:
            cost_func_derivative = grad(self.cost_func(target_batch))
            delta_matrix = output_derivative(self.z_layers[-1]) * cost_func_derivative(
                self.a_layers[-1]
            )

        # Output gradient
        gradient_weights = self.a_layers[-1][:, 1:].T @ delta_matrix
        gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

        gradient_weights += self.weights[-1][1:, :] * lmbda

        update_matrix = np.vstack(
            [
                self.schedulers_bias[-1].update_change(gradient_bias),
                self.schedulers_weight[-1].update_change(gradient_weights),
            ]
        )

        self.weights[-1] -= update_matrix

        # Back propagate the hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta_matrix = delta_matrix @ self.weights[i + 1][1:].T


class OneLayerNeuralNet:
    def __init__(
        self,
        X_data: np.ndarray,
        Y_data: np.ndarray,
        epochs: int,
        n_hidden_nodes: int,
        n_categories: int,
        batch_size: int,
        eta: float,
        lmbda: float,
        hidden_func: Callable = sigmoid,
        output_func: Callable = sigmoid,
        cost_func: Callable = CostCrossEntropy,
    ):
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_nodes = n_hidden_nodes
        self.n_categories = n_categories

        self.batch_size = batch_size
        self.iterations = min(self.n_inputs // self.batch_size, 1)
        self.epochs = epochs
        self.eta = eta
        self.lmbda = lmbda

        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func

        self.create_weights_and_bias()

    def create_weights_and_bias(self):
        # Weights and bias for the hidden layer
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_nodes)
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01

        # Weights and bias for the output layer
        self.output_weights = np.random.randn(self.n_hidden_nodes, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # Weighted sum over the inputs - $z_h = \sum_{i=1}^F w_{ij}^l x_i + b_{i}^l$
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias
        # Activation function - $a_h = f(z_h)$
        self.a_h = self.hidden_func(self.z_h)

        # Activation values $z$
        self.z_o = self.a_h @ self.output_weights + self.output_bias

        self.a_o = self.output_func(self.z_o)

    def feed_forward_out(self, X: np.ndarray[float]) -> np.ndarray[float]:
        # Weighted sum over the inputs - $z_h = \sum_{i=1}^F w_{ij}^l x_i + b_{i}^l$
        z_h = X @ self.hidden_weights + self.hidden_bias
        # Activation function - $a_h = f(z_h)$
        a_h = self.hidden_func(z_h)

        # Activation values $z$
        z_o = a_h @ self.output_weights + self.output_bias
        a_o = self.output_func(z_o)

        return a_o

    def back_propagation(self):
        # Derivative of functions
        derivative_hidden = elementwise_grad(self.hidden_func)
        derivative_output = elementwise_grad(self.output_func)

        # Derivative of cost function for our current targets
        cost_func_derivative = grad(self.cost_func(self.Y_data))

        # Error for output
        error_output = derivative_output(self.z_o) * cost_func_derivative(self.a_o)

        # Calculate gradients for output layer
        gradient_weights_output = self.a_h.T @ error_output
        gradient_bias_output = np.sum(error_output, axis=0)

        gradient_weights_output += self.lmbda * self.output_weights

        # Schedulers will eventually fit here

        self.output_weights -= self.eta * gradient_weights_output
        self.output_bias -= self.eta * gradient_bias_output

        # Error for hidden layer
        error_hidden = (
            error_output @ self.output_weights.T * derivative_hidden(self.z_h)
        )

        # Calculate gradients for hidden layer
        gradient_weights_hidden = self.X_data.T @ error_hidden
        gradient_bias_hidden = np.sum(error_hidden, axis=0)

        gradient_weights_hidden += self.lmbda * self.hidden_weights

        # Schedulers will eventually fit here
        self.hidden_weights -= self.eta * gradient_weights_hidden
        self.hidden_bias -= self.eta * gradient_bias_hidden

    def train(self):
        data_indices = np.arange(self.n_inputs)
        self.X_data = self.X_data_full
        self.Y_data = self.Y_data_full

        train_errors = np.zeros(self.epochs)

        for i in range(self.epochs):
            # self.feed_forward()
            # self.back_propagation()

            # train_error = cost_func_train(self.X_data)
            # train_errors[i] = train_error

            # We only have four datapoints in this case, otherwise:
            for j in range(self.iterations):
                batch_indices = np.random.choice(data_indices, size=self.batch_size)

                self.X_data = self.X_data_full[batch_indices]
                self.Y_data = self.Y_data_full[batch_indices]

                self.feed_forward()
                self.back_propagation()
        return train_errors

    def score(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return np.mean(Y_true == Y_pred)

    def predict(self, X):
        probabilites = self.feed_forward_out(X)
        return np.where(probabilites > 0.5, 1, 0)
