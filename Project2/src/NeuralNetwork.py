import jax.numpy as np
import numpy as onp
from jax import grad, vmap, jit, lax
from typing import Optional, Callable
from Activators import sigmoid, derivate
from CostFuncs import CostCrossEntropy, CostOLS_fast
from Schedules import Scheduler
from sklearn.utils import resample
from copy import deepcopy
from tqdm import tqdm
from utils import assign, vstack_arrs
from line_profiler import profile
from functools import partial


@jit
def fast_mul(a, b):
    return lax.mul(a, b)


@jit
def fast_dot(a, b):
    return lax.dot(a, b)


@jit
def fast_dot_with_T(a, b):
    return lax.dot(a, b.T)


@jit
def fast_plus_eq(a, b):
    return lax.add(a, b)


# @jit(static_argnames="lmbda")
@partial(jit, static_argnames="lmbda")
def calc_grad_w(a_layer, delta_matrix, weight_matrix, lmbda):
    gradient_weights = lax.dot(a_layer.T, delta_matrix)
    gradient_weights = lax.add(gradient_weights, lax.mul(weight_matrix, lmbda))
    return gradient_weights


@jit
def setup_bias(X_batch: np.ndarray) -> np.ndarray:
    bias = lax.mul(np.ones((X_batch.shape[0], 1)), 0.01)
    return np.hstack([bias, X_batch])


class NeuralNet:
    """
    A class representing a neural network.

    Attributes:
        dimensions : tuple[int]
            A tuple of integers representing the number of nodes in each layer of the neural network.
        hidden_func : Callable
            A Callable function representing the activation function for the hidden layers.
        output_func : Callable
            A Callable function representing the activation function for the output layer.
        cost_func : Callable
            A Callable function representing the cost function used to evaluate the performance of the neural network.
        seed : Optional[int]
            An optional integer representing the seed for the random number generator used to initialize the weights.

    Methods:
        reset_weights() -> None:
            Resets the weights of the neural network.
        fit(X_train: np.ndarray, target_train: np.ndarray, scheduler: Scheduler, **kwargs) -> dict[str, np.ndarray]:
            Trains the neural network on the given data.
        feed_forward(X_batch: np.ndarray) -> np.ndarray:
            Performs a feed forward pass through the neural network.
        back_propagate(X_batch: np.ndarray, target_batch: np.ndarray, lmbda: float) -> None:
            Performs back propagation to update the weights of the neural network.
        accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
            Calculates the accuracy of the neural network.
        set_classification() -> None:
            Sets the classification attribute of the neural network based on the cost function used.
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = sigmoid,
        cost_func: Callable = CostOLS_fast,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the neural network.

        Args:
            dimensions : tuple[int]
                A tuple of integers representing the number of nodes in each layer of the neural network.
            hidden_func : Callable
                A Callable function representing the activation function for the hidden layers.
            output_func : Callable
                A Callable function representing the activation function for the output layer.
            cost_func : Callable
                A Callable function representing the cost function used to evaluate
                the performance of the neural network.
            seed : Optional[int]
                An optional integer representing the seed for the random number generator
                used to initialize the weights.

        Raises:
            TypeError:
                If dimensions is not a tuple, if any value in dimensions is not an integer,
                or if seed is not an integer or None.
            ValueError:
                If any value in dimensions is less than or equal to 0.
        """
        if not isinstance(dimensions, tuple):
            raise TypeError(f"Dimensions must be tuple, not {type(dimensions)}")
        if not all(isinstance(layer, int) for layer in dimensions):
            raise TypeError(f"Values of dimensions must be ints, not {dimensions}")
        if not isinstance(seed, int) or seed is None:
            raise TypeError(f"Seed must be either None or int, not {type(seed)}")
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError(f"Number of dimensions must be positive, not {dimensions}")

        self.dimensions = dimensions
        self.hidden_func = jit(hidden_func)
        self.output_func = jit(output_func)
        self.cost_func = jit(cost_func)

        # Calculate derivates here to same on jit
        self.cost_func_derivative = jit(grad(self.cost_func))
        self.hidden_derivative = derivate(self.hidden_func)
        self.output_derivative = derivate(self.output_func)

        self.seed = seed

        self.z_layers: list[np.ndarray] = list()
        self.a_layers: list[np.ndarray] = list()

        self.reset_weights()
        self.set_classification()

    @profile
    def reset_weights(self) -> None:
        """
        Resets the weights of the neural network.
        """
        if self.seed is not None:
            onp.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            # Weights
            weight_array = onp.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )

            # Bias
            weight_array[0, :] = onp.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    @profile
    def fit(
        self,
        X_train: np.ndarray,
        target_train: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lmbda: float = 0.0,
        X_val: Optional[np.ndarray] = None,
        target_val: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Trains the neural network on the given data.

        Args:
            X_train : np.ndarray
                A numpy array representing the training data.
            target_train : np.ndarray
                A numpy array representing the target values for the training data.
            scheduler : Scheduler
                A scheduler object used to update the weights of the neural network.
            batches : int, optional
                An integer representing the number of batches to divide the training data into, by default 1.
            epochs : int, optional
                An integer representing the number of epochs to train the neural network for, by default 100.
            lmbda : float, optional
                A float representing the regularization parameter, by default 0.
            X_val : Optional[np.ndarray], optional
                An optional numpy array representing the validation data, by default None.
            target_val : Optional[np.ndarray], optional
                An optional numpy array representing the target values for the validation data, by default None.

        Returns:
            dict[str, np.ndarray]
                A dictionary containing the training and validation errors and accuracies (if applicable).

        Raises:
            TypeError:
                If scheduler is not of class Scheduler, if batches or epochs are not integers,
                or if lmbda is not a number.
            ValueError:
                If batches or epochs are less than or equal to 0, if lmbda is negative, or if
                the number of batches exceeds the number of training points.
        """
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
            onp.random.seed(self.seed)

        # Validate if available
        validate = False
        if X_val is not None and target_val is not None:
            validate = True

        # Cast to float for jax
        lmbda = float(lmbda)

        # Training metrics
        train_errors = np.empty(epochs)
        # train_errors.fill(np.nan)

        train_accuracies = np.empty(epochs)
        # train_accuracies.fill(np.nan)

        self.schedulers_weight: list[Scheduler] = list()
        self.schedulers_bias: list[Scheduler] = list()

        if X_train.shape[0] < batches:
            raise ValueError(
                f"Number of batches cannot exceed training points, {X_train.shape[0]} < {batches}"
            )

        # self.cost_func_derivative = grad(self.cost_func)

        batch_size = X_train.shape[0] // batches

        # One step of bootstrap
        X_train, target_train = resample(X_train, target_train)

        # cost_function_train = self.cost_func(target_train)
        if validate:
            # cost_function_validate = self.cost_func(target_val)

            # Validation metrics
            validation_errors = np.empty(epochs)
            # validation_errors.fill(np.nan)

            validation_accuracies = np.empty(epochs)
            # validation_accuracies.fill(np.nan)

        for i in range(len(self.weights)):
            # I believe deepcopy is necessary to ensure layers are not cross contaminated
            self.schedulers_weight.append(deepcopy(scheduler))
            self.schedulers_bias.append(deepcopy(scheduler))

        data_indices = np.arange(X_train.shape[0])

        pbar = tqdm(total=epochs * batches)
        for e in range(epochs):
            for i in range(batches):
                # Draw with replacement
                batch_idx = onp.random.choice(data_indices, batch_size)
                X_batch = X_train[batch_idx, :]
                target_batch = target_train[batch_idx]

                self.feed_forward(X_batch)
                self.back_propagate(X_batch, target_batch, lmbda)

                pbar.update(1)

            # Reset schedulers
            for weight_scheduler, bias_scheduler in zip(
                self.schedulers_weight, self.schedulers_bias
            ):
                weight_scheduler.reset()
                bias_scheduler.reset()

            pred_train = self.predict(X_train)
            train_error = self.cost_func(pred_train, target_train)
            # train_error = cost_function_train(pred_train)
            train_errors = assign(train_errors, e, train_error)

            if validate:
                pred_val = self.predict(X_val)
                validation_error = self.cost_func(pred_val, target_val)
                validation_errors = assign(validation_errors, e, validation_error)

            if self.classification:
                train_accuracy = self.accuracy(pred_train, target_train)
                train_accuracies = assign(train_accuracies, e, train_accuracy)

            if validate and self.classification:
                pred_accuracy = self.accuracy(pred_val, target_val)
                validation_accuracies = assign(validation_accuracies, e, pred_accuracy)

        scores = {"train_errors": train_errors}
        if validate:
            scores["validation_errors"] = validation_errors
        if self.classification:
            scores["train_accuracy"] = train_accuracies
        if validate and self.classification:
            scores["validation_accuracy"] = validation_accuracies

        return scores

    @profile
    def feed_forward(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Performs a feed forward pass through the neural network.

        Args:
            X_batch : np.ndarray
                A numpy array representing the input data.

        Returns:
            np.ndarray
                A numpy array representing the output of the neural network.
        """
        self.a_layers = list()
        self.z_layers = list()

        # Make sure X is a matrix
        if len(X_batch.shape) == 1:
            X_batch = X_batch.reshape((1, X_batch.size))

        # Add a bias
        X_batch = setup_bias(X_batch)
        # bias = np.ones((X_batch.shape[0], 1)) * 0.01
        # X_batch = np.hstack([bias, X_batch])

        a = X_batch
        self.a_layers.append(a)
        self.z_layers.append(a)

        # Feed forward for all but output layer
        for i in range(len(self.weights) - 1):
            z = fast_dot(a, self.weights[i])
            # z = a @ self.weights[i]
            self.z_layers.append(z)
            a = self.hidden_func(z)
            # print(
            #     np.isnan(a).any(),
            #     np.isnan(z).any(),
            #     np.isnan(tmp_a).any(),
            #     np.isnan(self.weights[i]).any(),
            # )

            # Add bias layer
            a = setup_bias(a)
            # bias = np.ones((a.shape[0], 1)) * 0.01
            # a = np.hstack([bias, a])
            self.a_layers.append(a)

        # for weight in self.weights[:-1]:
        #     # Calculate z for hidden layers
        #     z = a @ weight
        #     self.z_layers.append(z)
        #     # Activate layer
        #     a = self.hidden_func(z)

        #     # Add bias layer
        #     bias = np.ones((a.shape[0], 1)) * 0.01
        #     a = np.hstack((bias, a))
        #     self.a_layers.append(a)

        # Output layer
        z = fast_dot(a, self.weights[-1])
        # z = a @ self.weights[-1]
        a = self.output_func(z)

        self.a_layers.append(a)
        self.z_layers.append(z)

        # Return the output layer
        return a

    @profile
    def back_propagate(
        self, X_batch: np.ndarray, target_batch: np.ndarray, lmbda: float
    ) -> None:
        """
        Performs back propagation to update the weights of the neural network.

        Args:
            X_batch : np.ndarray
                A numpy array representing the input data.
            target_batch : np.ndarray
                A numpy array representing the target values for the input data.
            lmbda : float
                A float representing the regularization parameter.

        Raises:
            ValueError:
                If the shapes of prediction and target do not correspond.
        """

        # Start with output layer
        i = len(self.weights) - 1

        if self.output_func.__name__ == "softmax":
            delta_matrix = self.a_layers[i + 1] - target_batch
        else:
            # delta_matrix = output_delta(
            #     self.output_derivative,
            #     self.cost_func,
            #     self.z_layers[i + 1],
            #     self.a_layers[i + 1],
            #     target_batch,
            # )
            # cost_func_derivative = grad(self.cost_func(target_batch))

            left = self.output_derivative(self.z_layers[i + 1])
            right = self.cost_func_derivative(self.a_layers[i + 1], target_batch)
            # print("a:", self.a_layers[i + 1])
            # print(target_batch)
            # print(right)
            # quit()

            # delta_matrix = output_derivative(
            #     self.z_layers[i + 1]
            # ) * cost_func_derivative(self.a_layers[i + 1])
            delta_matrix = fast_mul(left, right)

        # Output gradient
        # gradient_weights = fast_dot(self.a_layers[i][:, 1:].T, delta_matrix)
        # gradient_weights = self.a_layers[i][:, 1:].T @ delta_matrix
        gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])
        gradient_weights = calc_grad_w(
            self.a_layers[i][:, 1:], delta_matrix, self.weights[i][1:, :], lmbda
        )

        # gradient_weights = fast_plus_eq(
        #     gradient_weights, fast_mul(self.weights[i][1:, :], lmbda)
        # )

        # gradient_weights += self.weights[i][1:, :] * lmbda

        update_matrix = vstack_arrs(
            self.schedulers_bias[i].update_change(gradient_bias),
            self.schedulers_weight[i].update_change(gradient_weights),
        )

        self.weights[i] -= update_matrix

        # Back propagate the hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Calculate error for layer
            left = fast_dot_with_T(self.weights[i + 1][1:, :], delta_matrix)
            # left = fast_dot(self.weights[i + 1][1:, :], delta_matrix.T)
            right = self.hidden_derivative(self.z_layers[i + 1])

            delta_matrix = fast_mul(left.T, right)
            # delta_matrix = (
            #     self.weights[i + 1][1:, :] @ delta_matrix.T
            # ).T * hidden_derivative(self.z_layers[i + 1])

            # Calculate gradients
            # gradient_weights = fast_dot(self.a_layers[i][:, 1:].T, delta_matrix)
            # gradient_weights = self.a_layers[i][:, 1:].T @ delta_matrix
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]
            )
            gradient_weights = calc_grad_w(
                self.a_layers[i][:, 1:], delta_matrix, self.weights[i][1:, :], lmbda
            )

            # Regularize
            # gradient_weights = fast_plus_eq(
            #     gradient_weights, fast_mul(self.weights[i][1:, :], lmbda)
            # )
            # gradient_weights += self.weights[i][1:, :] * lmbda

            update_matrix = vstack_arrs(
                self.schedulers_bias[i].update_change(gradient_bias),
                self.schedulers_weight[i].update_change(gradient_weights),
            )
            # Update weights
            self.weights[i] -= update_matrix

    def accuracy(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates the accuracy of the neural network.

        Args:
            prediction : np.ndarray
                A numpy array representing the predicted values.
            target : np.ndarray
                A numpy array representing the target values.

        Returns:
            float
                A float representing the accuracy of the neural network.

        Raises:
            ValueError:
                If the shapes of prediction and target do not correspond.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shapes must correspond, not {prediction.shape} and {target.shape}"
            )
        return np.average((target == prediction))

    def set_classification(self) -> None:
        """
        Sets the classification attribute of the neural network based on the cost function used.
        """
        self.classification = self.cost_func.__name__ in [
            "CostLogReg",
            "CostCrossEntropy",
            "CostCrossEntropy_fast",
            "CostCrossEntropy_binary" "BinaryCrossEntropy_fast",
            "CostCrossEntropy_binary",
            "oops",
        ]

    def predict(self, X: np.ndarray, *, theshold: float = 0.5) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Args:
            X (np.ndarray): The input data to predict the output for.
            theshold (float, optional): The threshold value for classification. Defaults to 0.5.

        Returns:
            np.ndarray: The predicted output for the given input data.
        """
        predict = self.feed_forward(X)
        if self.classification:
            return np.where(predict > theshold, 1.0, 0.0)
        return predict


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
        self.hidden_weights = onp.random.randn(self.n_features, self.n_hidden_nodes)
        self.hidden_bias = np.zeros(self.n_hidden_nodes) + 0.01

        # Weights and bias for the output layer
        self.output_weights = onp.random.randn(self.n_hidden_nodes, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # Weighted sum over the inputs - $z_h = \sum_{i=1}^F w_{ij}^l x_i + b_{i}^l$
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias
        # Activation function - $a_h = f(z_h)$
        self.a_h = self.hidden_func(self.z_h)

        # Activation values $z$
        self.z_o = self.a_h @ self.output_weights + self.output_bias

        self.a_o = self.output_func(self.z_o)

    def feed_forward_out(self, X: np.ndarray) -> np.ndarray:
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
        derivative_hidden = vmap(grad(self.hidden_func))
        derivative_output = vmap(grad(self.output_func))

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
                batch_indices = onp.random.choice(data_indices, size=self.batch_size)

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
