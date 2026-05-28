import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        self.dimensions = dimensions
        self.activations = activations
        self.n_layers = len(dimensions)

        self.weights = {}
        self.biases = {}
        self.learning_rate = 1e-3

        for i in range(1, self.n_layers):
            self.weights[i] = np.random.randn(
                dimensions[i - 1], dimensions[i]
            ) * 0.01

            self.biases[i] = np.zeros((1, dimensions[i]))

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        a = {}
        z = {}

        z[0] = x

        for i in range(1, self.n_layers):

            a[i] = np.dot(z[i - 1], self.weights[i]) + self.biases[i]

            z[i] = self.activations[i - 1].forward(a[i])

        return z, a


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        z, _ = self.feed_forward(x)

        return z[self.n_layers - 1]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        deltas = {}

        L = self.n_layers - 1

        # Output layer delta
        dloss = loss.gradient(y_true, z[L])

        dactivation = self.activations[L - 1].gradient(a[L])

        deltas[L] = dloss * dactivation

        # Hidden layers
        for i in range(L - 1, 0, -1):

            dactivation = self.activations[i - 1].gradient(a[i])

            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * dactivation

        # Update parameters
        for i in range(1, self.n_layers):

            dw = np.dot(z[i - 1].T, deltas[i]) / y_true.shape[0]

            self.update_w_b(i, dw, deltas[i])


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        db = np.mean(delta, axis=0, keepdims=True)

        self.weights[index] -= self.learning_rate * dw

        self.biases[index] -= self.learning_rate * db

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        self.learning_rate = learning_rate

        n_samples = x.shape[0]

        for epoch in range(epochs):

            # Uncomment to print the current training epoch
            print(f"Epoch {epoch + 1}/{epochs}")

            indices = np.random.permutation(n_samples)

            x_shuffled = x[indices]
            y_shuffled = y_true[indices]

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                z, a = self.feed_forward(x_batch)

                self.back_prop(
                    z,
                    a,
                    y_batch,
                    loss
                )

        return self.predict(x)
