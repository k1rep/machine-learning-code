import numpy as np

from neuralnet.NeuralNetwork import NeuralNetwork
from neuralnet.layers import Dense, Activation
from neuralnet.optimizers import Adam


class Autoencoder:
    def __init__(self, n_hidden=128, n_input=28*28):
        self.n_hidden = n_hidden
        self.n_input = n_input

        self.encoder = NeuralNetwork(
            layers=[
                Dense(512),
                Activation("leaky_relu"),
                Dense(256),
                Activation("leaky_relu"),
                Dense(self.n_hidden)
            ],
            optimizer=Adam(),
            loss="mse",
            metric="mse",
            batch_size=64,
            max_epochs=10
        )

        self.decoder = NeuralNetwork(
            layers=[
                Dense(256),
                Activation("leaky_relu"),
                Dense(512),
                Activation("leaky_relu"),
                Dense(self.n_input),
                Activation("tanh")
            ],
            optimizer=Adam(),
            loss="mse",
            metric="mse",
            batch_size=64,
            max_epochs=10
        )

    def fit(self, X):
        (self.encoder + self.decoder).fit(X, X)


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == "__main__":
    from dataset.dataset import load_mnist

    X_train, X_test, y_train, y_test = load_mnist()
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    X_train /= 255.0
    X_test /= 255.0

    y_train = one_hot(y_train.flatten())
    y_test = one_hot(y_test.flatten())

    model = Autoencoder()

    model.fit(X_train)
