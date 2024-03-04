import autograd.numpy as np
from autograd import elementwise_grad


from neuralnet.layers import Layer, ParamMixin, TimeDistributedDense
from neuralnet.activations import get_activations
from neuralnet.initializations import get_initializer
from neuralnet.parameters import Parameters


class RNN(Layer, ParamMixin):
    """Vanilla RNN layer."""
    def __init__(self, hidden_dim,  activation='tanh', inner_init='orthogonal', parameters=None, return_sequences=True):
        self.hidden_dim = hidden_dim
        self.activation = get_activations(activation)
        self.activation_d = elementwise_grad(self.activation)
        self.inner_init = get_initializer(inner_init)
        self.return_sequences = return_sequences
        if parameters is None:
            self._params = Parameters()
        else:
            self._params = parameters
        self.last_input = None
        self.states = None
        self.hprev = None
        self.input_dim = None

    def setup(self, x_shape):
        """
        Parameters
        ----------
        x_shape : np.array(batch_size, time_steps, input_shape)
            Shape of the input data.
        """
        self.input_dim = x_shape[2]

        # input -> hidden
        self._params["W"] = self._params.init((self.input_dim, self.hidden_dim))
        # bias
        self._params["b"] = np.full((self.hidden_dim, ), self._params.initial_bias)
        # hidden -> hidden
        self._params["U"] = self.inner_init((self.hidden_dim, self.hidden_dim))

        self._params.init_grad()

        self.hprev = np.zeros((x_shape[0], self.hidden_dim))

    def forward_pass(self, X):
        self.last_input = X
        n_samples, n_timesteps, input_shape = X.shape
        states = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        states[:, -1, :] = self.hprev.copy()
        p = self._params
        for i in range(n_timesteps):
            states[:, i, :] = np.tanh(np.dot(X[:, i, :], p["W"]) + np.dot(states[:, i-1, :], p["U"]) + p["b"])
        self.states = states
        self.hprev = states[:, n_timesteps-1, :].copy()
        if self.return_sequences:
            return states[:, 0:-1, :]
        else:
            return states[:, -2, :]

    def backward_pass(self, delta):
        if len(delta.shape) == 2:
            delta = delta[:, np.newaxis, :]
        n_samples, n_timesteps, input_shape = delta.shape
        p = self._params
        grad = {k: np.zeros_like(p[k]) for k in p.keys()}

        dh_next = np.zeros((n_samples, input_shape))
        output = np.zeros((n_samples, n_timesteps, self.input_dim))

        for i in reversed(range(n_timesteps)):
            dhi = self.activation_d(self.states[:, i, :]) * (delta[:, i, :] + dh_next)
            grad["W"] += np.dot(self.last_input[:, i, :].T, dhi)
            grad["U"] += np.dot(self.states[:, i-1, :].T, dhi)
            grad["b"] += np.sum(delta[:, i, :], axis=0)

            dh_next = np.dot(dhi, p["U"].T)
            d = np.dot(delta[:, i, :], p["U"].T)
            output[:, i, :] = np.dot(d, p["W"].T)

        for k in grad.keys():
            self._params.update_grad(k, grad[k])
        return output

    def shape(self, x_shape):
        if self.return_sequences:
            return x_shape[0], x_shape[1], self.hidden_dim
        else:
            return x_shape[0], self.hidden_dim


if __name__ == '__main__':
    from itertools import combinations, islice
    import numpy as np
    from sklearn.model_selection import train_test_split
    from neuralnet.layers import Activation
    from neuralnet.optimizers import Adam
    from neuralnet.NeuralNetwork import NeuralNetwork
    from neuralnet.constraints import SmallNorm
    from neuralnet.loss import accuracy

    def binary_addition_dataset(dim=10, n_samples=10000, batch_size=64):
        """Generate binary addition dataset.
            http://devankuleindiren.com/Projects/rnn_arithmetic.php
        """
        binary_format = "{:0" + str(dim) + "b}"

        # Generate all possible number combinations
        combs = list(islice(combinations(range(2 ** (dim - 1)), 2), n_samples))

        # Initialize empty arrays
        X = np.zeros((len(combs), dim, 2), dtype=np.uint8)
        y = np.zeros((len(combs), dim, 1), dtype=np.uint8)

        for i, (a, b) in enumerate(combs):
            # Convert numbers to binary format
            X[i, :, 0] = list(reversed([int(x) for x in binary_format.format(a)]))
            X[i, :, 1] = list(reversed([int(x) for x in binary_format.format(b)]))

            # Generate target variable (a+b)
            y[i, :, 0] = list(reversed([int(x) for x in binary_format.format(a + b)]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

        # Round number of examples for batch processing
        train_b = (X_train.shape[0] // batch_size) * batch_size
        test_b = (X_test.shape[0] // batch_size) * batch_size
        X_train = X_train[0:train_b]
        y_train = y_train[0:train_b]

        X_test = X_test[0:test_b]
        y_test = y_test[0:test_b]
        return X_train, y_train, X_test, y_test

    def binary_addition_problem(recurrent_layer):
        X_train, y_train, X_test, y_test = binary_addition_dataset(8, 5000)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        model = NeuralNetwork(
            layers=[
                recurrent_layer,
                TimeDistributedDense(1),
                Activation('sigmoid')
            ],
            loss='mse',
            optimizer=Adam(),
            metric='mse',
            batch_size=64,
            max_epochs=19,
        )
        model.fit(X_train, y_train)
        predictions = np.round(model.predict(X_test))
        predictions = np.packbits(predictions.astype(np.uint8))
        y_test = np.packbits(y_test.astype(np.int64))
        print('Accuracy:', accuracy(y_test, predictions))


    binary_addition_problem(RNN(16, parameters=Parameters(constraints={'W': SmallNorm(), 'U': SmallNorm()})))
