import autograd.numpy as np
from autograd import elementwise_grad


from neuralnet.layers import Layer, ParamMixin, TimeDistributedDense
from neuralnet.activations import get_activations
from neuralnet.initializations import get_initializer
from neuralnet.parameters import Parameters


class LSTM(Layer, ParamMixin):
    def __init__(self, hidden_dim, activation='tanh', inner_init='orthogonal', parameters=None, return_sequences=True):
        self.hidden_dim = hidden_dim
        self.activation = get_activations(activation)
        self.activation_d = elementwise_grad(self.activation)
        self.sigmoid = get_activations('sigmoid')
        self.sigmoid_d = elementwise_grad(self.sigmoid)
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
        self.outputs = None
        self.W = None
        self.U = None
        self.gates = None

    def setup(self, x_shape):
        """
        Naming convention:
        i : input gate
        f : forget gate
        o : output gate
        c : cell state

        Parameters
        ----------
        x_shape : np.array(batch_size, time_steps, input_shape)
            Shape of the input data.
        """
        self.input_dim = x_shape[-1]

        # input -> hidden
        W_params = ["W_i", "W_f", "W_o", "W_c"]
        # hidden -> hidden
        U_params = ["U_i", "U_f", "U_o", "U_c"]
        # bias terms
        b_params = ["b_i", "b_f", "b_o", "b_c"]
        # initialize parameters
        for param in W_params:
            self._params[param] = self._params.init((self.input_dim, self.hidden_dim))
        for param in U_params:
            self._params[param] = self.inner_init((self.hidden_dim, self.hidden_dim))
        for param in b_params:
            self._params[param] = np.full((self.hidden_dim,), self._params.initial_bias)

        self.W = [self._params[p] for p in W_params]
        self.U = [self._params[p] for p in U_params]
        self._params.init_grad()

        self.hprev = np.zeros((x_shape[0], self.hidden_dim))
        self.oprev = np.zeros((x_shape[0], self.hidden_dim))

    def forward_pass(self, X):
        self.last_input = X
        n_samples, n_timesteps, input_shape = X.shape
        p = self._params

        if self.hprev.shape[0] != n_samples:
            self.hprev = np.zeros((n_samples, self.hidden_dim))
            self.oprev = np.zeros((n_samples, self.hidden_dim))

        self.states = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        self.outputs = np.zeros((n_samples, n_timesteps + 1, self.hidden_dim))
        self.gates = {k: np.zeros((n_samples, n_timesteps, self.hidden_dim)) for k in ["i", "f", "o", "c"]}
        self.states[:, -1, :] = self.hprev
        self.outputs[:, -1, :] = self.oprev

        for i in range(n_timesteps):
            t_gates = np.dot(X[:, i, :], self.W) + np.dot(self.outputs[:, i-1, :], self.U)
            # input
            self.gates["i"][:, i, :] = self.sigmoid(t_gates[:, 0, :] + p["b_i"])
            # forget
            self.gates["f"][:, i, :] = self.sigmoid(t_gates[:, 1, :] + p["b_f"])
            # output
            self.gates["o"][:, i, :] = self.sigmoid(t_gates[:, 2, :] + p["b_o"])
            # cell
            self.gates["c"][:, i, :] = self.activation(t_gates[:, 3, :] + p["b_c"])
            # (previous state * forget) + input * cell
            self.states[:, i, :] = (self.states[:, i-1, :] * self.gates["f"][:, i, :]
                                    + self.gates["i"][:, i, :] * self.gates["c"][:, i, :])
            self.outputs[:, i, :] = self.gates["o"][:, i, :] * self.activation(self.states[:, i, :])
        self.hprev = self.states[:, n_timesteps-1, :].copy()
        self.oprev = self.outputs[:, n_timesteps-1, :].copy()
        if self.return_sequences:
            return self.outputs[:, 0:-1, :]
        else:
            return self.outputs[:, -2, :]

    def backward_pass(self, delta):
        if len(delta.shape) == 2:
            delta = delta[:, np.newaxis, :]
        n_samples, n_timesteps, input_shape = delta.shape
        p = self._params
        grad = {k: np.zeros_like(p[k]) for k in p.keys()}

        dh_next = np.zeros((n_samples, input_shape))
        output = np.zeros((n_samples, n_timesteps, self.input_dim))

        for i in reversed(range(n_timesteps)):
            dhi = self.gates["o"][:, i, :] * self.activation_d(self.states[:, i, :]) * delta[:, i, :] + dh_next
            og = delta[:, i, :] * self.activation(self.states[:, i, :])

            de_o = og * self.sigmoid_d(self.gates["o"][:, i, :])
            grad["W_o"] += np.dot(self.last_input[:, i, :].T, de_o)
            grad["U_o"] += np.dot(self.outputs[:, i - 1, :].T, de_o)
            grad["b_o"] += de_o.sum(axis=0)

            de_f = (dhi * self.states[:, i-1, :]) * self.sigmoid_d(
                self.gates["f"][:, i, :])
            grad["W_f"] += np.dot(self.last_input[:, i, :].T, de_f)
            grad["U_f"] += np.dot(self.outputs[:, i - 1, :].T, de_f)
            grad["b_f"] += de_f.sum(axis=0)

            de_i = (dhi * self.gates["c"][:, i, :]) * self.sigmoid_d(
                self.gates["i"][:, i, :])
            grad["W_i"] += np.dot(self.last_input[:, i, :].T, de_i)
            grad["U_i"] += np.dot(self.outputs[:, i - 1, :].T, de_i)
            grad["b_i"] += de_i.sum(axis=0)

            de_c = (dhi * self.gates["i"][:, i, :]) * self.activation_d(self.gates["c"][:, i, :])
            grad["W_c"] += np.dot(self.last_input[:, i, :].T, de_c)
            grad["U_c"] += np.dot(self.outputs[:, i-1, :].T, de_c)
            grad["b_c"] += de_c.sum(axis=0)

            dh_next = dhi * self.gates["f"][:, i, :]

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
            max_epochs=30,
        )
        model.fit(X_train, y_train)
        predictions = np.round(model.predict(X_test))
        predictions = np.packbits(predictions.astype(np.uint8))
        y_test = np.packbits(y_test.astype(np.int64))
        print('Accuracy:', accuracy(y_test, predictions))


    binary_addition_problem(LSTM(16))
