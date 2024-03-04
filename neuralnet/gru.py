import autograd.numpy as np
from autograd import elementwise_grad


from neuralnet.layers import Layer, ParamMixin, TimeDistributedDense
from neuralnet.activations import get_activations
from neuralnet.initializations import get_initializer
from neuralnet.parameters import Parameters


class GRU(Layer, ParamMixin):
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
        Parameters
        ----------
        x_shape : np.array(batch_size, time_steps, input_shape)
            Shape of the input data.
        """
        self.input_dim = x_shape[2]

        # GRU parameters: reset and update gates, and candidate hidden state
        W_params = ["W_z", "W_r", "W_h"]
        U_params = ["U_z", "U_r", "U_h"]
        b_params = ["b_z", "b_r", "b_h"]
        # initialize parameters
        for param in W_params:
            self._params[param] = self._params.init((self.input_dim, self.hidden_dim))
        for param in U_params:
            self._params[param] = self.inner_init((self.hidden_dim, self.hidden_dim))
        for param in b_params:
            self._params[param] = np.zeros((self.hidden_dim,))

        self.W = [self._params[p] for p in W_params]
        self.U = [self._params[p] for p in U_params]
        self._params.init_grad()

        self.hprev = np.zeros((x_shape[0], self.hidden_dim))

    def forward_pass(self, X):
        self.last_input = X
        n_samples, n_timesteps, input_shape = X.shape
        p = self._params

        self.states = np.zeros((n_samples, n_timesteps, self.hidden_dim))
        self.gates = {k: np.zeros((n_samples, n_timesteps, self.hidden_dim)) for k in ["z", "r", "h"]}
        self.outputs = np.zeros_like(self.states)

        for i in range(n_timesteps):
            x_t = X[:, i, :]
            h_prev = self.outputs[:, i - 1, :] if i > 0 else self.hprev

            z = self.sigmoid(np.dot(x_t, p["W_z"]) + np.dot(h_prev, p["U_z"]) + p["b_z"])
            r = self.sigmoid(np.dot(x_t, p["W_r"]) + np.dot(h_prev, p["U_r"]) + p["b_r"])
            h_candidate = self.activation(np.dot(x_t, p["W_h"]) + np.dot(r * h_prev, p["U_h"]) + p["b_h"])

            h = (1 - z) * h_prev + z * h_candidate
            self.outputs[:, i, :] = h
            self.gates["z"][:, i, :] = z
            self.gates["r"][:, i, :] = r
            self.gates["h"][:, i, :] = h_candidate

        self.hprev = self.outputs[:, -1, :].copy()
        return self.outputs[:, :-1, :] if self.return_sequences else self.outputs[:, -2, :]

    def backward_pass(self, delta):
        if len(delta.shape) == 2:
            delta = delta[:, np.newaxis, :]
        n_samples, n_timesteps, input_shape = delta.shape
        p = self._params
        grad = {k: np.zeros_like(p[k]) for k in p.keys()}

        dh_next = np.zeros((n_samples, input_shape))
        output = np.zeros((n_samples, n_timesteps, self.input_dim))

        for t in reversed(range(n_timesteps)):
            x_t = self.last_input[:, t, :]
            h_prev = self.outputs[:, t - 1, :] if t > 0 else self.hprev

            z = self.gates["z"][:, t, :]
            r = self.gates["r"][:, t, :]
            h_candidate = self.gates["h"][:, t, :]
            h = self.outputs[:, t, :]

            dh = delta[:, t, :] + dh_next
            dh_candidate = dh * z
            dh_prev = dh * (1 - z)

            dz = dh * (h_candidate - h_prev)
            dr = dh_candidate * (p["U_h"].T @ (r * (1 - r)))

            dW_z = x_t.T @ dz
            dU_z = h_prev.T @ dz
            db_z = dz.sum(axis=0)

            dW_r = x_t.T @ dr
            dU_r = h_prev.T @ dr
            db_r = dr.sum(axis=0)

            dh_candidate_d = dh_candidate * self.activation_d(h_candidate)
            dW_h = x_t.T @ (dh_candidate_d * r)
            dU_h = (r * h_prev).T @ dh_candidate_d
            db_h = (dh_candidate_d * r).sum(axis=0)

            # Update gradients
            grad["W_z"] += dW_z
            grad["U_z"] += dU_z
            grad["b_z"] += db_z

            grad["W_r"] += dW_r
            grad["U_r"] += dU_r
            grad["b_r"] += db_r

            grad["W_h"] += dW_h
            grad["U_h"] += dU_h
            grad["b_h"] += db_h

            dh_next = dh_prev + (dh_candidate_d * p["U_h"]).sum(axis=1)

        for k in grad.keys():
            self._params.update_grad(k, grad[k]/n_samples)
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


    binary_addition_problem(GRU(16))
