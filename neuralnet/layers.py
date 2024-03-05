import numpy as np
from autograd import elementwise_grad

from neuralnet.parameters import Parameters
from neuralnet.activations import get_activations


np.random.seed(42)


class Layer:
    def setup(self, X_shape):
        """Setup the layer with the input shape."""
        pass

    def forward_pass(self, X):
        """Compute the output of the layer."""
        raise NotImplementedError

    def backward_pass(self, delta):
        """Compute the delta of the layer."""
        raise NotImplementedError

    def shape(self, x_shape):
        """Compute the shape of the output."""
        raise NotImplementedError


class ParamMixin:
    @property
    def parameters(self):
        return self._params


class PhaseMixin:
    _train = False

    @property
    def is_training(self):
        return self._train

    @is_training.setter
    def is_training(self, is_train=True):
        self._train = is_train

    @property
    def is_testing(self):
        return not self._train

    @is_testing.setter
    def is_testing(self, is_test=True):
        self._train = not is_test


class Dense(Layer, ParamMixin):
    def __init__(self, output_dim, parameters=None):
        """A fully connected layer.

        Parameters
        ----------
        output_dim : int
            Dimension of the output.
        """
        self.last_input = None
        self.output_dim = output_dim
        self._params = parameters
        if self._params is None:
            self._params = Parameters()

    def setup(self, x_shape):
        self._params.setup_weights((x_shape[1], self.output_dim))

    def weight(self, X):
        W = np.dot(X, self._params['W'])
        return W + self._params['b']

    def forward_pass(self, X):
        self.last_input = X
        return self.weight(X)

    def backward_pass(self, delta):
        d_W = np.dot(self.last_input.T, delta)
        d_b = np.sum(delta, axis=0)

        self._params.update_grad("W", d_W)
        self._params.update_grad("b", d_b)
        return delta.dot(self._params['W'].T)

    def shape(self, x_shape):
        return x_shape[0], self.output_dim


class Activation(Layer):
    def __init__(self, activation):
        """An activation layer.

        Parameters
        ----------
        activation : str
            The activation function to use.
        """
        self.last_input = None
        self.activation = get_activations(activation)
        self.activation_d = elementwise_grad(self.activation)

    def forward_pass(self, X):
        self.last_input = X
        return self.activation(X)

    def backward_pass(self, delta):
        return self.activation_d(self.last_input) * delta

    def shape(self, x_shape):
        return x_shape


class Dropout(Layer, PhaseMixin):
    """Randomly set a fraction of the input to zero."""
    def __init__(self, p=0.1):
        """A dropout layer.

        Parameters
        ----------
        p : float
            The probability of setting a value to zero.
        """
        self.p = p
        self._mask = None

    def forward_pass(self, X):
        assert 0 <= self.p < 1
        if self.is_training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            y = X * self._mask
        else:
            y = X * (1.0 - self.p)

        return y

    def backward_pass(self, delta):
        return delta * self._mask

    def shape(self, x_shape):
        return x_shape


class TimeStepSlicer(Layer):
    """Take a specific time step from 3D tensor."""
    def __init__(self, step=-1):
        self.step = step

    def forward_pass(self, x):
        return x[:, self.step, :]

    def backward_pass(self, delta):
        return np.repeat(delta[:, np.newaxis, :], 2, axis=1)

    def shape(self, x_shape):
        return x_shape[0], x_shape[2]


class TimeDistributedDense(Layer):
    """Apply regular dense layer to each time step."""
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.dense = None
        self.input_dim = None
        self.n_timesteps = None

    def setup(self, X_shape):
        self.dense = Dense(self.output_dim)
        self.dense.setup((X_shape[0], X_shape[-1]))
        self.input_dim = X_shape[-1]

    def forward_pass(self, X):
        n_timesteps = X.shape[1]
        X = X.reshape(-1, X.shape[-1])
        y = self.dense.forward_pass(X)
        y = y.reshape((-1, n_timesteps, self.output_dim))
        return y

    def backward_pass(self, delta):
        n_timesteps = delta.shape[1]
        X = delta.reshape(-1, delta.shape[-1])
        y = self.dense.backward_pass(X)
        y = y.reshape((-1, n_timesteps, self.input_dim))
        return y

    @property
    def parameters(self):
        return self.dense._params

    def shape(self, x_shape):
        return x_shape[0], x_shape[1], self.output_dim


class EmbeddingLayer(Layer, ParamMixin):
    def __init__(self, input_dim, output_dim, parameters=None):
        """
        初始化嵌入层。

        Parameters
        ----------
        input_dim : int
            词汇表的大小（即，最大整数索引 + 1）。
        output_dim : int
            嵌入的维度。
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._params = parameters
        if self._params is None:
            self._params = Parameters()
        self.last_input = None

    def setup(self, x_shape):
        # 嵌入矩阵的形状应为(input_dim, output_dim)
        self._params.setup_weights((self.input_dim, self.output_dim))

    def forward_pass(self, X):
        self.last_input = X  # Store the input indices for use in backward pass
        return self._params['W'][X]

    def backward_pass(self, delta):
        pass

    def shape(self, x_shape):
        # 输出形状为(batch_size, sequence_length, output_dim)
        return x_shape[0], x_shape[1], self.output_dim
