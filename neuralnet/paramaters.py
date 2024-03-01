import numpy as np
from neuralnet.initializations import get_initializer


class Parameters:
    def __init__(self, init='glorot_uniform', scale=0.5, bias=1.0, regularizers=None, constraints=None):
        """A container for layer's parameters.
        Parameters:
        -----------
        init: str or callable
            The initializer for the weights.
        scale: float
            The scale of the initializer.
        bias: float
            The initial value for the bias.
        regularizers: dict
            A dictionary of regularizers for each parameter.
        constraints: dict
            A dictionary of constraints for each parameter.
        """
        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints

        if regularizers is None:
            self.regularizers = {}
        else:
            self.regularizers = regularizers

        self.init = get_initializer(init)
        self.scale = scale
        self.initial_bias = bias

        self._params = {}
        self._grads = {}

    def setup_weights(self, W_shape, b_shape=None):
        if "W" not in self._params:
            self._params["W"] = self.init(W_shape, scale=self.scale)
            if b_shape is None:
                self._params["b"] = np.full(W_shape[1], self.initial_bias)
            else:
                self._params["b"] = np.full(b_shape, self.initial_bias)
        self.init_grad()

    def init_grad(self):
        """Initialize gradient arrays corresponding to each weight array."""
        for key in self._params.keys():
            if key not in self._grads:
                self._grads[key] = np.zeros_like(self._params[key])

    def step(self, name, value):
        """Update the parameter with the given name."""
        self._params[name] += value

        if name in self.constraints:
            self._params[name] = self.constraints[name].clip(self._params[name])

    def update_grad(self, name, value):
        """Update the gradient with the given name."""
        self._grads[name] = value

        if name in self.regularizers:
            self._grads[name] += self.regularizers[name](self._params[name])

    @property
    def n_params(self):
        """Return the total number of parameters in this layer."""
        return sum([np.prod(self._params[p].shape) for p in self._params.keys()])

    def keys(self):
        """Return the keys of the parameter dictionary."""
        return self._params.keys()

    @property
    def grad(self):
        """Return the gradient dictionary."""
        return self._grads

    def __getitem__(self, item):
        """Return the parameter with the given name, e.g. layer.parameters['W']"""
        if item in self._params:
            return self._params[item]
        else:
            raise KeyError(f"Parameter {item} not found")

    def __setitem__(self, key, value):
        """Set the parameter with the given name."""
        self._params[key] = value
