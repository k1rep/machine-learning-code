import logging

import numpy as np
from autograd import elementwise_grad

from neuralnet.layers import PhaseMixin
from neuralnet.loss import get_loss, mse_grad
from neuralnet.optimizers import batch_iterator

np.random.seed(42)


class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()


class NeuralNetwork(BaseEstimator):
    def __init__(self, layers, optimizer, loss, max_epochs=10,
                 batch_size=64, metric="mse", shuffle=False, verbose=True):
        self.optimizer = optimizer
        self.loss = get_loss(loss)
        if loss == "binary_cross_entropy":
            self.loss_grad = lambda actual, predicted: -(actual - predicted)
        elif loss == "mse":
            self.loss_grad = mse_grad
        else:
            self.loss_grad = elementwise_grad(self.loss, 1)
        self.shuffle = shuffle
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.layers = layers
        self.metric = get_loss(metric)
        self.metric_name = metric
        self._initialized = False
        self.training = False
        self._n_layers = 0
        self.log_metric = True if loss != metric else False
        self.bprop_entry = self._find_bprop_entry()

    def _setup_layers(self, x_shape):
        """Initialize layers and set up optimizer."""
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size
        for layer in self.layers:
            layer.setup(x_shape)
            x_shape = layer.shape(x_shape)

        self._n_layers = len(self.layers)
        self.optimizer.setup(self)
        self._initialized = True
        logging.info("Total parameters: %s" % self.n_params)

    def _find_bprop_entry(self):
        """Find the first layer that has a backward method."""
        if len(self.layers) > 0 and not hasattr(self.layers[-1], "parameters"):
            return -1
        return len(self.layers)

    def fit(self, X, y=None):
        if not self._initialized:
            self._setup_layers(X.shape)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        self._setup_input(X, y)
        self.is_training = True
        self.optimizer.optimize(self)
        self.is_training = False

    def update(self, X, y):
        y_pred = self.fprop(X)
        grad = self.loss_grad(y, y_pred)
        for layer in reversed(self.layers[:self.bprop_entry]):
            grad = layer.backward_pass(grad)
        return self.loss(y, y_pred)

    def fprop(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def _predict(self, X=None):
        if not self._initialized:
            self._setup_layers(X.shape)

        y = []
        X_batch = batch_iterator(X, self.batch_size)
        for X in X_batch:
            y.append(self.fprop(X))
        return np.concatenate(y)

    @property
    def parametric_layers(self):
        return [layer for layer in self.layers if hasattr(layer, "parameters")]

    @property
    def parameters(self):
        return [layer.parameters for layer in self.parametric_layers]

    def error(self, X=None, y=None):
        training_phase = self.is_training
        if training_phase:
            self.is_training = False
        if X is None and y is None:
            y_pred = self._predict(self.X)
            score = self.metric(self.y, y_pred)
        else:
            y_pred = self._predict(X)
            score = self.metric(y, y_pred)
        if training_phase:
            self.is_training = True
        return score

    @property
    def is_training(self):
        return self.training

    @is_training.setter
    def is_training(self, value):
        self.training = value
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.is_training = value

    def shuffle_dataset(self):
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X = self.X.take(indices, axis=0)
        self.y = self.y.take(indices, axis=0)

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def n_params(self):
        return sum(layer.parameters.n_params for layer in self.parametric_layers)

    def reset(self):
        self._initialized = False
