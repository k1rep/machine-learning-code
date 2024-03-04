import numpy as np
from neuralnet.layers import Layer, ParamMixin, PhaseMixin
from neuralnet.parameters import Parameters


class BatchNormalization(Layer, ParamMixin, PhaseMixin):
    """
    References:
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """
    def __init__(self, momentum=0.9, eps=1e-5, parameters=None):
        super().__init__()
        self._params = parameters
        if self._params is None:
            self._params = Parameters()
        self.momentum = momentum
        self.eps = eps
        self.ema_mean = None
        self.ema_var = None

    def setup(self, x_shape):
        self._params.setup_weights((1, x_shape[1]))

    def _forward_pass(self, X):
        gamma = self._params["W"]
        beta = self._params["b"]

        if self.is_testing:
            mu = self.ema_mean
            xmu = X - mu
            var = self.ema_var
            sqrtvar = np.sqrt(var + self.eps)
            ivar = 1.0 / sqrtvar
            xhat = xmu * ivar
            gammax = gamma * xhat
            return gammax + beta

        N, D = X.shape
        # step 1: compute mean
        mu = 1.0 / N * np.sum(X, axis=0)

        # step 2: subtract mean vector of each sample
        xmu = X - mu

        # step 3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step 4: calculate variance
        var = 1.0 / N * np.sum(sq, axis=0)

        # step 5: calculate square root of variance
        sqrtvar = np.sqrt(var + self.eps)

        # step 6: calculate inverse variance
        ivar = 1.0 / sqrtvar

        # step 7: execute normalization
        xhat = xmu * ivar

        # step 8: nor the two transformation steps
        gammax = gamma * xhat

        # step 9: add beta
        out = gammax + beta

        # store running averages of mean and variance during training for use during testing
        if self.ema_mean is None or self.ema_var is None:
            self.ema_mean = mu
            self.ema_var = var
        else:
            self.ema_mean = self.momentum * self.ema_mean + (1 - self.momentum) * mu
            self.ema_var = self.momentum * self.ema_var + (1 - self.momentum) * var

        # store variables for backward pass
        self.cache = (xhat, gamma, xmu, ivar, sqrtvar, var)

        return out

    def forward_pass(self, X):
        if len(X.shape) == 2:
            # regular layer
            return self._forward_pass(X)
        elif len(X.shape) == 4:
            # convolutional layer
            N, C, H, W = X.shape
            x_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
            out_flat = self._forward_pass(x_flat)
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise ValueError("Invalid input shape")

    def _backward_pass(self, delta):
        # unfold the variables stored in cache
        xhat, gamma, xmu, ivar, sqrtvar, var = self.cache

        # get the dimensions of the input/output
        N, D = delta.shape

        # step 9
        dbeta = np.sum(delta, axis=0)
        dgammax = delta

        # step 8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step 7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step 6
        dsqrtvar = -1.0 / (sqrtvar ** 2) * divar

        # step 5
        dvar = 0.5 * 1.0 / np.sqrt(var + self.eps) * dsqrtvar

        # step 4
        dsq = 1.0 / N * np.ones((N, D)) * dvar

        # step 3
        dxmu2 = 2 * xmu * dsq

        # step 2
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step 1
        dx2 = 1.0 / N * np.ones((N, D)) * dmu

        # step 0
        dx = dx1 + dx2

        # update gradient values
        self._params["W"].update_grad("W", dgamma)
        self._params["b"].update_grad("b", dbeta)

        return dx

    def backward_pass(self, X):
        if len(X.shape) == 2:
            # regular layer
            return self._backward_pass(X)
        elif len(X.shape) == 4:
            # convolutional layer
            N, C, H, W = X.shape
            x_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
            out_flat = self._backward_pass(x_flat)
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise ValueError("Invalid input shape")

    def shape(self, x_shape):
        return x_shape
