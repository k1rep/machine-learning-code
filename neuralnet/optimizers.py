import time
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def batch_iterator(X, batch_size=64):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size
    batch_end = 0
    for b in range(n_batches):
        batch_begin = b * batch_size
        batch_end = batch_begin + batch_size
        X_batch = X[batch_begin:batch_end]
        yield X_batch

    if n_batches * batch_size < n_samples:
        yield X[batch_end:]


class Optimizer:
    def optimize(self, network):
        loss_history = []
        for i in range(network.max_epochs):
            if network.shuffle:
                network.shuffle_dataset()

            start_time = time.time()
            loss = self.train_epoch(network)
            loss_history.append(loss)
            if network.verbose:
                msg = "Epoch:%s, train loss: %s" % (i, loss)
                if network.log_metric:
                    msg += ", train %s: %s" % (network.metric_name, network.error())
                msg += ", elapsed: %s sec." % (time.time() - start_time)
                logging.info(msg)
        return loss_history

    def update(self, network):
        raise NotImplementedError

    def train_epoch(self, network):
        losses = []
        X_batch = batch_iterator(network.X, network.batch_size)
        y_batch = batch_iterator(network.y, network.batch_size)

        batch = zip(X_batch, y_batch)
        if network.verbose:
            batch = tqdm(batch, total=int(np.ceil(network.n_samples / network.batch_size)))

        for X, y in batch:
            loss = np.mean(network.update(X, y))
            self.update(network)
            losses.append(loss)

        epoch_loss = np.mean(losses)
        return epoch_loss

    def train_batch(self, network, X, y):
        loss = np.mean(network.update(X, y))
        self.update(network)
        return loss

    def setup(self, network):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False):
        self.lr = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None
        self.iteration = 0

    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))

        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                update = self.momentum * self.velocity[i][n] - lr * grad
                self.velocity[i][n] = update
                if self.nesterov:
                    update = self.momentum * update - lr * grad
                layer.parameters.step(n, update)

        self.iteration += 1

    def setup(self, network):
        self.velocity = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.velocity[i][n] = np.zeros_like(layer.parameters[n])


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.m = None
        self.v = None
        self.iteration = 0

    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))

        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.m[i][n] = self.beta1 * self.m[i][n] + (1 - self.beta1) * grad
                self.v[i][n] = self.beta2 * self.v[i][n] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[i][n] / (1 - self.beta1 ** (self.iteration + 1))
                v_hat = self.v[i][n] / (1 - self.beta2 ** (self.iteration + 1))
                update = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                layer.parameters.step(n, -update)

        self.iteration += 1

    def setup(self, network):
        self.m = defaultdict(dict)
        self.v = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.m[i][n] = np.zeros_like(layer.parameters[n])
                self.v[i][n] = np.zeros_like(layer.parameters[n])


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.cache[i][n] = self.decay * self.cache[i][n] + (1 - self.decay) * grad ** 2
                update = self.lr * grad / (np.sqrt(self.cache[i][n]) + self.epsilon)
                layer.parameters.step(n, -update)

    def setup(self, network):
        self.cache = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.cache[i][n] = np.zeros_like(layer.parameters[n])


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.cache[i][n] += grad ** 2
                update = self.lr * grad / (np.sqrt(self.cache[i][n]) + self.epsilon)
                layer.parameters.step(n, -update)

    def setup(self, network):
        self.cache = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.cache[i][n] = np.zeros_like(layer.parameters[n])


class AdaDelta(Optimizer):
    def __init__(self, decay=0.95, epsilon=1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
        self.accu = None
        self.delta = None

    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] = self.decay * self.accu[i][n] + (1 - self.decay) * grad ** 2
                update = (np.sqrt(self.delta[i][n] + self.epsilon) / np.sqrt(self.accu[i][n] + self.epsilon)) * grad
                self.delta[i][n] = self.decay * self.delta[i][n] + (1 - self.decay) * update ** 2
                layer.parameters.step(n, -update)

    def setup(self, network):
        self.accu = defaultdict(dict)
        self.delta = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])
                self.delta[i][n] = np.zeros_like(layer.parameters[n])


class AdaMax(Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.m = None
        self.u = None
        self.iteration = 0

    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))

        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.m[i][n] = self.beta1 * self.m[i][n] + (1 - self.beta1) * grad
                self.u[i][n] = np.maximum(self.beta2 * self.u[i][n], np.abs(grad))
                update = lr * self.m[i][n] / (1 - self.beta1 ** (self.iteration + 1)) / (self.u[i][n] + self.epsilon)
                layer.parameters.step(n, -update)

        self.iteration += 1

    def setup(self, network):
        self.m = defaultdict(dict)
        self.u = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.m[i][n] = np.zeros_like(layer.parameters[n])
                self.u[i][n] = np.zeros_like(layer.parameters[n])


class Yogi(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3, decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.m = None
        self.v = None
        self.iteration = 0

    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))

        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.m[i][n] = self.beta1 * self.m[i][n] + (1 - self.beta1) * grad
                self.v[i][n] = self.v[i][n] + (1 - self.beta2) * (grad ** 2 - self.v[i][n])
                update = lr * self.m[i][n] / (np.sqrt(self.v[i][n]) + self.epsilon)
                layer.parameters.step(n, -update)

        self.iteration += 1

    def setup(self, network):
        self.m = defaultdict(dict)
        self.v = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.m[i][n] = np.zeros_like(layer.parameters[n])
                self.v[i][n] = np.zeros_like(layer.parameters[n])
