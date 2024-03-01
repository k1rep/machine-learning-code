import autograd.numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def elu(x, alpha=1.0):
    return x if x > 0 else alpha * (np.exp(x) - 1)


def selu(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    return scale * elu(x, alpha)


def prelu(x, alpha):
    return np.maximum(0, x) + alpha * np.minimum(0, x)


def softplus(x):
    return np.log(1 + np.exp(x))


def swish(x):
    return x / (1 + np.exp(-x))


def mish(x):
    return x * np.tanh(softplus(x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def linear(x):
    return x


def step(x):
    return np.array(x > 0, dtype=np.int)


def hard_sigmoid(x):
    return np.maximum(0, np.minimum(1, 0.2 * x + 0.5))


def get_activations(name):
    """Return the activation function from its name."""
    try:
        return globals()[name]
    except KeyError:
        raise ValueError("Invalid activation function.")
