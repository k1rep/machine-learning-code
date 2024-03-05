import numpy as np

EPS = 1e-15


def unhot(function):
    """Convert one-hot encoding to one column."""

    def wrapper(y_true, y_pred):
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return function(y_true, y_pred)

    return wrapper


@unhot
def classification_error(y_true, y_pred):
    """Calculate classification error."""
    return (y_true != y_pred).sum() / float(y_true.shape[0])


@unhot
def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return 1.0 - classification_error(y_true, y_pred)


def absolute_error(y_true, y_pred):
    """Calculate absolute error."""
    return np.abs(y_true - y_pred)


def squared_error(y_true, y_pred):
    """Calculate squared error."""
    return (y_true - y_pred) ** 2


def squared_log_error(y_true, y_pred):
    """Calculate squared log error."""
    return (np.log1p(y_true) - np.log1p(y_pred)) ** 2


def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error."""
    return np.mean(absolute_error(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error."""
    return np.mean(squared_error(y_true, y_pred))


def mse_grad(y_true, y_pred):
    """Calculate gradient of mean squared error."""
    return 2 * (y_pred - y_true) / y_true.shape[0]


def mean_squared_log_error(y_true, y_pred):
    """Calculate mean squared log error."""
    return np.mean(squared_log_error(y_true, y_pred))


def root_mean_squared_error(y_true, y_pred):
    """Calculate root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_mean_squared_log_error(y_true, y_pred):
    """Calculate root mean squared log error."""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def binary_cross_entropy(y_true, y_pred):
    """Calculate binary cross-entropy."""
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_grad(y_true, y_pred):
    """Calculate gradient of binary cross-entropy."""
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]


def hinge_loss(y_true, y_pred):
    """Calculate hinge loss."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def log_loss(y_true, y_pred):
    """Calculate log loss."""
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(y_true * np.log(y_pred))


# Aliases
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error


def get_loss(name):
    """Get loss function by name."""
    try:
        return globals()[name]
    except KeyError:
        raise ValueError('Invalid loss function name: {}'.format(name))
