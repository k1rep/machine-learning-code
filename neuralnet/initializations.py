import numpy as np


def normal(shape, scale=0.05):
    return np.random.normal(0, scale, shape)


def uniform(shape, scale=0.05):
    return np.random.uniform(-scale, scale, shape)


def zero(shape):
    return np.zeros(shape)


def one(shape):
    return np.ones(shape)


def orthogonal(shape, scale=0.5):
    if len(shape) < 2:
        raise ValueError('Only shapes of length 2 or more are supported.')
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]


def _glorot_fan(shape):
    assert len(shape) >= 2

    if len(shape) == 4:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in , fan_out = shape[:2]
    return float(fan_in), float(fan_out)


def glorot_normal(shape):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, s)


def glorot_uniform(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, s)


def he_normal(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2.0 / fan_in)
    return normal(shape, s)


def he_uniform(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6.0 / fan_in)
    return uniform(shape, s)


def get_initializer(name):
    try:
        return globals()[name]
    except Exception:
        raise ValueError('Unknown initializer: %s' % name)
