import autograd.numpy as np

from neuralnet.NeuralNetwork import NeuralNetwork
from neuralnet.parameters import Parameters
from neuralnet.layers import Layer, ParamMixin


class ConvolutionalNeuralNetwork(Layer, ParamMixin):
    def __init__(self, n_filters=8, filter_shape=(3, 3), stride=(1, 1), padding=(0, 0), parameters=None):
        """A 2D convolutional layer.
        Input shape: (batch_size, channels, height, width)

        Parameters
        ----------
        n_filters : int
            Number of filters.
        filter_shape : tuple
            Shape of the filters.
        stride : tuple
            Stride of the filters.
        padding : tuple
            Padding of the input.
        """
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self._params = parameters
        if self._params is None:
            self._params = Parameters()

    def setup(self, X_shape):
        n_channels, self.height, self.width = X_shape[1:]

        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = self.n_filters
        self._params.setup_weights(W_shape, b_shape)

    def forward_pass(self, X):
        n_images, n_channels, height, width = self.shape(X.shape)
        self.last_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self._params['W'].reshape(self.n_filters, -1).T

        out = np.dot(self.col, self.col_W) + self._params['b']
        out = out.reshape(n_images, height, width, -1).transpose(0, 3, 1, 2)
        return out

    def backward_pass(self, delta):
        delta = delta.transpose(0, 2, 3, 1).reshape(-1, self.n_filters)
        d_W = np.dot(self.col.T, delta).transpose(1, 0).reshape(self._params['W'].shape)
        d_b = np.sum(delta, axis=0)
        self._params.update_grad("b", d_b)
        self._params.update_grad("W", d_W)

        d_c = np.dot(delta, self.col_W.T)
        return column_to_image(d_c, self.last_input.shape, self.filter_shape, self.stride, self.padding)

    def shape(self, x_shape):
        height, width = convolution_shape(self.height, self.width, self.filter_shape, self.stride, self.padding)
        return x_shape[0], self.n_filters, height, width


class MaxPooling(Layer):
    def __init__(self, pool_shape=(2, 2), stride=(1, 1), padding=(0, 0)):
        """Max pooling layer.
        Input shape: (batch_size, channels, height, width)
        Output shape: (batch_size, channels, height // pool_shape[0], width // pool_shape[1])

        Parameters
        ----------
        pool_shape : tuple
            Shape of the pooling window.
        stride : tuple
            Stride of the pooling window.
        padding : tuple
            Padding of the input.
        """
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

    def forward_pass(self, X):
        self.last_input = X

        out_height, out_width = pooling_shape(self.pool_shape, X.shape, self.stride)
        n_images, n_channels, in_height, in_width = X.shape

        col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        col = col.reshape(-1, self.pool_shape[0] * self.pool_shape[1])

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        self.arg_max = arg_max
        return out.reshape(n_images, out_height, out_width, n_channels).transpose(0, 3, 1, 2)

    def backward_pass(self, delta):
        delta = delta.transpose(0, 2, 3, 1)

        pool_size = self.pool_shape[0] * self.pool_shape[1]
        y_max = np.zeros((delta.size, pool_size))
        y_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
        y_max = y_max.reshape(delta.shape + (pool_size,))

        dcol = y_max.reshape(y_max.shape[0] * y_max.shape[1] * y_max.shape[2], -1)
        return column_to_image(dcol, self.last_input.shape, self.pool_shape, self.stride, self.padding)

    def shape(self, x_shape):
        h, w = convolution_shape(x_shape[2], x_shape[3], self.pool_shape, self.stride, self.padding)
        return x_shape[0], x_shape[1], h, w


class Flatten(Layer):
    """Flattens multidimensional input to 2D matrix."""

    def forward_pass(self, X):
        self.last_input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward_pass(self, delta):
        return delta.reshape(self.last_input_shape)

    def shape(self, x_shape):
        return x_shape[0], np.prod(x_shape[1:])


def image_to_column(images, filter_shape, stride, padding):
    """Rearrange image blocks into columns.

        Parameters
        ----------

        filter_shape : tuple(height, width)
        images : np.array, shape (n_images, n_channels, height, width)
        padding: tuple(height, width)
        stride : tuple (height, width)

        """
    n_images, n_channels, height, width = images.shape
    f_height, f_width = filter_shape
    out_height, out_width = convolution_shape(height, width, (f_height, f_width), stride, padding)
    images = np.pad(images, ((0, 0), (0, 0), padding, padding), mode="constant")

    col = np.zeros((n_images, n_channels, f_height, f_width, out_height, out_width))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            col[:, :, y, x, :, :] = images[:, :, y: y_bound: stride[0], x: x_bound: stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_images * out_height * out_width, -1)
    return col


def column_to_image(columns, images_shape, filter_shape, stride, padding):
    """Rearrange columns into image blocks.

        Parameters
        ----------
        columns
        images_shape : tuple(n_images, n_channels, height, width)
        filter_shape : tuple(height, _width)
        stride : tuple(height, width)
        padding : tuple(height, width)
        """
    n_images, n_channels, height, width = images_shape
    f_height, f_width = filter_shape

    out_height, out_width = convolution_shape(height, width, (f_height, f_width), stride, padding)
    columns = columns.reshape(n_images, out_height, out_width, n_channels, f_height, f_width).transpose(
        0, 3, 4, 5, 1, 2
    )

    img_h = height + 2 * padding[0] + stride[0] - 1
    img_w = width + 2 * padding[1] + stride[1] - 1
    img = np.zeros((n_images, n_channels, img_h, img_w))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            img[:, :, y: y_bound: stride[0], x: x_bound: stride[1]] += columns[:, :, y, x, :, :]

    return img[:, :, padding[0]: height + padding[0], padding[1]: width + padding[1]]


def convolution_shape(img_height, img_width, filter_shape, stride, padding):
    """Calculate output shape for convolution layer."""
    height = (img_height + 2 * padding[0] - filter_shape[0]) / float(stride[0]) + 1
    width = (img_width + 2 * padding[1] - filter_shape[1]) / float(stride[1]) + 1

    assert height % 1 == 0
    assert width % 1 == 0

    return int(height), int(width)


def pooling_shape(pool_shape, image_shape, stride):
    """Calculate output shape for pooling layer."""
    n_images, n_channels, height, width = image_shape

    height = (height - pool_shape[0]) / float(stride[0]) + 1
    width = (width - pool_shape[1]) / float(stride[1]) + 1

    assert height % 1 == 0
    assert width % 1 == 0

    return int(height), int(width)


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


if __name__ == "__main__":
    import logging
    from neuralnet.layers import Dense, Activation, Dropout
    from neuralnet.optimizers import AdaDelta
    from neuralnet.loss import accuracy
    from dataset.dataset import load_mnist

    logging.basicConfig(level=logging.DEBUG)

    X_train, X_test, y_train, y_test = load_mnist()

    X_train /= 255.0
    X_test /= 255.0

    y_train = one_hot(y_train.flatten())
    y_test = one_hot(y_test.flatten())

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = NeuralNetwork(
        layers=[
            ConvolutionalNeuralNetwork(n_filters=32, filter_shape=(3, 3), stride=(1, 1), padding=(1, 1)),
            Activation("relu"),
            ConvolutionalNeuralNetwork(n_filters=32, filter_shape=(3, 3), stride=(1, 1), padding=(1, 1)),
            Activation("relu"),
            MaxPooling(pool_shape=(2, 2), stride=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
            Activation("softmax")
        ],
        loss="binary_cross_entropy",
        metric="accuracy",
        batch_size=128,
        max_epochs=3,
        optimizer=AdaDelta()
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(accuracy(y_test, predictions))
