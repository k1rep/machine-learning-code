import autograd.numpy as np


from neuralnet.layers import Layer, ParamMixin
from neuralnet.lstm import LSTM


class BiLSTM(Layer, ParamMixin):
    def __init__(self, hidden_dim, activation='tanh', inner_init='orthogonal', parameters=None, return_sequences=True):
        # Initialize parameters for both forward and backward LSTM layers
        self.forward_lstm = LSTM(hidden_dim, activation, inner_init, parameters, return_sequences)
        self.backward_lstm = LSTM(hidden_dim, activation, inner_init, parameters, return_sequences)
        self.return_sequences = return_sequences

    def setup(self, x_shape):
        # Setup both forward and backward LSTM layers
        self.forward_lstm.setup(x_shape)
        self.backward_lstm.setup(x_shape)

    def forward_pass(self, X):
        # Forward pass for the forward LSTM
        forward_outputs = self.forward_lstm.forward_pass(X)

        # Reverse the input for the backward LSTM pass
        X_reversed = np.flip(X, axis=1)
        backward_outputs = self.backward_lstm.forward_pass(X_reversed)

        # Reverse the outputs from the backward pass to align with the original sequence order
        backward_outputs = np.flip(backward_outputs, axis=1)

        outputs = np.concatenate((forward_outputs, backward_outputs), axis=-1)

        return outputs

    def backward_pass(self, delta):
        # Assuming delta has the shape of the output from forward_pass
        # Split the gradients for the forward and backward LSTM outputs
        if self.return_sequences:
            # If return_sequences is True, delta is split along the last dimension
            forward_delta, backward_delta = np.split(delta, 2, axis=-1)
        else:
            # If return_sequences is False, reshape delta to split the gradients
            forward_delta = delta[:, :self.forward_lstm.hidden_dim]
            backward_delta = delta[:, self.forward_lstm.hidden_dim:]

        # For the backward LSTM, we need to reverse the gradient sequence to match the reversed input
        backward_delta_reversed = np.flip(backward_delta, axis=1)

        # Call backward_pass on both LSTM layers
        forward_grads = self.forward_lstm.backward_pass(forward_delta)
        backward_grads = self.backward_lstm.backward_pass(backward_delta_reversed)

        # Combine gradients from both directions if necessary
        # Note: This step depends on how you plan to update your parameters
        # and whether the forward and backward LSTMs share parameters or not.
        # In this simplified example, it's assumed they don't share parameters
        # and thus each backward pass independently updates its own parameters.

        # Return combined or individual gradients if needed
        return forward_grads, backward_grads

    def shape(self, x_shape):
        # Adjust the output shape based on the return_sequences flag and the fact that outputs are concatenated
        output_dim = self.forward_lstm.hidden_dim + self.backward_lstm.hidden_dim
        if self.return_sequences:
            return x_shape[0], x_shape[1], output_dim
        else:
            return x_shape[0], output_dim


if __name__ == '__main__':
    import numpy as np

    from neuralnet.layers import Activation, EmbeddingLayer, Dense
    from neuralnet.optimizers import Adam
    from neuralnet.NeuralNetwork import NeuralNetwork
    from neuralnet.loss import accuracy

    from tensorflow.keras.preprocessing import sequence
    from keras.datasets import imdb

    # 加载数据
    max_features = 20000  # 词汇表大小
    maxlen = 100  # 序列最大长度
    batch_size = 32

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    # 序列填充
    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    model = NeuralNetwork(
        layers=[
            EmbeddingLayer(max_features, 128),
            BiLSTM(64, return_sequences=False),
            Dense(1),
            Activation('sigmoid')
        ],
        loss='binary_cross_entropy',
        optimizer=Adam(),
        metric='binary_cross_entropy',
        batch_size=64,
        max_epochs=30
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Accuracy:', accuracy(y_test, predictions))
