import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.rand(num_visible, num_hidden)
        self.visible_bias = np.random.rand(num_visible)
        self.hidden_bias = np.random.rand(num_hidden)
        self.n_samples = None

    def encode(self, X):
        return sigmoid(X @ self.weights + self.hidden_bias)

    def decode(self, X):
        return sigmoid(X @ self.weights.T + self.visible_bias)

    def gibbs_sample(self, X, max_cd):
        x = X
        for _ in range(max_cd):
            hidden = self.encode(x)
            hidden = np.random.binomial(1, hidden, (self.n_samples, self.num_hidden))
            x = self.decode(hidden)
            x = np.random.binomial(1, x, (self.n_samples, self.num_visible))
        return x

    def update(self, X, x_cd, learning_rate):
        v0 = X
        h0 = self.encode(v0)
        v1 = x_cd
        h1 = self.encode(v1)
        self.weights += learning_rate * (v0.T @ h0 - v1.T @ h1)
        self.visible_bias += learning_rate * np.mean(v0 - v1, axis=0)
        self.hidden_bias += learning_rate * np.mean(h0 - h1, axis=0)
        return

    def fit(self, X, max_step=100, max_cd=2, learning_rate=0.1):
        self.n_samples = X.shape[0]
        for _ in range(max_step):
            x_cd = self.gibbs_sample(X, max_cd)
            self.update(X, x_cd, learning_rate)
            error = np.sum((X - x_cd) ** 2) / self.n_samples / self.num_visible
            print(f"step {_}: error {error}")
        return

    def predict(self, X):
        h = self.encode(X)[0]
        states = h >= np.random.rand(len(h))
        return states.astype(int)


if __name__ == "__main__":
    rbm_model = RBM(6, 2)
    train_data = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]])
    rbm_model.fit(train_data, max_step=100, max_cd=1, learning_rate=0.1)
    print(rbm_model.weights, rbm_model.visible_bias, rbm_model.hidden_bias)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(rbm_model.predict(user))
