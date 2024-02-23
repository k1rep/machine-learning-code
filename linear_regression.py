import numpy as np


class L1Regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        return self.alpha * np.sign(w)


class L2Regularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 2) ** 2

    def grad(self, w):
        return self.alpha * w


class LinearRegression:
    def __init__(self, n_iter=1000, learning_rate=0.01, regularization=None, gradient=True):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.regularization = regularization if regularization else lambda x: 0
        self.regularization.grad = regularization.grad if regularization else lambda x: 0
        self.w = None
        self.b = None

    def init_weights(self, X):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

    def fit(self, X, y):
        self.init_weights(X)
        if self.gradient:
            # Gradient Descent
            for _ in range(self.n_iter):
                self._update_weights(X, y)
        else:
            # Normal Equation
            self._normal_equation(X, y)

    def _update_weights(self, X, y):
        y_pred = self.predict(X)
        w_grad = X.T.dot(y_pred - y) / X.shape[0]
        b_grad = np.mean(y_pred - y)
        w_reg = self.regularization.grad(self.w)
        self.w -= self.learning_rate * (w_grad + w_reg)
        self.b -= self.learning_rate * b_grad

    def _normal_equation(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_boston()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LinearRegression(n_iter=10000, learning_rate=0.001, regularization=L2Regularization(alpha=0.1))
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f'MSE: {round(mse, 4)}')
