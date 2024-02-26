from autograd import numpy as np
from autograd import elementwise_grad


def mean_squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_crossentropy(y_true, y_pred):
    return np.mean(-np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)) + (1 - y_true) * np.log(1 - np.clip(y_pred, 1e-15, 1 - 1e-15))))


class FactorizationMachine:
    def __init__(self, n_components=10, learning_rate=0.0001, n_iter=100, init_stdev=0.1,
                 reg_v=0.1, reg_w=0.5, reg_bias=0.0):
        self.bias = None
        self.V = None
        self.weights = None
        self.n_samples = None
        self.n_features = None
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.init_stdev = init_stdev
        self.reg_v = reg_v
        self.reg_w = reg_w
        self.reg_bias = reg_bias
        self.loss = None
        self.loss_grad = None
        self.X = None
        self.y = None

    def _setup_input(self, X, y):
        y_required = True
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.size == 0:
            raise ValueError("Got an empty matrix.")
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        self.X = X
        if y_required:
            if y is None:
                raise ValueError("Missed required argument y")
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")
        self.y = y

    def fit(self, X, y):
        self._setup_input(X, y)
        self.weights = np.zeros(self.n_features)
        self.V = np.random.normal(scale=self.init_stdev, size=(self.n_features, self.n_components))
        self.bias = 0
        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            loss = self.loss_grad(y, y_pred)
            w_grad = np.dot(loss, X) / float(self.n_samples)
            self.weights -= self.learning_rate * w_grad + 2 * self.reg_w * self.weights
            self.bias -= self.learning_rate * (loss.mean() + 2 * self.reg_bias * self.bias)
            self._factor_step(loss)

    def _factor_step(self, loss):
        for ix, x in enumerate(self.X):
            for i in range(self.n_features):
                v_grad = loss[ix] * (x.dot(self.V).dot(x[i])[0] - self.V[i] * x[i] ** 2)
                self.V[i] -= self.learning_rate * v_grad + (2 * self.reg_v * self.V[i])

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        interactions = np.dot(X, self.V) ** 2 - np.dot(X ** 2, self.V ** 2)
        return linear + interactions.sum(axis=1) / 2.0


class FMRegressor(FactorizationMachine):
    def __init__(self):
        super(FMRegressor, self).__init__()
        self.loss = mean_squared_loss
        self.loss_grad = elementwise_grad(mean_squared_loss)

    def fit(self, X, y):
        super(FMRegressor, self).fit(X, y)


class FMClassifier(FactorizationMachine):
    def __init__(self):
        super(FMClassifier, self).__init__()
        self.loss = binary_crossentropy
        self.loss_grad = elementwise_grad(binary_crossentropy)

    def fit(self, X, y):
        super(FMClassifier, self).fit(X, y)

    def predict(self, X):
        predictions = super(FMClassifier, self).predict(X)
        return np.sign(predictions)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = FMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = FMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
