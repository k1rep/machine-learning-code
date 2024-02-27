from autograd import numpy as np
from autograd import grad


def mean_squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class FactorizationMachine:
    def __init__(self, n_components=10, learning_rate=0.0001, n_iter=100, init_stdev=0.1,
                 reg_v=0.1, reg_w=0.8, reg_bias=0.0):
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
            self.n_samples, self.n_features = 1, len(X)
        else:
            self.n_samples, self.n_features = X.shape
        self.X = X
        if y_required:
            if y is None:
                raise ValueError("Missed required argument y")
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y

    def fit(self, X, y):
        self._setup_input(X, y)
        self.weights = np.zeros(self.n_features)
        self.V = np.random.normal(scale=self.init_stdev, size=(self.n_features, self.n_components))
        self.bias = 0
        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            loss = self.loss_grad(y, y_pred)
            w_grad = np.dot(X.T, loss) / self.n_samples
            self.weights -= self.learning_rate * (w_grad + 2 * self.reg_w * self.weights)
            self.bias -= self.learning_rate * (np.mean(loss) + 2 * self.reg_bias * self.bias)
            self._factor_step(loss)

    def _factor_step(self, loss):
        for i in range(self.n_features):
            # 计算 Xi * V[i]，这里 Xi 是特征 i 的值
            xiV = self.X[:, i:i + 1] * self.V[i]
            # 计算 (XV - XiV) * Xi，这是对于特征 i 的交互项梯度部分
            XV_minus_xiV = self.X.dot(self.V) - xiV
            grad_component = XV_minus_xiV * self.X[:, i:i + 1]
            v_grad = np.dot(grad_component.T, loss).reshape(self.n_components, ) / self.n_samples
            self.V[i] -= self.learning_rate * (v_grad + 2 * self.reg_v * self.V[i])

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        linear_part = np.dot(X, self.weights) + self.bias
        interaction_part = 0.5 * np.sum(np.power(X.dot(self.V), 2) - (X ** 2).dot(self.V ** 2), axis=1, keepdims=True)
        return linear_part + interaction_part.flatten()


class FMRegressor(FactorizationMachine):
    def __init__(self):
        super(FMRegressor, self).__init__()
        self.loss = mean_squared_loss
        self.loss_grad = grad(mean_squared_loss)

    def fit(self, X, y):
        super(FMRegressor, self).fit(X, y)


class FMClassifier(FactorizationMachine):
    def __init__(self):
        super(FMClassifier, self).__init__()
        self.loss = binary_crossentropy
        self.loss_grad = grad(binary_crossentropy)

    def fit(self, X, y):
        super(FMClassifier, self).fit(X, y)

    def predict(self, X):
        predictions = super(FMClassifier, self).predict(X)
        return np.round(1 / (1 + np.exp(-predictions)))


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
