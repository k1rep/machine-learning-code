import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel
        self.alpha = None
        self.b = None
        self.C = None
        self.X = None
        self.Y = None
        self.m = None
        self.n = None
        self.E = None
        self.sigma = None

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels

        self.b = 0.0

        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        self.C = 1.0  # 松弛变量

    def _KKT(self, i):
        """KKT条件"""
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def _g(self, i):
        """g(x) = sum_{j=1}^N(alpha_j * y * kernel(x, x_j)) + b"""
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r

    def kernel(self, x1, x2):
        """核函数"""
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2
        elif self._kernel == 'rbf':
            return np.exp(-sum([(x1[k] - x2[k]) ** 2 for k in range(self.n)]) / (2 * self.sigma ** 2))
        return 0

    def _E(self, i):
        """g(x) - y"""
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        """初始化alpha"""
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
        return -1, -1

    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):
        self.init_args(features, labels)

        for t in range(self.max_iter):
            i1, i2 = self._init_alpha()
            if i1 == -1:
                break

            E1 = self.E[i1]
            E2 = self.E[i2]
            alpha_old_1 = self.alpha[i1].copy()
            alpha_old_2 = self.alpha[i2].copy()
            if self.Y[i1] != self.Y[i2]:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            else:
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])
            if L == H:
                continue

            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                continue

            alpha_new_2 = alpha_old_2 + self.Y[i2] * (E1 - E2) / eta
            alpha_new_2 = self._compare(alpha_new_2, L, H)
            alpha_new_1 = alpha_old_1 + self.Y[i1] * self.Y[i2] * (alpha_old_2 - alpha_new_2)

            b1 = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha_new_1 - alpha_old_1) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha_new_2 - alpha_old_2) + self.b
            b2 = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha_new_1 - alpha_old_1) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha_new_2 - alpha_old_2) + self.b
            if 0 < alpha_new_1 < self.C:
                b = b1
            elif 0 < alpha_new_2 < self.C:
                b = b2
            else:
                b = (b1 + b2) / 2

            self.alpha[i1] = alpha_new_1
            self.alpha[i2] = alpha_new_2
            self.b = b
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        return 'train done!'

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVM(max_iter=200)
    svm2 = LinearSVC(max_iter=200)
    svm.fit(X_train, y_train)
    svm2.fit(X_train, y_train)
    print("SVM score: ", round(svm.score(X_test, y_test), 4))
    print("SVC score: ", round(svm2.score(X_test, y_test), 4))
