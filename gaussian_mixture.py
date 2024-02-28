import numpy as np


class GaussianMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.pi = None
        self.log_likelihood = -np.inf

    def initialize(self, X):
        np.random.seed(0)
        random_idx = np.random.permutation(X.shape[0])
        self.means = X[random_idx[:self.n_components]]
        self.covariances = [np.cov(X.T) + 1e-6 * np.eye(X.shape[1]) for _ in range(self.n_components)]  # 正则化
        self.pi = np.ones(self.n_components) / self.n_components

    def gaussian_density(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        inv = np.linalg.inv(covariance)
        N = np.sqrt((2 * np.pi) ** n_features * det)
        exp = np.exp(-0.5 * np.sum((X - mean) @ inv * (X - mean), axis=1))
        return exp / N

    def e_step(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            gamma[:, k] = self.pi[k] * self.gaussian_density(X, self.means[k], self.covariances[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def m_step(self, X, gamma):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        Nk = np.sum(gamma, axis=0)
        self.means = np.dot(gamma.T, X) / Nk[:, np.newaxis]
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(gamma[:, k] * diff.T, diff) / Nk[k] + 1e-6 * np.eye(n_features)
        self.pi = Nk / n_samples

    def compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += np.sum(np.log(
                self.pi[k] * self.gaussian_density(X, self.means[k], self.covariances[k]) + 1e-6))
        return log_likelihood

    def fit(self, X):
        self.initialize(X)
        for i in range(self.max_iter):
            gamma = self.e_step(X)
            self.m_step(X, gamma)
            log_likelihood = self.compute_log_likelihood(X)
            if np.abs(log_likelihood - self.log_likelihood) < self.tol:
                break
            self.log_likelihood = log_likelihood

    def predict(self, X):
        gamma = self.e_step(X)
        return np.argmax(gamma, axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=50000, centers=50, random_state=42)
    gmm = GaussianMixture(n_components=50)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    for i in range(50):
        plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], label=f'Cluster {i + 1}')

    plt.legend()
    plt.show()
