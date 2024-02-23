import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idxs]
        sorted_eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = sorted_eigenvectors[0:self.n_components]
        # explained variance
        self.explained_variance_ratio_ = sorted_eigenvalues / sum(sorted_eigenvalues)

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)

    def explained_variance_ratio(self):
        return self.explained_variance_ratio_


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_digits()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Shape of X:', X.shape)
    print('Shape of transformed X:', X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    print('Explained variance:', pca.explained_variance_ratio())

    plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 10))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio()) + 1), pca.explained_variance_ratio(), marker='o',
             linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Manual PCA Components')
    plt.grid(True)
    plt.show()
    # 前10个主成分的方差占比（累计90%）
    print(sum(pca.explained_variance_ratio()[:21]))

