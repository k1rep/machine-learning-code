import numpy as np


class KMeans:
    def __init__(self, K, max_iters=100, init="random"):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.init = init

    def fit(self, X):
        if self.init == "random":
            self.centroids = self._init_random(X)
        elif self.init == "kmeans++":
            self.centroids = self._init_kmeanspp(X)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(X)
            prev_centroids = self.centroids
            self.centroids = self._update_centroids(X)
            if self._is_converged(prev_centroids, self.centroids):
                break

    def predict(self, X):
        return self._create_clusters(X)

    def _init_random(self, X):
        n_samples, _ = X.shape
        return X[np.random.choice(n_samples, self.K, replace=False)]

    def _init_kmeanspp(self, X):
        n_samples, _ = X.shape
        centroids = [X[np.random.choice(n_samples)]]
        for _ in range(1, self.K):
            dist = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            probs = dist / dist.sum()
            centroids.append(X[np.random.choice(n_samples, p=probs)])
        return centroids

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(X):
            closest_centroid = self._closest_centroid(sample, self.centroids)
            clusters[closest_centroid].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        return np.argmin([np.linalg.norm(sample - centroid) for centroid in centroids])

    def _update_centroids(self, X):
        centroids = []
        for cluster in self.clusters:
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids.append(cluster_mean)
        return centroids

    def _is_converged(self, prev_centroids, centroids):
        return np.array_equal(prev_centroids, centroids)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(centers=10, n_samples=500, random_state=42)
    kmeans = KMeans(K=10, max_iters=100, init="kmeans++")
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    for cluster in clusters:
        plt.scatter(X[cluster][:, 0], X[cluster][:, 1])

    for centroid in kmeans.centroids:
        plt.scatter(centroid[0], centroid[1], s=130, marker="x")

    plt.show()


