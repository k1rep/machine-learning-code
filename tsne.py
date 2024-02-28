import numpy as np


class TSNE:
    def __init__(self, no_dims=2, initial_dims=50, perplexity=30.0, n_iter=1000, learning_rate=200.0, random_state=None):
        self.no_dims = no_dims
        self.initial_dims = initial_dims
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _Hbeta(self, D=np.array([]), beta=1.0):
        P = np.exp(-D.copy() * beta)
        sumP = np.sum(P)
        if sumP == 0:
            sumP = np.finfo(float).eps
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def _x2p(self, X=np.array([]), tol=1e-5):
        print("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(self.perplexity)

        for i in range(n):
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            H, thisP = self._Hbeta(Di, beta[i, 0])

            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
                H, thisP = self._Hbeta(Di, beta[i, 0])
                Hdiff = H - logU
                tries += 1
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

        return P

    def fit_transform(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        (n, d) = X.shape
        momentum = 0.5
        final_momentum = 0.8
        eta = self.learning_rate
        min_gain = 0.01

        Y = np.random.randn(n, self.no_dims)
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        P = self._x2p(X, 1e-5)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.
        P = np.maximum(P, 1e-12)

        for iter in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (Y[i, :] - Y), 0)

            if iter < 20:
                momentum = 0.5
            else:
                momentum = 0.8
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            if iter == 100:
                P = P / 4.

                if iter % 100 == 0:
                    C = np.sum(P * np.log(P / Q))
                    print("Iteration %d: error is %f" % (iter, C))

                if iter == 1000:
                    P = P * 4.

        return Y


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt

    digits = load_digits()
    X = digits.data
    y = digits.target

    tsne = TSNE(no_dims=2, initial_dims=64, perplexity=30.0, n_iter=1000, learning_rate=200.0, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10")
    plt.colorbar()
    plt.show()
