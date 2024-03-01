import numpy as np


class Regularizer:
    def __init__(self, C=0.01):
        self.C = C
        self._grad = None

    def _penalty(self, W):
        raise NotImplementedError

    def grad(self, W):
        raise NotImplementedError

    def __call__(self, W):
        return self.grad(W)


class L1(Regularizer):
    def _penalty(self, W):
        return self.C * np.sum(np.abs(W))

    def grad(self, W):
        if self._grad is None:
            self._grad = self.C * np.sign(W)
        return self._grad


class L2(Regularizer):
    def _penalty(self, W):
        return self.C * np.sum(W ** 2)

    def grad(self, W):
        if self._grad is None:
            self._grad = 2 * self.C * W
        return self._grad


class ElasticNet(Regularizer):
    """Linear combination of L1 and L2 regularization"""
    def __init__(self, C=0.01, l1_ratio=0.5):
        super().__init__(C)
        self.l1_ratio = l1_ratio
        self.l1 = L1(C * l1_ratio)
        self.l2 = L2(C * (1 - l1_ratio))

    def _penalty(self, W):
        return self.l1._penalty(W) + self.l2._penalty(W)

    def grad(self, W):
        return self.l1.grad(W) + self.l2.grad(W)
