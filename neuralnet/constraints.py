import numpy as np

EPSILON = 1e-8


class Constraint:
    def clip(self, p):
        return p


class MaxNorm(Constraint):
    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis

    def clip(self, p):
        norms = np.linalg.norm(p, axis=self.axis)
        desired = (norms < self.max_value)
        return p * desired + p * (self.max_value / (norms + EPSILON)) * (1 - desired)


class NonNeg(Constraint):
    def clip(self, p):
        return np.maximum(p, 0)


class SmallNorm(Constraint):
    def clip(self, p, threshold=5):
        return np.clip(p, -threshold, threshold)


class UnitNorm:
    def __init__(self, axis=0):
        self.axis = axis

    def clip(self, p):
        return p / (np.linalg.norm(p, axis=self.axis) + EPSILON)
