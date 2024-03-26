import numpy as np


class C:
    def __init__(self, A: np.ndarray, b: np.ndarray, ineq: bool = True):

        self.A = A
        self.num_constraints = A.shape[0]

        self.b = b
        self.box_max = np.max(b)
        self.ineq = ineq

    def evaluate(self, x):
        if self.ineq:
            return np.dot(self.A, x) - self.b <= 0
        else:
            return np.dot(self.A, x) - self.b == 0


def create_A(n: int, index_set: list):
    A = [[0 for _ in range(n)], [-1 for _ in range(n)]]
    for i in index_set:
        A[0][i] = 1

    return np.array(A)


def create_b():
    return np.array([1, 0])
