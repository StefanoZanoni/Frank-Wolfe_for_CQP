import numpy as np


class Constraints:
    def __init__(self, A: np.ndarray, b: np.ndarray, ineq: bool = True):

        self.A = A
        self.b = b
        self.ineq = ineq
        self.num_constraints = len(b)

    def evaluate(self, x):
        if self.ineq:
            return np.dot(self.A, x) - self.b <= 0
        else:
            return np.dot(self.A, x) - self.b == 0


class BoxConstraints(Constraints):
    def __init__(self, A: np.ndarray, b: np.ndarray, ineq: bool = True):
        self.box_min = np.min(b)
        self.box_max = np.max(b)

        super().__init__(A, b, ineq)


def create_A(n: int, index_set: list):
    A = [[0 for _ in range(n)], [-1 for _ in range(n)]]
    for i in index_set:
        A[0][i] = 1

    return np.array(A)


def create_b():
    return np.array([1, 0])
