import numpy as np


def generate_pos_semdef_matrix(dim: int, rank: int):
    A = np.random.rand(rank, dim)
    B = np.dot(A.T, A)
    return B


class QP:
    def __init__(self, dim: int, rank: int = None, c: bool = False, seed: int = None):
        if dim < 1:
            raise ValueError("dim must be greater than 0")
        if rank is None:
            rank = dim
        if rank > dim:
            raise ValueError("rank must be less than or equal to dim")
        elif rank < 1:
            raise ValueError("rank must be greater or equal than 1")
        if seed:
            np.random.seed(seed)

        self.dim = dim
        self.Q = generate_pos_semdef_matrix(dim, rank)
        self.q = np.random.rand(dim)
        if c:
            self.c = np.random.rand(1)
        else:
            self.c = 0

    def evaluate(self, x):
        return np.dot(np.dot(x, self.Q), x) + np.dot(self.q, x) + self.c

    def derivative(self, x):
        return 2 * np.dot(self.Q, x) + self.q
