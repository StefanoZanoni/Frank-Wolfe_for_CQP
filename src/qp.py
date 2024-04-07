import numpy as np


def generate_Q(dim: int, rank: int, eccentricity: float):
    A = np.random.uniform(-2, 2, (rank, dim))
    Q = np.dot(A.T, A)
    V, D = np.linalg.eig(Q)
    if D[0, 0] > 1e-14:
        d = np.diag(D)
        l = (d[0] * np.ones(dim) + (d[0] / (d[dim - 1] - d[0]))
             * (2 * eccentricity / (1 - eccentricity))
             * (d - d[0] * np.ones(dim)))
        Q = V * np.diag(l) * V.T
    return Q


def generate_q(dim: int, Q: np.ndarray, active: float):
    z = np.zeros(dim)
    outb = np.random.rand(dim) <= active
    lr = np.random.rand(dim) <= 0.5
    l = outb & lr
    r = outb & ~lr
    u = np.random.rand(dim)
    z[l] = -np.random.rand(np.sum(l)) * u[l]
    z[r] = (1 + np.random.rand(np.sum(r))) * u[r]
    outb = ~outb
    z[outb] = np.random.rand(np.sum(outb)) * u[outb]

    return np.dot(-Q, z)


class QP:
    def __init__(self, dim: int, rank: int = None, eccentricity: float = 0.9, active: float = 1, c: bool = False,
                 seed: int = None):
        if dim < 1:
            raise ValueError("dim must be greater than 0")
        if rank is None:
            rank = dim
        elif rank > dim:
            raise ValueError("rank must be less than or equal to dim")
        elif rank < 1:
            raise ValueError("rank must be greater or equal than 1")
        if seed:
            np.random.seed(seed)

        self.dim = dim
        self.Q = generate_Q(dim, rank, eccentricity)
        self.subQ = self.Q
        self.q = generate_q(dim, self.Q, active)
        self.subq = self.q
        if c:
            self.c = np.random.uniform(-1, 1)
        else:
            self.c = 0

    def evaluate(self, x):
        return np.dot(np.dot(x, self.subQ), x) + np.dot(self.subq, x) + self.c

    def derivative(self, x):
        return 2 * np.dot(self.subQ, x) + self.subq

    def set_subproblem(self, indexes):
        self.subQ = self.Q[indexes][:, indexes]
        self.subq = self.q[indexes]
