import numpy as np


def generate_Q(dim: int, rank: int, eccentricity: float):
    """
    Generate a positive semi-definite matrix Q of size dim x dim.

    Parameters:
    dim (int): The dimension of the matrix Q.
    rank (int): The rank of the matrix A used to generate Q.
    eccentricity (float): The eccentricity parameter used to modify Q.

    Returns:
    Q (ndarray): The generated positive semi-definite matrix Q.
    """
    A = np.random.uniform(-1, 1, (rank, dim))
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
    """
    Generates a random vector 'z' based on the given parameters and returns the dot product of '-Q' and 'z'.

    Parameters:
    - dim (int): The dimension of the vector 'z'.
    - Q (np.ndarray): The matrix 'Q' used in the dot product calculation.
    - active (float): The probability of each element in 'z' being active.

    Returns:
    - np.ndarray: The result of the dot product between '-Q' and 'z'.
    """
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
    """
    Quadratic Programming (QP) class.

    Args:
        dim (int): The dimension of the problem.
        rank (int, optional): The rank of the problem. Defaults to None.
        eccentricity (float, optional): The eccentricity of the problem. Defaults to 0.9.
        active (float, optional): The activity level of the problem. Defaults to 1.
        c (bool, optional): Whether to include a random constant term. Defaults to False.
        seed (int, optional): The seed for the random number generator. Defaults to None.

    Raises:
        ValueError: If `dim` is less than 1.
        ValueError: If `rank` is greater than `dim`.
        ValueError: If `rank` is less than 1.

    Attributes:
        dim (int): The dimension of the problem.
        Q (ndarray): The quadratic matrix.
        subQ (ndarray): The submatrix of Q.
        q (ndarray): The linear term.
        subq (ndarray): The subvector of q.
        c (float): The constant term.

    Methods:
        evaluate(x): Evaluate the quadratic function at point x.
        derivative(x): Computes the derivative of the quadratic function at point x.
        set_subproblem(indexes): Sets the subproblem by selecting specific indexes.

    """

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
        """
        Evaluates the quadratic function at point x.

        Args:
            x (ndarray): The point at which to evaluate the function.

        Returns:
            float: The value of the quadratic function at point x.

        """
        return np.dot(np.dot(x, self.subQ), x) + np.dot(self.subq, x) + self.c

    def derivative(self, x):
        """
        Computes the derivative of the quadratic function at point x.

        Args:
            x (ndarray): The point at which to compute the derivative.

        Returns:
            ndarray: The derivative of the quadratic function at point x.

        """
        return 2 * np.dot(self.subQ, x) + self.subq

    def set_subproblem(self, indexes):
        """
        Sets the subproblem by selecting specific indexes.

        Args:
            indexes (ndarray): The indexes to select for the subproblem.

        """
        self.subQ = self.Q[indexes][:, indexes]
        self.subq = self.q[indexes]
