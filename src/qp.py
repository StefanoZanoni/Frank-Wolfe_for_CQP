import numpy as np


def generate_Q(dim: int, rank: float, eccentricity: float) -> np.ndarray:
    """
    Generate a positive semi-definite matrix Q of size m x n, where m = rank * n.

    Parameters:
    dim (int): The dimension of the matrix Q.
    rank (float): The rank of the matrix Q, expressed as a fraction of the dimension.
    eccentricity (float): The eccentricity of the matrix Q.

    Returns:
    Q (ndarray): The generated positive semi-definite matrix Q.
    """

    A = np.random.rand(max(round(rank * dim), 1), dim)
    Q = np.dot(A.T, A)
    D, V = np.linalg.eig(Q)
    D = np.sort(D)

    if D[0] > 1e-14:
        l = D[0] * np.ones(dim) + (D[0] / (D[dim-1] - D[0])) * (2 * eccentricity / (1 - eccentricity)) * (D - D[0] * np.ones(dim))
        Q = np.linalg.multi_dot([V, np.diag(l), V.T])
    return Q


def generate_q(dim: int, Q: np.ndarray, active: float) -> np.ndarray:
    """
    Generates a random vector 'z' based on the given parameters and returns the dot product of '-Q' and 'z'.

    Parameters:
    - dim (int): The dimension of the vector q.
    - Q (np.ndarray): The matrix Q.
    - active (float): The fraction of active constraints.

    Returns:
    - np.ndarray: The q vector.
    """

    # in our case, the upper bound of the box is 1 so u was simply omitted

    z = np.zeros(dim)
    outb = np.random.rand(dim) <= active
    lr = np.random.rand(dim) <= 0.5
    l = np.logical_and(outb, lr)
    r = np.logical_and(outb, np.logical_not(lr))
    z[l] = -np.random.rand(np.sum(l))
    z[r] = (1 + np.random.rand(np.sum(r)))
    outb = np.logical_not(outb)
    z[outb] = np.random.rand(np.sum(outb))

    return -np.dot(Q, z)


class QP:
    """
    Quadratic Problem (QP) class.

    Args:
        dim (int): The dimension of the problem.
        rank (float, optional): The rank of the problem defined as a percentage of the dimension.
        Default to 1.
        eccentricity (float, optional): The eccentricity of the problem.
        Default to 0.9.
        active (float, optional): The percentage of active constraints.
        Default to 1.
        c (bool, optional): Whether to include a random constant term.
        Defaults to False.
        seed (int, optional): The seed for the random number generator.
        Defaults to None.

    Raises:
        ValueError: If `dim` is less than 1.
        ValueError: If `rank` is less than 0.
        ValueError: If `eccentricity` is less than 0.
        ValueError: If `eccentricity` is greater than 1.
        ValueError: If `active` is less than 0.
        ValueError: If `active` is greater than 1.

    Attributes:
        dim (int): The dimension of the problem.
        _Q (ndarray): The quadratic matrix.
        _subQ (ndarray): The submatrix of Q.
        _q (ndarray): The linear term.
        _subq (ndarray): The sub vector of q.
        c (float): The constant term.

    Methods:
        evaluate(x): Evaluate the quadratic function at point x.
        derivative(x): Computes the derivative of the quadratic function at point x.
        minimum(): Returns the minimum value of the quadratic function.
        set_subproblem(indexes): Sets the subproblem by selecting specific indexes.
        get_Q(): Returns the submatrix of Q.
        get_q(): Returns the sub vector of q.
    """

    def __init__(self, dim: int, rank: float = 1, eccentricity: float = 0.9, active: float = 1,
                 c: bool = False, seed: int = None) -> None:
        if dim < 1:
            raise ValueError("dim must be greater than 0")
        if rank <= 0:
            raise ValueError("rank must be greater than 0")
        if rank > 1:
            rank = 1
        if eccentricity < 0:
            raise ValueError("eccentricity must be greater or equal than 0")
        elif eccentricity >= 1:
            raise ValueError("eccentricity must be less than 1")
        if active < 0:
            raise ValueError("active must be greater or equal than 0")
        elif active > 1:
            raise ValueError("active must be less or equal than 1")
        if seed:
            np.random.seed(seed)

        self.dim = dim
        self._Q = generate_Q(dim, rank, eccentricity)
        self._subQ = self._Q
        self._q = generate_q(dim, self._Q, active)
        self._subq = self._q
        if c:
            self._c = np.random.uniform(0, 1)
        else:
            self._c = 0

        if seed:
            np.random.seed()

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluates the quadratic function at point x.

        Args:
            x (ndarray): The point at which to evaluate the function.

        Returns:
            float: The value of the quadratic function at point x.

        """

        return (np.linalg.multi_dot([x.T, self._subQ, x]) / 2) + np.dot(self._subq.T, x) + self._c

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the quadratic function at point x.

        Args:
            x (ndarray): The point at which to compute the derivative.

        Returns:
            ndarray: The derivative of the quadratic function at point x.

        """

        return np.dot(self._subQ, x) + self._subq

    def minimum(self):
        """
        Returns the minimum value of the quadratic function.
        """

        x_minimum = -np.dot(np.linalg.pinv(self._subQ), self._subq)
        return self.evaluate(x_minimum)

    def set_subproblem(self, indexes: list):
        """
        Sets the subproblem by selecting specific indexes.

        Args:
            indexes (ndarray): The indexes to select for the subproblem.
        """

        self._subQ = self._Q[indexes][:, indexes]
        self._subq = self._q[indexes]

    def get_Q(self):
        return self._subQ

    def get_q(self):
        return self._subQ
