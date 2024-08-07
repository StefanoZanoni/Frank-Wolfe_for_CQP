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

    # All eigenvalues are real and >= 0 due to the structure of Q. However,
    # there might be small numerical errors that make them complex with a negligible imaginary part (e.g., 1e-17),
    D = D.real
    V = V.real
    # sort the eigenvalues and the corresponding eigenvectors in ascending order since np.linalg.eig does not
    # guarantee this
    sort_indices = np.argsort(D)
    D = D[sort_indices]
    V = V[:, sort_indices]
    # moreover, when this happens, even the real part might result slightly negative (e.g., -1e-17),
    # so we set it to 0
    D = np.maximum(D, 0)

    if D[0] > 1e-14:
        l = (D[0] * np.ones(dim) + (D[0] / (D[dim-1] - D[0] + 1e-14)) * (2 * eccentricity / (1 - eccentricity))
             * (D - D[0] * np.ones(dim)))
        Q = np.linalg.multi_dot([V, np.diag(l), V.T])
    return Q


def generate_q(dim: int, Q: np.ndarray, index_sets: list[list[int]], active: float) -> np.ndarray:
    """
    Generates the minimum unconstrained vector 'z'
    based on the given parameters and returns the dot product of '-Q' and 'z'.

    Parameters:
    - dim (int): The dimension of the vector q.
    - Q (np.ndarray): The matrix Q.
    - active (float): The fraction of active constraints.

    Returns:
    - np.ndarray: The q vector.
    """

    # in our case, the upper bound of the box is 1 and the lower bound is 0

    z = np.zeros(dim)

    # make z satisfy the equality constraint
    for I in index_sets:
        subset_size = len(I)
        # Generate random values for the elements within the subset
        random_values = np.random.rand(subset_size)
        # Normalize the values so that their sum is 1
        random_values /= np.sum(random_values)
        for idx, value in zip(I, random_values):
            z[idx] = value

    # probability of being outside the box
    outb = np.random.rand(dim) >= active

    # 50/50 of being left of the lower bound or right of the upper bound
    lr = np.random.rand(dim) <= 0.5
    l = np.logical_and(outb, lr)
    r = np.logical_and(outb, np.logical_not(lr))

    # a small random amount to the left of the lower bound
    z[l] -= 1 + np.random.rand(np.sum(l))
    # a small random amount to the right of the upper bound
    z[r] += 1 + np.random.rand(np.sum(r))

    return np.dot(-z.T, Q)


class QP:
    """
    Quadratic Problem (QP) class.

    Args:
        dim (int): The dimension of the problem.
        index_sets (list[list[int]]): The index sets of the problem.
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
        ValueError: If `rank` is less or equal than 0.
        ValueError: If `eccentricity` is less than 0.
        ValueError: If `eccentricity` is greater or equal than 1.
        ValueError: If `active` is less than 0.
        ValueError: If `active` is greater than 1.

    Attributes:
        dim (int): The dimension of the problem.
        _Q (ndarray): The quadratic matrix.
        _q (ndarray): The linear term.
        _c (float): The constant term.

    Methods:
        evaluate(x): Evaluate the quadratic function at point x.
        derivative(x): Computes the derivative of the quadratic function at point x.
        minimum_point(): Returns the point where the minimum value of the quadratic function is achieved.
        minimum(): Returns the minimum value of the quadratic function.
        get_Q(): Returns the matrix Q.
        get_q(): Returns the vector q.
        get_c(): Returns the constant term.
    """

    def __init__(self, dim: int, index_sets: list[list[int]], rank: float = 1, eccentricity: float = 0.9,
                 active: float = 1, c: bool = False, seed: int = None) -> None:
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
        self._q = generate_q(dim, self._Q, index_sets, active)
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

        return (np.linalg.multi_dot([x.T, self._Q, x]) / 2) + np.dot(self._q.T, x) + self._c

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the quadratic function at point x.

        Args:
            x (ndarray): The point at which to compute the derivative.

        Returns:
            ndarray: The derivative of the quadratic function at point x.

        """

        return np.dot(self._Q, x) + self._q

    def minimum_point(self) -> np.ndarray:
        """
        Returns the point where the minimum value of the quadratic function is achieved.
        """

        x_minimum = np.dot(np.linalg.pinv(self._Q), -self._q)
        return x_minimum

    def minimum(self) -> float:
        """
        Returns the minimum value of the quadratic function.
        """

        return self.evaluate(self.minimum_point())

    def get_Q(self) -> np.ndarray:
        return self._Q.copy()

    def get_q(self) -> np.ndarray:
        return self._q.copy()

    def get_c(self) -> float:
        return self._c
