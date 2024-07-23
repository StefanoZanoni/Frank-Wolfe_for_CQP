import numpy as np
import copy


class Constraints:
    """
    Represents a set of linear constraints.

    Parameters:
    - A (np.ndarray): The coefficient matrix of the constraints.
    - b (np.ndarray): The right-hand side vector of the constraints.
    - K (int): The number of index sets.
    - n (int): The dimension of the problem.
    - ineq (bool, optional): Specifies whether the constraints are inequalities or equalities. Default is True.

    Attributes:
    - A (np.ndarray): The coefficient matrix of the constraints.
    - b (np.ndarray): The right-hand side vector of the constraints.
    - ineq (bool): Specifies whether the constraints are inequalities or equalities.
    - num_constraints (int): The number of constraints.
    - _tol (float): The tolerance for the constraints.
    - K (int): The number of index sets.
    - dim (int): The dimension of the problem.

    Methods:
    - evaluate(x): Evaluates the constraints at a given point x.
    - active_constraints(x): Returns the indexes of the active constraints at a given point x.
    - check_position(x): Checks the position of x in the feasible region.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, K: int, n: int, ineq: bool = True) -> None:
        self.A = A
        self.b = b
        self.ineq = ineq
        self.num_constraints = len(b)
        self._tol = 1e-8
        self.K = K
        self.dim = n

    def evaluate(self, x: np.ndarray) -> bool:
        """
        Evaluates the constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - result (bool): The result of evaluating the constraints.
        """

        if self.ineq:
            return (np.dot(self.A, x) <= self.b + self._tol).all()
        else:
            return (np.dot(self.A, x) == self.b + self._tol).all()

    def active_constraints(self, x: np.ndarray) -> np.ndarray[bool]:
        """
        Returns the indexes of the active constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - active (np.ndarray): The indexes of the active constraints.
        """

        if self.ineq:
            return np.dot(self.A, x) <= self.b + self._tol
        else:
            return np.dot(self.A, x) == self.b + self._tol

    def check_position(self, x: np.ndarray) -> str:
        """
        Checks the position of x in the feasible region.

        Parameters:
        - x (np.ndarray): The point at which to check the position.

        Returns:
        - message (str): A message indicating whether the solution is inside, on the edge or
         outside the feasible region.
        """

        if not self.evaluate(x):
            return 'outside'

        if np.any(x == 1):
            return 'edge'
        else:
            return 'inside'


def create_A(n: int, Is: list[list]) -> np.ndarray:
    """
    Create a matrix A with the specified dimension and index sets.

    Parameters:
    - n (int): The number of columns in the matrix A.
    - Is (list): The list of index sets

    Returns:
    - A (numpy.ndarray): The created matrix A.

    """

    A1 = []
    for I in Is:
        row_k = np.zeros(n)
        for index in I:
            row_k[index] = 1
        A1.append(row_k)
    A1 = np.vstack(A1)
    I = np.eye(n)

    A = np.vstack([A1, -A1, -I])
    A[A == -0.0] = 0.0
    return A


def create_b(n, K) -> np.ndarray:
    """
    Creates and returns a numpy array representing the vector b.

    Args:
    n (int): The number of dimensions.
    K (int): The number of index sets.

    Returns:
    np.array: The vector b.
    """

    ones = np.ones(K)
    zeros = np.zeros(n)

    return np.concatenate((ones, -ones, zeros))
