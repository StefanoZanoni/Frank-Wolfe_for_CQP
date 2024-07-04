import numpy as np
import copy


class Constraints:
    """
    Represents a set of linear constraints.

    Parameters:
    - A (np.ndarray): The coefficient matrix of the constraints.
    - b (np.ndarray): The right-hand side vector of the constraints.
    - ineq (bool, optional): Specifies whether the constraints are inequalities or equalities. Default is True.

    Attributes:
    - A (np.ndarray): The coefficient matrix of the constraints.
    - b (np.ndarray): The right-hand side vector of the constraints.
    - ineq (bool): Specifies whether the constraints are inequalities or equalities.
    - num_constraints (int): The number of constraints.

    Methods:
    - evaluate(x): Evaluates the constraints at a given point x.

    """

    def __init__(self, A: np.ndarray, b: np.ndarray, K: int, n: int, ineq: bool = True) -> None:
        self.A = A.copy()
        self._subA = A
        self.b = b.copy()
        self._subb = b
        self.ineq = ineq
        self.num_constraints = len(b)
        self._tol = 1e-8
        self.K = K
        self.dim = n
        self._sub_dim = n

    def evaluate(self, x: np.ndarray) -> bool:
        """
        Evaluates the constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - result (bool): The result of evaluating the constraints.

        """

        if self.ineq:
            return (np.dot(self._subA, x) <= self._subb + self._tol).all()
        else:
            return (np.dot(self._subA, x) == self._subb + self._tol).all()

    def active_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the indexes of the active constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - active (np.ndarray): The indexes of the active constraints.

        """
        if self.ineq:
            return np.dot(self._subA, x) <= self._subb + self._tol
        else:
            return np.dot(self._subA, x) == self._subb + self._tol

    def check_position(self, x: np.ndarray) -> str:
        """
        Checks the position of x in the feasible region.

        Parameters:
        - x (np.ndarray): The point at which to check the position.

        Returns:
        - message (str): A message indicating whether the solution is inside, on the edge or
         outside of the feasible region.

        """
        if np.sum(x) != 1 or np.any(x) < 0:
            return 'outside'

        if np.any(x == 0):
            return 'edge'
        else:
            return 'inside'

    def set_subproblem(self, k: int, dimensions: np.ndarray[bool]) -> None:
        """
            Sets the subproblem by selecting specific dimensions and the k-th index set.

            Parameters:
            - dimensions (np.ndarray): The dimensions to select.
            - k (int): The k-th index set.

            """
        self._sub_dim = sum(dimensions)

        b_zero = self.b[-self._sub_dim:]
        self._subb = np.concatenate(([1], [-1], b_zero))

        A_one = self.A[k, dimensions]
        A_minus_one = self.A[k + self.K, dimensions]
        A_zero = self.A[-self._sub_dim:, dimensions]
        self._subA = np.vstack((A_one, A_minus_one, A_zero))

    def get_dim(self):
        return self._sub_dim


class BoxConstraints(Constraints):
    """
    Represents box constraints for optimization problems.

    Attributes:
        box_min (float): The minimum value allowed for the variables.
        box_max (float): The maximum value allowed for the variables.

    Args:
        A (np.ndarray): The coefficient matrix of the linear constraints.
        b (np.ndarray): The right-hand side vector of the linear constraints.
        ineq (bool, optional): Indicates whether the constraints are inequalities. Defaults to True.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, K: int, n: int, ineq: bool = True) -> None:
        super().__init__(A, b, K, n, ineq)


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
