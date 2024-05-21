import numpy as np


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

    def __init__(self, A: np.ndarray, b: np.ndarray, ineq: bool = True) -> None:
        self.A = A
        self._subA = A
        self.b = b
        self.ineq = ineq
        self.num_constraints = len(b)

    def evaluate(self, x: np.ndarray) -> np.ndarray or bool:
        """
        Evaluates the constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - result (np.ndarray or bool): The result of evaluating the constraints. If the constraints are inequalities,
          the result is a boolean array indicating whether each constraint is satisfied. If the constraints are equalities,
          the result is a boolean indicating whether all constraints are satisfied.

        """
        if self.ineq:
            return (np.dot(self._subA, x) <= self.b + 1e-6).all()
        else:
            return (np.dot(self._subA, x) == self.b + 1e-6).all()

    def set_subproblem(self, indexes: np.ndarray) -> None:
        """
            Sets the subproblem by selecting specific indexes.

            Parameters:
            - indexes (np.ndarray): The indexes to select.

            """
        self._subA = self.A[:, indexes]


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

    def __init__(self, A: np.ndarray, b: np.ndarray, ineq: bool = True) -> None:
        self.box_min = np.min(b)
        self.box_max = np.max(b)

        super().__init__(A, b, ineq)


def create_A(n: int, index_set: list) -> np.ndarray:
    """
    Create a matrix A with specified dimensions and index set.

    Parameters:
    - n (int): The number of columns in the matrix A.
    - index_set (list): The list of indices to set to 1 in the first row of A.

    Returns:
    - A (numpy.ndarray): The created matrix A.

    """
    A = [[0 for _ in range(n)], [0 for _ in range(n)]]
    for i in index_set:
        A[0][i] = 1
        A[1][i] = -1

    return np.array(A)


def create_b() -> np.ndarray:
    """
    Creates and returns a numpy array representing the vector b.

    Returns:
    np.array: The vector b.
    """
    return np.array([1, -1])
