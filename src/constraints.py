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
        self.A = A.copy()
        self._subA = A
        self.b = b.copy()
        self.ineq = ineq
        self.num_constraints = len(b)
        self._tol = 1e-8

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
            return (np.dot(self._subA, x) <= self.b + self._tol).all()
        else:
            return (np.dot(self._subA, x) == self.b + self._tol).all()

    def active_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the indexes of the active constraints at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to evaluate the constraints.

        Returns:
        - active (np.ndarray): The indexes of the active constraints.

        """
        if self.ineq:
            return np.dot(self._subA, x) <= self.b + self._tol
        else:
            return np.dot(self._subA, x) == self.b + self._tol

    def check_KKT(self, x: np.ndarray, derivative: callable(np.ndarray)) -> str:
        """
        Checks the KKT conditions at a given point x.

        Parameters:
        - x (np.ndarray): The point at which to check the KKT conditions.
        - derivative (callable): The derivative of the objective function.

        Returns:
        - message (str): A message indicating whether the solution is inside or on the edge of the feasible region.

        """
        grad = derivative(x)

        # Identify the active constraints
        active = self.active_constraints(x)
        num_active = sum(active)

        # Calculate the lagrangian multipliers
        mu = np.zeros(self.num_constraints)
        if num_active > 0:
            A_active = self._subA[active, :]
            mu[active] = np.linalg.lstsq(A_active.T, grad, rcond=None)[0]

        if num_active > 0:
            stationarity = np.allclose(grad, np.dot(self._subA.T, mu), atol=self._tol)
        else:
            stationarity = False

        if self.ineq:
            first_condition = np.all(np.dot(self._subA, x) <= self.b + self._tol)
        else:
            first_condition = np.all(np.dot(self._subA, x) == self.b + self._tol)

        # In our case, the second condition depends on the first constraint since the second constraint
        # is the negative of the first due to the decomposition of the equality constraint into two inequalities.
        if num_active == 2:
            mu[mu < 0] = mu[mu > 0]
            second_condition = np.all(mu >= -self._tol)
        elif num_active == 1:
            if active[0]:
                second_condition = mu[0] >= -self._tol
            else:
                second_condition = mu[0] <= self._tol
                if mu[0] < 0:
                    mu[0] = -mu[0]
        else:
            second_condition = True

        if first_condition and second_condition and stationarity:
            if (mu[active] > 0).all():
                return 'inside'
            else:
                return 'on the edge'
        elif not first_condition:
            return 'outside'
        elif not second_condition:
            return 'non optimal'
        else:
            if (mu[active] > 0).all():
                return 'not stationary (inside)'
            else:
                return 'not stationary (on the edge)'

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
