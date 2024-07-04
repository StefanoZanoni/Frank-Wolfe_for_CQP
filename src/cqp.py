import numpy as np
from src.qp import QP
from src.constraints import Constraints, BoxConstraints


class CQP:
    """
    Represents a Convex Quadratic Program (CQP).

    Attributes:
        problem (QP): The quadratic program to be solved.
        constraints (Constraints): The constraints of the problem.
    """

    def __init__(self, qp: QP, c: Constraints) -> None:
        self.problem = qp
        self.constraints = c

    def set_subproblem(self, k: int, dimensions: np.ndarray[bool]) -> None:
        """
        Sets the subproblem by selecting specific dimensions and the k-th index_set.

        Parameters:
        - k (int): The k-th index set.
        - dimensions (np.ndarray): The dimensions to select.

        """
        self.problem.set_subproblem(dimensions)
        self.constraints.set_subproblem(k, dimensions)


class BCQP(CQP):
    """
    Represents a Box-Constrained Quadratic Program (BCQP).

    Inherits from the CQP class and adds box constraints.

    Parameters:
    - qp (QP): The Quadratic Program object.
    - c (BoxConstraints): The box constraints object.

    """
    def __init__(self, qp: QP, c: BoxConstraints) -> None:
        super().__init__(qp, c)
