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

    def set_subproblem(self, indexes: np.ndarray) -> None:
        """
        Sets the subproblem by selecting specific indexes.

        Parameters:
        - indexes (np.ndarray): The indexes to select.

        """
        self.problem.set_subproblem(indexes)
        self.constraints.set_subproblem(indexes)


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
