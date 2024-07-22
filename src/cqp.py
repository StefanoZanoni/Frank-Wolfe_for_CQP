import numpy as np
from src.qp import QP
from src.constraints import Constraints, BoxConstraints


class CQP:
    """
    Represents a Constrained Quadratic Problem (CQP).

    Attributes:
        problem (QP): The quadratic problem to be solved.
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
        - dimensions (np.ndarray[bool]): The dimensions to be selected.

        Returns:
        - None
        """

        self.problem.set_subproblem(dimensions)
        self.constraints.set_subproblem(k, dimensions)


class BCQP(CQP):
    """
    Represents a Box-Constrained Quadratic Problem (BCQP).

    Inherits from the CQP class and adds box constraints.

    Parameters:
    - qp (QP): The Quadratic Problem object.
    - c (BoxConstraints): The box constraints object.
    """

    def __init__(self, qp: QP, c: BoxConstraints) -> None:
        super().__init__(qp, c)
