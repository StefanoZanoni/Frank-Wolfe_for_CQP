import numpy as np
from src.qp import QP
from src.constraints import Constraints


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
