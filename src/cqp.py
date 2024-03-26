import numpy as np

from qp import QP
from src.constraints import Constraints, BoxConstraints


class CQP:
    def __init__(self, qp: QP, c: Constraints):
        self.problem = qp
        self.constraints = c


class BCQP(CQP):
    def __init__(self, qp: QP, c: BoxConstraints):
        super().__init__(qp, c)
