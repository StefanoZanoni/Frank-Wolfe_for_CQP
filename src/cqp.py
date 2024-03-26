import numpy as np

from qp import QP
from src.constraints import C


class CQP:
    def __init__(self, qp: QP, c: C):
        self.problem = qp
        self.constraints = c
