import numpy as np
from abc import abstractmethod

from src.qp import QP


class LineSearch:
    def __init__(self, f: QP):
        self.f = f

    @abstractmethod
    def compute(self, xk, pk):
        pass


class ExactLineSearch(LineSearch):
    def __init__(self, f: QP):
        super().__init__(f)

    def compute(self, xk, pk):
        den = np.dot(np.dot(pk, self.f.subQ), pk)
        if den <= 1e-16:
            alpha = 1
        else:
            alpha = min(np.dot(-self.f.derivative(xk), pk) / den, 1)

        return alpha


class BackTrackingLineSearch(LineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.1):
        super().__init__(f)
        self.alpha = alpha
        self.tau = tau

    def compute(self, xk, pk):
        while self.f.evaluate(xk + self.alpha * pk) >= self.f.evaluate(xk):
            self.alpha = self.alpha * self.tau

        return self.alpha


class BackTrackingArmijoLineSearch(BackTrackingLineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.1, beta: float = 1e-4):
        super().__init__(f, alpha, tau)
        self.beta = beta

    def compute(self, xk, pk):
        while (self.f.evaluate(xk + self.alpha * pk) >=
               self.f.evaluate(xk) + self.alpha * self.beta * np.dot(self.f.derivative(xk), pk)):
            self.alpha = self.alpha * self.tau

        return self.alpha


class BackTrackingArmijoStrongWolfeLineSearch(BackTrackingArmijoLineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.1, beta: float = 1e-4, seed: int = None):
        super().__init__(f, alpha, tau, beta)
        if seed:
            np.random.seed(seed)
        self.sigma = np.random.uniform(self.beta, 1)

    def compute(self, xk, pk):
        while (self.f.evaluate(xk + self.alpha * pk) >=
               self.f.evaluate(xk) + self.beta * self.alpha * np.dot(self.f.derivative(xk), pk)) and (
                np.abs(np.dot(self.f.derivative(xk * self.alpha * pk), pk)) >=
                self.sigma * np.abs(np.dot(self.f.derivative(xk), pk))):
            self.alpha = self.alpha * self.tau

        return self.alpha
