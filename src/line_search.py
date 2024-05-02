import numpy as np
from abc import abstractmethod

from src.qp import QP


# The LineSearch class is an abstract base class that defines the interface for a line search strategy.
class LineSearch:
    def __init__(self, f: QP):
        """
        Initialize the LineSearch with a quadratic problem (QP) function.

        :param f: A QP function.
        """
        self.f = f

    @abstractmethod
    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Abstract method to compute the line search.
        Must be implemented by subclasses.

        :param xk: The current point.
        :param pk: The search direction.
        """
        pass


# The ExactLineSearch class implements the exact line search strategy.
class ExactLineSearch(LineSearch):
    def __init__(self, f: QP):
        """
        Initialize the ExactLineSearch with a QP function.

        :param f: A QP function.
        """
        super().__init__(f)

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the exact line search.
        Alpha = -g^T p / p^T Q p.

        :param xk: The current point.
        :param pk: The search direction.
        :return: The step size alpha.
        """
        den = np.dot(np.dot(pk, self.f.subQ), pk)
        if den <= 1e-16:
            alpha = 1
        else:
            alpha = min(np.dot(-self.f.derivative(xk), pk) / den, 1)

        return alpha


# The BackTrackingLineSearch class implements the backtracking line search strategy.
class BackTrackingLineSearch(LineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5):
        """
        Initialize the BackTrackingLineSearch with a QP function,
         an initial step size alpha, and a reduction factor tau.

        :param f: A QP function.
        :param alpha: The initial step size (default is 1).
        :param tau: The reduction factor (default is 0.5).
        """
        super().__init__(f)
        if alpha <= 0 or alpha > 1:
            print('The initial step size must be in the interval (0, 1). Defaulted to 1.')
            alpha = 1
        self.alpha = alpha
        if tau <= 0 or tau >= 1:
            print('The reduction factor must be in the interval (0, 1). Defaulted to 0.5.')
            tau = 0.5
        self.tau = tau

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the backtracking line search.
        While f(xk + alpha * pk) >= f(xk),
         alpha = tau * alpha.

        :param xk: The current point.
        :param pk: The search direction.
        :return: The step size alpha.
        """
        while self.f.evaluate(xk + self.alpha * pk) > self.f.evaluate(xk):
            self.alpha = self.alpha * self.tau

        return self.alpha


# The BackTrackingArmijoLineSearch class implements the backtracking line search strategy with Armijo condition.
class BackTrackingArmijoLineSearch(BackTrackingLineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5, c1: float = 1e-3):
        """
        Initialize the BackTrackingArmijoLineSearch with a QP function, an initial step size alpha,
        a reduction factor tau, and a beta factor for the Armijo condition.

        :param f: A QP function.
        :param alpha: The initial step size (default is 1).
        :param tau: The reduction factor (default is 0.5).
        :param beta: The beta factor for the Armijo condition (default is 1e-3).
        """
        super().__init__(f, alpha, tau)
        if c1 <= 0 or c1 >= 1:
            print('The beta factor must be in the interval (0, 1). Defaulted to 1e-3.')
            c1 = 1e-3
        self.c1 = c1

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the backtracking line search with Armijo condition.
        While f(xk + alpha * pk) >= f(xk) + alpha * beta * g^T p,
         alpha = tau * alpha.

        :param xk: The current point.
        :param pk: The search direction.
        :return: The step size alpha.
        """
        while (self.f.evaluate(xk + self.alpha * pk) >
               self.f.evaluate(xk) + self.alpha * self.c1 * np.dot(pk, self.f.derivative(xk))):
            self.alpha = self.alpha * self.tau

        return self.alpha


# The BackTrackingArmijoStrongWolfeLineSearch class implements the backtracking line search strategy
# with Armijo and Strong Wolfe conditions.
class BackTrackingArmijoStrongWolfeLineSearch(BackTrackingArmijoLineSearch):
    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5, c1: float = 1e-3, seed: int = None):
        """
        Initialize the BackTrackingArmijoStrongWolfeLineSearch with a QP function, an initial step size alpha,
         a reduction factor tau, a beta factor for the Armijo condition,
          and a seed for the random number generator.

        :param f: A QP function.
        :param alpha: The initial step size (default is 1).
        :param tau: The reduction factor (default is 0.5).
        :param beta: The beta factor for the Armijo condition (default is 1e-3).
        :param seed: The seed for the random number generator (default is None).
        """
        super().__init__(f, alpha, tau, c1)
        if seed:
            np.random.seed(seed)
        self.c2 = np.random.uniform(self.c1, 1)
        if self.c2 <= 0.01:
            factor = 0.1 / self.c2
            self.c2 = self.c2 * factor

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the backtracking line search with Armijo and Strong Wolfe conditions.
        While f(xk + alpha * pk) >= f(xk) + alpha * beta * g^T p and |g(xk + alpha * pk)^T p| >= sigma |g(xk)^T p|,
            alpha = tau * alpha.

        :param xk: The current point.
        :param pk: The search direction.
        :return: The step size alpha.
        """
        while (self.f.evaluate(xk + self.alpha * pk) >
               self.f.evaluate(xk) + self.alpha * self.c1 * np.dot(pk, self.f.derivative(xk))) and (
                np.abs(np.dot(pk, self.f.derivative(xk * self.alpha * pk))) >
                self.c2 * np.abs(np.dot(pk, self.f.derivative(xk)))):
            self.alpha = self.alpha * self.tau

        return self.alpha
