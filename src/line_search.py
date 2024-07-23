import numpy as np
from abc import abstractmethod

from src.qp import QP


class LineSearch:
    """
    Abstract base class for line search strategies.

    This class defines the interface for line search strategies used in optimization algorithms to find
    an appropriate step size.

    Attributes:
    - f (QP): A quadratic problem function that the line search strategy operates on.
    """

    def __init__(self, f: QP) -> None:
        """
        Initializes the LineSearch with a quadratic problem (QP) object.

        Parameters:
        - f (QP): A QP object.
        """

        self.f = f

    @abstractmethod
    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Abstract method to compute the line search.

        This method must be implemented by subclasses to define how the line search computes the step size.

        Parameters:
        - xk (np.ndarray): The current point in the search space.
        - pk (np.ndarray): The search direction.

        Returns:
        - float: The computed step size.
        """

        pass


# The ExactLineSearch class implements the exact line search strategy.
class ExactLineSearch(LineSearch):
    """
    Implements the exact line search strategy.

    This class is a concrete implementation of the LineSearch abstract base class, providing a method to
    compute the exact step size.
    """

    def __init__(self, f: QP) -> None:
        super().__init__(f)

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the exact line search step size.

        The step size is computed based on the formula: Alpha = -g^T p / p^T Q p.

        Parameters:
        - xk (np.ndarray): The current point in the search space.
        - pk (np.ndarray): The search direction.

        Returns:
        - float: The step size alpha.
        """

        den = np.linalg.multi_dot([pk.T, self.f.get_Q(), pk])
        if den <= 1e-16:
            alpha = 1
        else:
            alpha = min(np.dot(-self.f.derivative(xk).T, pk) / den, 1)

        return alpha


class BackTrackingLineSearch(LineSearch):
    """
    Implements the backtracking line search strategy.

    This class is a concrete implementation of the LineSearch abstract base class, providing a method to
    compute the step size by iteratively reducing it until a decrease in the objective function is observed.
    """

    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5) -> None:
        """
        Initializes the BackTrackingLineSearch with a quadratic problem (QP) object, an initial step size alpha,
         and a reduction factor tau.

        Parameters:
        - f (QP): A quadratic problem function.
        - alpha (float, optional): The initial step size. Default to 1.
        - tau (float, optional): The reduction factor. Default to 0.5.
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
        Computes the step size by iteratively reducing it until a decrease in the objective function is observed.

        Parameters:
        - xk (np.ndarray): The current point in the search space.
        - pk (np.ndarray): The search direction.

        Returns:
        - float: The computed step size.
        """
        while self.f.evaluate(xk + self.alpha * pk) > self.f.evaluate(xk):
            self.alpha = self.alpha * self.tau

        return self.alpha


class BackTrackingArmijoLineSearch(BackTrackingLineSearch):
    """
    Implements the backtracking line search strategy with the Armijo condition.

    This strategy ensures a sufficient decrease in the objective function, adjusted by the Armijo condition,
     before accepting a step size.

    Inherits:
    - BackTrackingLineSearch

    Attributes:
    - c1 (float): The Armijo condition constant, ensuring sufficient decrease.

    Methods:
    - compute(xk: np.ndarray, pk: np.ndarray) -> float: Computes the step size, ensuring the Armijo condition is met.
    """

    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5, c1: float = 1e-3) -> None:
        """
        Initializes the BackTrackingArmijoLineSearch with a quadratic problem (QP) function, an initial step size alpha,
         a reduction factor tau, and a c1 factor for the Armijo condition.

        Parameters:
        - f (QP): A quadratic problem function.
        - alpha (float, optional): The initial step size. Defaults to 1.
        - tau (float, optional): The reduction factor. Defaults to 0.5.
        - c1 (float, optional): The c1 factor for the Armijo condition. Default to 1e-3.
        """

        super().__init__(f, alpha, tau)
        if c1 <= 0 or c1 >= 1:
            print('The beta factor must be in the interval (0, 1). Defaulted to 1e-3.')
            c1 = 1e-3
        self.c1 = c1

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the backtracking line search with Armijo condition.
        While f(xk + alpha * pk) >= f(xk) + alpha * c1 * g^T p,
         alpha = tau * alpha.

        Parameters:
        - xk (np.ndarray): The current point in the search space.
        - pk (np.ndarray): The search direction.

        Returns:
        - float: The computed step size.
        """
        while (self.f.evaluate(xk + self.alpha * pk) >=
               self.f.evaluate(xk) + self.alpha * self.c1 * np.dot(pk.T, self.f.derivative(xk))):
            self.alpha = self.alpha * self.tau

        return self.alpha


class BackTrackingArmijoStrongWolfeLineSearch(BackTrackingArmijoLineSearch):
    """
    Implements the backtracking line search strategy with both Armijo and Strong Wolfe conditions.

    This class extends the BackTrackingArmijoLineSearch by incorporating the Strong Wolfe condition, ensuring
    that the step size not only achieves a sufficient decrease in the objective function but also maintains
    curvature conditions for more robust convergence in optimization algorithms.

    Attributes:
    - c2 (float): The Strong Wolfe condition constant, ensuring curvature conditions are met.

    Methods:
    - compute(xk: np.ndarray, pk: np.ndarray) -> float: Computes the step size,
     ensuring both Armijo and Strong Wolfe conditions are met.
    """

    def __init__(self, f: QP, alpha: float = 1, tau: float = 0.5, c1: float = 1e-3, seed: int = None) -> None:
        """
        Initializes the BackTrackingArmijoStrongWolfeLineSearch with a quadratic problem (QP) function,
         an initial step size alpha, a reduction factor tau, a c1 factor for the Armijo condition,
          and a c2 factor for the Strong Wolfe condition.

        Parameters:
        - f (QP): A quadratic problem function.
        - alpha (float, optional): The initial step size. Defaults to 1.
        - tau (float, optional): The reduction factor. Defaults to 0.5.
        - c1 (float, optional): The c1 factor for the Armijo condition. Default to 1e-3.
        - c2 (float, optional): The c2 factor for the Strong Wolfe condition. Default to 0.9.
        """

        super().__init__(f, alpha, tau, c1)
        if seed:
            np.random.seed(seed)
        self.c2 = np.random.uniform(self.c1 + 1e-6, 1)
        if self.c2 <= 0.01:
            factor = 0.1 / self.c2
            self.c2 *= factor

    def compute(self, xk: np.ndarray, pk: np.ndarray) -> float:
        """
        Compute the backtracking line search step size, ensuring both Armijo and Strong Wolfe conditions are met.

        The method iteratively reduces the step size until the Armijo condition of sufficient decrease and the
        Strong Wolfe condition of curvature are satisfied.

        Parameters:
        - xk (np.ndarray): The current point in the search space.
        - pk (np.ndarray): The search direction.

        Returns:
        - float: The computed step size.
        """

        while (self.f.evaluate(xk + self.alpha * pk) >=
               self.f.evaluate(xk) + self.alpha * self.c1 * np.dot(pk.T, self.f.derivative(xk))) and (
                np.abs(np.dot(pk.T, self.f.derivative(xk + self.alpha * pk))) >=
                self.c2 * np.abs(np.dot(pk.T, self.f.derivative(xk)))):
            self.alpha = self.alpha * self.tau

        return self.alpha
