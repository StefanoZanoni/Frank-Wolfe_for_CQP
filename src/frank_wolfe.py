from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingLineSearch, BackTrackingArmijoLineSearch, \
    BackTrackingArmijoStrongWolfeLineSearch
import numpy as np


# The frank_wolfe function implements the Frank-Wolfe algorithm for solving convex optimization problems.
def frank_wolfe(cqp: CQP, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    """
    Implement the Frank-Wolfe algorithm for solving convex optimization problems.

    :param cqp: A convex quadratic problem (CQP).
    :param x0: The initial point.
    :param eps: The tolerance for the stopping criterion (default is 1e-6).
    :param max_iter: The maximum number of iterations (default is 1000).
    :return: The optimal point and the number of iterations.
    """

    x = x0
    # best lower bound found so far
    best_lb = -np.Inf
    i = 0
    ls = BackTrackingArmijoStrongWolfeLineSearch(cqp.problem, alpha=1, tau=0.5, beta=1e-4)
    while i < max_iter:
        v = cqp.problem.evaluate(x)
        grad = cqp.problem.derivative(x)

        # solve the linear minimization oracle
        z = np.zeros_like(x)
        ind = grad < 0
        true_ind = sum(ind)
        if true_ind > 0:
            z[ind] = cqp.constraints.box_max / true_ind

        # first order model evaluation
        lb = v + np.dot(grad, z - x)

        # update the best lower bound
        if lb > best_lb:
            best_lb = lb

        gap = v - best_lb / max(np.abs(v), 1)
        if gap < eps:
            print(f'Iteration {i}: status = optimal, v = {round(v, 5)}, gap = {gap}')
            break

        # line search for alpha
        d = z - x
        alpha = ls.compute(x, d)
        x = x + alpha * d

        print(f'Iteration {i}: status = stopped, v = {round(v, 5)}, gap = {gap}')
        i += 1

    return x, i
