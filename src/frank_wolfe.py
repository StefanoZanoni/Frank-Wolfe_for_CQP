from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingLineSearch, BackTrackingArmijoLineSearch, \
    BackTrackingArmijoStrongWolfeLineSearch
import numpy as np
import sys


def solve_LMO(grad: np.ndarray) -> np.ndarray:
    z = np.zeros_like(grad)
    if sum(grad < 0) > 0:
        min_index = np.argmin(grad)
        z[min_index] = 1
    return z


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

    # starting point
    x = x0

    # best lower bound found so far
    best_lb = -np.Inf

    # line search method
    ls = BackTrackingArmijoLineSearch(cqp.problem, alpha=1, tau=0.9)

    i = 0
    while i < max_iter:
        v = cqp.problem.evaluate(x)
        grad = cqp.problem.derivative(x)

        # solve the linear minimization oracle
        z = solve_LMO(grad)

        # first order model evaluation
        d = z - x
        lb = v + np.dot(grad.T, d)

        # update the best lower bound
        if lb > best_lb:
            best_lb = lb

        gap = (v - best_lb) / max(np.abs(v), 1)
        if gap < eps:
            if gap == 0:
                print(f'Iteration {i}: status = optimal, v = {v}, gap = {gap}')
            else:
                print(f'Iteration {i}: status = approximated, v = {v}, gap = {gap}')
            break

        # line search for alpha
        alpha = ls.compute(x, d)
        x = x + alpha * d

        if i + 1 >= max_iter:
            print(f'Iteration {i}: status = stopped, v = {v}, gap = {gap}')
        else:
            print(f'Iteration {i}: status = non optimal, v = {v}, gap = {gap}')
        i += 1

    return x, i
