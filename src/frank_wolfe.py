from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingLineSearch, BackTrackingArmijoLineSearch, \
    BackTrackingArmijoStrongWolfeLineSearch
import numpy as np
import sys


def solve_LMO(grad: np.ndarray) -> np.ndarray:
    z = np.zeros_like(grad)
    min_index = np.argmin(grad)
    z[min_index] = 1
    return z


def frank_wolfe(cqp: CQP, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000, verbose: int = 1)\
        -> tuple[np.ndarray, float, int, list[float], list[float]]:
    """
    Implement the Frank-Wolfe algorithm for solving convex optimization problems.

    :param cqp: A convex quadratic problem (CQP).
    :param x0: The initial point.
    :param eps: The tolerance for the stopping criterion (default is 1e-6).
    :param max_iter: The maximum number of iterations (default is 1000).
    :param verbose: The verbosity level (default is 1).
    :return: The approximated point, the approximated value, the number of iterations, the gap history
     and the convergence rate history.
    """

    # starting point
    x = x0

    # best lower bound found so far
    best_lb = -np.Inf

    # line search method
    ls = ExactLineSearch(cqp.problem)

    # gap history
    gaps = []

    # convergence rate history
    convergence_rates = []

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
        gaps.append(gap)
        if gap < eps:
            if verbose == 1:
                print(f'Iteration {i}: status = approximated, v = {v}, gap = {gap}')
            break

        delta_k = v - cqp.problem.minimum()

        # line search for alpha
        alpha = ls.compute(x, d)
        x = x + alpha * d

        delta_k_plus_1 = cqp.problem.evaluate(x) - cqp.problem.minimum()
        convergence_rate = delta_k_plus_1 / delta_k
        convergence_rates.append(convergence_rate)

        if verbose == 1:
            if i + 1 >= max_iter:
                print(f'Iteration {i}: status = stopped, v = {v}, gap = {gap}')
            else:
                print(f'Iteration {i}: status = non optimal, v = {v}, gap = {gap}')
        i += 1

    return x, v, i, gaps, convergence_rates
