from src.cqp import CQP
from src.line_search import ExactLineSearch
import numpy as np


def solve_LMO(grad: np.ndarray) -> np.ndarray:
    z = np.zeros_like(grad)
    min_index = np.argmin(grad)
    z[min_index] = 1
    return z


def frank_wolfe(cqp: CQP, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000, verbose: int = 1) \
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
    x = x0.copy()
    # best lower bound found so far
    best_lb = -np.Inf
    # line search method
    ls = ExactLineSearch(cqp.problem)
    # gap history
    gaps = [np.inf] * max_iter
    # convergence rate history
    convergence_rates = [1] * max_iter

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
        gaps[i] = gap
        if gap < eps:
            delta_k = v - best_lb
            convergence_rates[i] = delta_k / delta_k_minus_1 if i >= 1 else 0
            if verbose == 1:
                if gap == 0:
                    print(f'Iteration {i}: status = optimal, v = {v}, gap = {gap}')
                else:
                    print(f'Iteration {i}: status = approximated, v = {v}, gap = {gap}')
            break

        # line search for alpha
        alpha = ls.compute(x, d)
        x += alpha * d

        # compute the convergence rate
        delta_k = v - best_lb
        convergence_rates[i] = delta_k / delta_k_minus_1 if i >= 1 else 0
        delta_k_minus_1 = delta_k

        i += 1
        if verbose == 1:
            if i >= max_iter:
                print(f'Iteration {i}: status = stopped, v = {v}, gap = {gap}')
            else:
                print(f'Iteration {i}: status = non optimal, v = {v}, gap = {gap}')

    if i == 0:
        return x, v, i, [gaps[0]], [convergence_rates[0]]
    else:
        return x, v, i, gaps[:i + 1], convergence_rates[:i + 1]
