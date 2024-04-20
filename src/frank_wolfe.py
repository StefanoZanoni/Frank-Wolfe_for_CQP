from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingLineSearch, BackTrackingArmijoLineSearch, \
    BackTrackingArmijoStrongWolfeLineSearch
import numpy as np
import sys


def solve_LMO(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.zeros_like(x)
    negative_indexes = grad < 0
    num_negative_indexes = sum(negative_indexes)
    if num_negative_indexes > 0:
        # I have to multiply the smallest negative gradient by the largest value of z
        if num_negative_indexes == 1:
            z[negative_indexes] = 1
        else:
            sorted_indexes = np.argsort(grad[negative_indexes])

            # u = machine precision
            # z1 + z2 + ... + zn = 1
            # denominator = n + nu
            # zn = n / n + nu
            # zn-1 = (denominator - prec_numerator)(1 - u) / n + nu
            # zn-2 = (prec_numerator - denominator - numerator)(1 - u) / n + nu
            # ...
            u = sys.float_info.epsilon

            # base case
            denominator = num_negative_indexes + num_negative_indexes * u
            numerator = num_negative_indexes
            z[sorted_indexes[0]] = numerator / denominator
            difference = denominator - numerator
            # recursive case
            for index in sorted_indexes[1:]:
                numerator = difference * (1 - u)
                z[index] = numerator / denominator
                difference -= numerator
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
    ls = BackTrackingArmijoStrongWolfeLineSearch(cqp.problem, alpha=1, tau=0.9, beta=1e-4)

    i = 0
    while i < max_iter:
        v = cqp.problem.evaluate(x)
        grad = cqp.problem.derivative(x)

        # solve the linear minimization oracle
        z = solve_LMO(grad, x)

        # first order model evaluation
        lb = v + np.dot(grad, z - x)

        # update the best lower bound
        if lb > best_lb:
            best_lb = lb

        gap = (v - best_lb) / max(np.abs(v), 1)
        if gap < eps:
            print(f'Iteration {i}: status = optimal, v = {v}, gap = {gap}')
            break

        # line search for alpha
        d = z - x
        alpha = ls.compute(x, d)
        x = x + alpha * d

        if i + 1 >= max_iter:
            print(f'Iteration {i}: status = stopped, v = {v}, gap = {gap}')
        else:
            print(f'Iteration {i}: status = non optimal, v = {v}, gap = {gap}')
        i += 1

    return x, i
