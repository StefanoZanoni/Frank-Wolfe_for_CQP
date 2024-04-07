from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingLineSearch, BackTrackingArmijoLineSearch, BackTrackingArmijoStrongWolfeLineSearch


import numpy as np


def frank_wolfe(cqp: CQP, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    x = x0
    best_lb = -np.Inf
    i = 0
    ls = BackTrackingArmijoStrongWolfeLineSearch(cqp.problem, alpha=1, tau=0.5, beta=1e-4)
    while i < max_iter:
        print(f'Iteration {i}')
        v = cqp.problem.evaluate(x)
        grad = cqp.problem.derivative(x)

        print(f'x: {x}')
        print(f'v: {v}')
        print(f'grad: {grad}')

        # solve the linear subproblem
        z = np.zeros_like(x)
        ind = grad < 0
        true_ind = sum(ind)
        if true_ind > 0:
            z[ind] = cqp.constraints.box_max / true_ind
        print(f'z: {z}')

        lb = v + np.dot(grad, z - x)
        if lb > best_lb:
            best_lb = lb
        print(f'best_lb: {best_lb}')
        gap = v - best_lb / max(np.abs(v), 1)
        print(f'gap: {gap}')

        if gap < eps:
            print('\n')
            break

        # line search
        d = z - x
        alpha = ls.compute(x, d)

        x = x + alpha * d
        print(f'alpha: {alpha}')
        print(f'x: {x}\n')

        i += 1

    return x, i
