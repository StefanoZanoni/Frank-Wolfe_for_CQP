from src.cqp import CQP

import numpy as np


def frank_wolfe(cqp: CQP, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000):
    x = x0
    best_lb = -np.Inf
    i = 0
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
        z[ind] = cqp.constraints.box_max

        lb = v + grad * (z - x)
        if np.linalg.norm(lb) > best_lb:
            best_lb = np.linalg.norm(lb)
        gap = (v - best_lb) / max(np.abs(v), 1)

        if np.linalg.norm(gap) < eps:
            break

        # line search
        d = z - x
        print(f'd: {d}')
        den = d.T * cqp.problem.subQ * d
        if np.linalg.norm(den) <= 1e-16:
            alpha = 1
        else:
            alpha = min(np.linalg.norm(-grad.T * d) / np.linalg.norm(den), 1)

        x = x + alpha * d

        i += 1

        

    return x, i
