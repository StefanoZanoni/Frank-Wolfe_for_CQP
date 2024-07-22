from src.cqp import CQP
from src.line_search import ExactLineSearch, BackTrackingArmijoStrongWolfeLineSearch
import numpy as np


def solve_LMO(grad: np.ndarray) -> np.ndarray:
    z = np.zeros_like(grad)
    min_index = np.argmin(grad)
    z[min_index] = 1
    return z


def frank_wolfe(cqp, x0: np.ndarray, eps: float = 1e-6, max_iter: int = 1000, verbose: int = 1) \
        -> tuple[np.ndarray, float, int, list[float], list[float]]:
    """
    Implements the Frank-Wolfe algorithm for solving convex quadratic optimization problems.

    Parameters:
    - cqp (CQP): An instance of a convex quadratic problem.
    - x0 (np.ndarray): The initial point for the optimization process.
    - eps (float, optional): The tolerance for the stopping criterion. Defaults to 1e-6.
    - max_iter (int, optional): The maximum number of iterations to perform. Defaults to 1000.
    - verbose (int, optional): The verbosity level of the output. If set to 1, detailed iteration data is printed.
     Default to 1.

    Returns:
    - tuple[np.ndarray, float, int, list[float], list[float]]: A tuple containing the approximated optimal point,
     the value of the objective function at this point, the number of iterations performed,
      the history of the duality gaps, and the history of convergence rates.
    """
    if len(x0) == 1:
        return x0, cqp.problem.evaluate(x0), 0, [0], [1]

    # Starting point
    x = x0.copy()
    # Best lower bound found so far
    best_lb = -np.Inf
    # Line search method
    ls = ExactLineSearch(cqp.problem)
    # Gap history
    gaps = []
    # Convergence rate history
    convergence_rates = []

    i = 0
    while i < max_iter:
        # Evaluate the objective function at the current point
        v = cqp.problem.evaluate(x)
        # Compute the gradient at the current point
        grad = cqp.problem.derivative(x)

        # Solve the linear minimization oracle
        z = solve_LMO(grad)

        # Compute the direction
        d = z - x
        # Compute the lower bound
        lb = v + np.dot(grad.T, d)

        # Update the best lower bound
        if lb > best_lb:
            best_lb = lb

        # Compute the duality gap
        gap = (v - best_lb) / max(np.abs(v), 1)
        gaps.append(gap)

        if i > 0:
            # Compute the convergence rate
            convergence_rate = gaps[-1] / gaps[-2] if gaps[-2] != 0 else convergence_rates[-1]
            convergence_rates.append(convergence_rate)
        else:
            # Append an initial convergence rate for the first iteration
            convergence_rates.append(1)

        # Check the stopping criterion
        if gap < eps:
            if verbose == 1:
                print(f'Iteration {i}: status = {"optimal" if gap == 0 else "approximated"}, v = {v}, gap = {gap}')
            break

        # Line search for step size
        alpha = ls.compute(x, d)
        x += alpha * d

        i += 1
        if verbose == 1:
            if i >= max_iter:
                print(f'Iteration {i}: status = stopped, v = {v}, gap = {gap}')
            else:
                print(f'Iteration {i}: status = non-optimal, v = {v}, gap = {gap}')

    # Handle the convergence rate at the last iteration
    if len(gaps) > 1:
        final_convergence_rate = gaps[-1] / gaps[-2] if gaps[-2] != 0 else convergence_rates[-1]
        convergence_rates.append(final_convergence_rate)
    
    if len(convergence_rates) > 1:
        convergence_rates[0] = convergence_rates[1]
    
    return x, v, i, gaps, convergence_rates
