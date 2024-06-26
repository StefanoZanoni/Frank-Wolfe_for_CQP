import numpy as np
import time

from src.frank_wolfe import frank_wolfe
from src.cqp import BCQP
from src.qp import QP
from src.constraints import BoxConstraints


def solve(problem: QP, constraints: list[BoxConstraints], As: list[np.ndarray], n: int, max_iter: int = 1000,
          verbose: int = 1) -> tuple[np.ndarray, float, list, list[list], list[list], list[float], list[float]]:
    """
    Solve the optimization problem for each index set.

    :param problem: The optimization problem.
    :param constraints: The constraints of the problem.
    :param As: The matrices of constraints.
    :param n: The dimension of the problem.
    :param max_iter: The maximum number of iterations.
    :param verbose: The verbosity level.
    :return: The optimal solution, execution time, the number of iterations,
    all the gap histories, all the convergence rate histories, the optimal minimums and the approximated minimums.
    """

    x_optimal = np.empty(n)
    iterations = []
    all_gaps = []
    all_convergence_rates = []
    optimal_minimums = []
    approximated_minimums = []

    start = time.time()
    for i, c in enumerate(constraints):
        if verbose == 1:
            print(f'Sub problem {i}')
        # compute the indexes of k-th I index set
        indexes = As[i][0] != 0
        # initialize the starting point
        x_init = np.zeros(sum(indexes))
        x_init[0] = 1
        # construct the BCQP problem from the actual constraints
        bcqp = BCQP(problem, c)
        # consider only the subproblem relative to the indexes
        bcqp.set_subproblem(indexes)
        # solve the subproblem
        x_i, v_i, iteration, gaps, convergence_rates = frank_wolfe(bcqp, x_init, eps=1e-6, max_iter=max_iter,
                                                                   verbose=verbose)
        # merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i
        iterations.append(iteration)
        all_gaps.append(gaps)
        all_convergence_rates.append(convergence_rates)
        optimal_minimums.append(bcqp.problem.minimum())
        approximated_minimums.append(v_i)
        if verbose == 1:
            print('\n')
    end = time.time()

    return x_optimal, end - start, iterations, all_gaps, all_convergence_rates, optimal_minimums, approximated_minimums
