import numpy as np
import argparse
import time
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.cqp import BCQP
from src.frank_wolfe import frank_wolfe
from src.qp import QP


def solve(problem: QP, constraints: list[BoxConstraints], As: list[np.ndarray], n: int, verbose: int = 1) -> \
        tuple[np.ndarray, float, list, list[list]]:
    """
    Solve the optimization problem for each index set.

    :param problem: The optimization problem.
    :param constraints: The constraints of the problem.
    :param As: The matrices of constraints.
    :param n: The dimension of the problem.
    :param verbose: The verbosity level.
    :return: The optimal solution, execution time, and the number of iterations.
    """
    x_optimal = np.empty(n)
    iterations = []
    all_gaps = []

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
        x_i, iteration, gaps = frank_wolfe(bcqp, x_init, eps=1e-6, max_iter=1000, verbose=verbose)
        # merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i
        iterations.append(iteration)
        all_gaps.append(gaps)
        if verbose == 1:
            print('\n')
    end = time.time()

    return x_optimal, end - start, iterations, all_gaps


def main():
    parser = argparse.ArgumentParser(description='This is a program to solve optimization problems using the '
                                                 'Frank-Wolfe algorithm.')
    parser.add_argument('--dimensions', '-n',
                        type=int,
                        default=10,
                        help='The dimension of the problem. Default is 10.')
    parser.add_argument('--rank', '-r',
                        type=int,
                        default=10,
                        help='The rank of the matrix Q. Default is 10.')
    parser.add_argument('--eccentricity', '-e',
                        type=float,
                        default=0.9,
                        help='The eccentricity of the matrix Q. Default is 0.9.')
    parser.add_argument('--active', '-a',
                        type=float,
                        default=1.0,
                        help='The active constraints percentage of the problem. Default is 1.0.')
    parser.add_argument('--verbose', '-v',
                        type=int,
                        default=1,
                        help='The verbosity level. Default is 1.')
    args = parser.parse_args()

    n = args.dimensions
    rank = args.rank
    eccentricity = args.eccentricity
    active = args.active
    verbose = args.verbose
    if verbose != 0:
        verbose = 1

    Is = create_index_sets(n, uniform=False)
    As = [create_A(n, I) for I in Is]
    b = create_b()

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False)

    solution, execution_time, iterations, _ = solve(problem, constraints, As, n, verbose=verbose)

    print(f"Execution Time: {round(execution_time * 1000, 4)} ms")
    print(f"Iterations for each sub-problem: {iterations}")
    print(f'Optimal solution: {solution}')


if __name__ == '__main__':
    main()
