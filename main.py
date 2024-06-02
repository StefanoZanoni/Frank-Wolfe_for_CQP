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

if not os.path.exists(f'tests'):
    os.makedirs(f'tests')


def store_results(results: dict, dirname: str):
    """
    Store the results in a given directory.

    :param results: The results to store.
    :param dirname: The name of the directory in which the files need to be stored.
    """
    with open(f'{dirname}/results.json', 'w') as f:
        json.dump({'execution_time': results['execution_time'], 'iterations': results['iterations']}, f)
    with open(f'{dirname}/configuration.json', 'w') as f:
        json.dump({'dimensions': int(results['dimensions']), 'rank': int(results['rank']),
                   'eccentricity': float(results['eccentricity']),
                   'active_constraints_percentage': float(results['active'])}, f)

    for i, gaps in enumerate(results['all_gaps']):
        plt.plot(gaps)
        plt.xlabel('Iteration')
        plt.ylabel('Gap')
        plt.savefig(f'{dirname}/subproblem_{i}_gap.png')
        plt.close()


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
    test_dimensions = np.arange(10, 1010, 10)
    for n in test_dimensions:
        if not os.path.exists(f'tests/dimension_{n}'):
            os.makedirs(f'tests/dimension_{n}')

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False)
        As = [create_A(n, I) for I in Is]
        b = create_b()

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]

        eccentricity = np.random.uniform(0.1, 1)
        rank = np.random.randint(1, n + 1)
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=1, c=False)

        _, execution_time, iterations, all_gaps = solve(problem, constraints, As, n, verbose=0)
        store_results({'execution_time': execution_time, 'iterations': iterations, 'all_gaps': all_gaps,
                       'dimensions': n, 'rank': rank, 'eccentricity': eccentricity, 'active': 1},
                      f'tests/dimension_{n}')


if __name__ == '__main__':
    main()
