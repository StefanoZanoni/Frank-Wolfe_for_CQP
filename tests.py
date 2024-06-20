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
from main import solve

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


def test():
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
        active = np.random.uniform(0.1, 1)
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False)

        _, execution_time, iterations, all_gaps = solve(problem, constraints, As, n, verbose=0)
        store_results({'execution_time': execution_time, 'iterations': iterations, 'all_gaps': all_gaps,
                       'dimensions': n, 'rank': rank, 'eccentricity': eccentricity, 'active': active},
                      f'tests/dimension_{n}')


if __name__ == '__main__':
    test()
