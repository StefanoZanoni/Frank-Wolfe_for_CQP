import numpy as np
import argparse
import time
import os
import json
import multiprocessing
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

seed = 5


def append_to_json_file(data, filename):
    with open(filename, 'r') as f:
        file_data = json.load(f)

    file_data.update(data)

    with open(filename, 'w') as f:
        json.dump(file_data, f, indent=2)


def random_test():
    test_dimensions = np.arange(10, 1010, 10)
    for n in test_dimensions:
        if not os.path.exists(f'tests/dimension_{n}'):
            os.makedirs(f'tests/dimension_{n}')

    iterations_number = 0
    iteration_limit_number = 0
    b = create_b()
    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False)
        As = [create_A(n, I) for I in Is]

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]

        random_eccentricity = round(np.random.uniform(0.1, 1), 4)
        random_rank = np.random.randint(1, n + 1)
        random_active = round(np.random.uniform(0.1, 1), 1)
        problem = QP(n, rank=random_rank, eccentricity=random_eccentricity, active=random_active, c=False)
        _, execution_time, iterations, all_gaps = solve(problem, constraints, As, n, verbose=0)
        iterations_number += len(iterations)
        iteration_limit_number += iterations.count(1000)

        with open(f'tests/dimension_{n}/random_results.json', 'w') as f:
            json.dump({'rank': random_rank, 'eccentricity': random_eccentricity, 'active': random_active,
                       'execution_time': execution_time, 'iterations': iterations}, f, indent=2)

        for i, gaps in enumerate(all_gaps):
            plt.plot(gaps)
            plt.xlabel('Iteration')
            plt.ylabel('Gap')
            plt.savefig(f'tests/dimension_{n}/subproblem_{i}_gap.png')
            plt.close()

    max_iterations_percentage = round(iteration_limit_number / iterations_number, 3) * 100
    with open(f'tests/max_iterations.json', 'w') as f:
        json.dump({'max_iteration_percentage': max_iterations_percentage}, f)


def test_dimension_scaling():
    test_dimensions = [10, 100, 200, 500, 1000]
    eccentricity = 0.5
    active = 1.0
    b = create_b()

    with open(f'tests/dimension_scaling.json', 'w') as f:
        json.dump({'dimensions': test_dimensions, "ranks": test_dimensions,
                   "eccentricity": eccentricity, "active": active}, f, indent=2)

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False, seed=seed)
        As = [create_A(n, I) for I in Is]

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]
        problem = QP(n, rank=n, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(1000):
            _, execution_time, _, _ = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_dimension_{n}': mean, f'standard_deviation_dimension_{n}': std}
        append_to_json_file(new_data, f'tests/dimension_scaling.json')


def test_rank_scaling():
    n = 50
    test_ranks = list(np.arange(1, n + 1, 1, dtype=int))
    test_ranks = [int(rank) for rank in test_ranks]
    eccentricity = 0.5
    active = 1.0
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    b = create_b()
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/rank_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "eccentricity": eccentricity, "active": active, 'ranks': test_ranks, }, f,
                  indent=2)

    for rank in tqdm(test_ranks):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(1000):
            _, execution_time, _, _ = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_rank_{rank}': mean, f'standard_deviation_dimension_{rank}': std}
        append_to_json_file(new_data, f'tests/rank_scaling.json')


def test_eccentricity_scaling():
    n = 50
    test_eccentricities = list(np.arange(0.1, 1, 0.01))
    test_eccentricities = [float(eccentricity) for eccentricity in test_eccentricities]
    active = 1.0
    b = create_b()
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/eccentricity_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "rank": n, "active": active, 'eccentricities': test_eccentricities}, f,
                  indent=2)

    for eccentricity in tqdm(test_eccentricities):
        problem = QP(n, rank=n, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(1000):
            _, execution_time, _, _ = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_eccentricity_{eccentricity}': mean,
                    f'standard_deviation_eccentricity_{eccentricity}': std}
        append_to_json_file(new_data, f'tests/eccentricity_scaling.json')


def test_active_scaling():
    n = 50
    eccentricity = 0.5
    test_actives = list(np.arange(0.1, 1.1, 0.1))
    test_actives = [float(active) for active in test_actives]
    b = create_b()
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/active_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "rank": n, "eccentricity": eccentricity, 'actives': test_actives}, f, indent=2)

    for active in tqdm(test_actives):
        problem = QP(n, rank=n, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(1000):
            _, execution_time, _, _ = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_active_{active}': mean, f'standard_deviation_active_{active}': std}
        append_to_json_file(new_data, f'tests/active_scaling.json')


def test():
    test_functions = [test_dimension_scaling, test_rank_scaling, test_eccentricity_scaling,
                      test_active_scaling]
    processes = [multiprocessing.Process(target=test_function) for test_function in test_functions]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    print('All tests done.\n', flush=True)


if __name__ == '__main__':
    test()
