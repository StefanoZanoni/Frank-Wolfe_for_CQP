import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.qp import QP
from src.solver import solve

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
    mae_list = []
    iterations_mean_list = []
    execution_times = []
    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False)
        As = [create_A(n, I) for I in Is]

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]

        random_eccentricity = round(np.random.uniform(0, 1), 4)
        random_rank = np.random.uniform(0.01, 1.01)
        random_active = round(np.random.uniform(0, 1), 1)
        problem = QP(n, rank=random_rank, eccentricity=random_eccentricity, active=random_active, c=False)
        _, execution_time, iterations, all_gaps, all_convergence_rates, optimal_minimums, approximated_minimums =\
            solve(problem, constraints, As, n, verbose=0)

        execution_times.append(execution_time)
        iterations_number += len(iterations)
        mean_iterations = round(np.mean(iterations))
        iterations_mean_list.append(mean_iterations)
        iteration_limit_number += iterations.count(1000)
        mae = np.mean(np.abs(np.array(optimal_minimums) - np.array(approximated_minimums)))
        mae_list.append(mae)

        with open(f'tests/dimension_{n}/random_results.json', 'w') as f:
            json.dump({'rank': random_rank, 'eccentricity': random_eccentricity, 'active': random_active,
                       'execution_time': execution_time, 'iterations': iterations, 'mae': mae}, f, indent=2)

        for i, gaps in enumerate(all_gaps):
            for j, gap in enumerate(gaps):
                if gap != 0:
                    gaps[j] = np.log1p(gap)
            plt.plot(gaps)
            plt.xlabel('Iteration')
            plt.ylabel('Gap (log scale)')
            plt.savefig(f'tests/dimension_{n}/subproblem_{i}_gap.png')
            plt.close()

        for i, convergence_rates in enumerate(all_convergence_rates):
            plt.plot(convergence_rates)
            plt.xlabel('Iteration')
            plt.ylabel('Convergence rate')
            plt.savefig(f'tests/dimension_{n}/subproblem_{i}_convergence_rate.png')
            plt.close()

    mean_execution_times = round(np.mean(execution_times) * 1000, 5)
    mean_mae = np.mean(mae_list)
    mean_iterations = round(np.mean(iterations_mean_list))
    max_iterations_percentage = round(iteration_limit_number / iterations_number, 3) * 100
    with open(f'tests/statistics.json', 'w') as f:
        json.dump({'max_iteration_percentage': max_iterations_percentage, 'mean_mae': mean_mae,
                   'mean_iterations': mean_iterations, 'mean_execution_time (ms)': mean_execution_times}, f)


def test_dimension_scaling():
    test_dimensions = [10, 100, 200, 500, 1000]
    rank = 1
    eccentricity = 0.99
    active = 1.0
    b = create_b()

    with open(f'tests/dimension_scaling.json', 'w') as f:
        json.dump({'dimensions': test_dimensions, "rank": rank,
                   "eccentricity": eccentricity, "active": active}, f, indent=2)

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False, seed=seed)
        As = [create_A(n, I) for I in Is]

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums\
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_dimension_{n} (ms)': mean, f'standard_deviation_dimension_{n} (ms)': std}
        append_to_json_file(new_data, f'tests/dimension_scaling.json')


def test_rank_scaling():
    n = 50
    test_ranks = list(np.arange(0.1, 1.1, 0.1, dtype=float))
    test_ranks = [float(rank) for rank in test_ranks]
    eccentricity = 0.99
    active = 1.0
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    b = create_b()
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/rank_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "eccentricity": eccentricity, "active": active, 'ranks': test_ranks}, f,
                  indent=2)

    for rank in tqdm(test_ranks):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums\
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_rank_{rank} (ms)': mean, f'standard_deviation_dimension_{rank} (ms)': std}
        append_to_json_file(new_data, f'tests/rank_scaling.json')


def test_eccentricity_scaling():
    n = 50
    rank = 1
    test_eccentricities = list(np.arange(0, 1, 0.01))
    test_eccentricities = [float(eccentricity) for eccentricity in test_eccentricities]
    active = 1.0
    b = create_b()
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/eccentricity_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "rank": rank, "active": active, 'eccentricities': test_eccentricities}, f,
                  indent=2)

    for eccentricity in tqdm(test_eccentricities):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums\
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_eccentricity_{eccentricity} (ms)': mean,
                    f'standard_deviation_eccentricity_{eccentricity} (ms)': std}
        append_to_json_file(new_data, f'tests/eccentricity_scaling.json')


def test_active_scaling():
    n = 50
    rank = 1
    eccentricity = 0.99
    test_actives = list(np.arange(0, 1.1, 0.1))
    test_actives = [float(active) for active in test_actives]
    b = create_b()
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    with open(f'tests/active_scaling.json', 'w') as f:
        json.dump({"dimensions": n, "rank": rank, "eccentricity": eccentricity, 'actives': test_actives}, f, indent=2)

    for active in tqdm(test_actives):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums\
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean = round(np.mean(execution_times) * 1000, 5)
        std = round(np.std(execution_times) * 1000, 5)

        new_data = {f'mean_execution_time_active_{active} (ms)': mean, f'standard_deviation_active_{active} (ms)': std}
        append_to_json_file(new_data, f'tests/active_scaling.json')


def test():
    random_test()
    test_dimension_scaling()
    test_rank_scaling()
    test_eccentricity_scaling()
    test_active_scaling()

    print('All tests done.\n', flush=True)


if __name__ == '__main__':
    test()
