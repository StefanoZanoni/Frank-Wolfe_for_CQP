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


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_mean_std(data):
    return round(np.mean(data) * 1000, 5), round(np.std(data) * 1000, 5)


def plot_and_save(data, xlabel, ylabel, filename):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


ensure_dir_exists('tests')

seed = 5


def append_to_json_file(data, filename):
    with open(filename, 'r') as f:
        file_data = json.load(f)

    file_data.update(data)

    with open(filename, 'w') as f:
        json.dump(file_data, f, indent=2)


def random_test():
    test_dimensions = np.arange(1, 101, 1)
    for n in test_dimensions:
        ensure_dir_exists(f'tests/dimension_{n}')

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
        _, execution_time, iterations, all_gaps, all_convergence_rates, optimal_minimums, approximated_minimums = \
            solve(problem, constraints, As, n, verbose=0)

        execution_times.append(execution_time)
        iterations_number += len(iterations)
        mean_iterations = round(np.mean(iterations))
        iterations_mean_list.append(mean_iterations)
        iteration_limit_number += iterations.count(1000)
        mae = np.mean(np.abs(np.array(optimal_minimums) - np.array(approximated_minimums)))
        mae_list.append(mae)

        data = {'rank': random_rank, 'eccentricity': random_eccentricity, 'active': random_active,
                'execution_time': execution_time, 'iterations': iterations, 'mae': mae}
        dump_json(data, f'tests/dimension_{n}/random_results.json')

        for i, gaps in enumerate(all_gaps):
            for j, gap in enumerate(gaps):
                if gap != 0:
                    gaps[j] = np.log1p(gap)
            plot_and_save(gaps, 'Iteration', 'Gap (log scale)', f'tests/dimension_{n}/subproblem_{i}_gap.png')

        for i, convergence_rates in enumerate(all_convergence_rates):
            plot_and_save(convergence_rates, 'Iteration', 'Convergence rate',
                          f'tests/dimension_{n}/subproblem_{i}_convergence_rate.png')

    mean_execution_times = round(np.mean(execution_times) * 1000, 5)
    mean_mae = np.mean(mae_list)
    mean_iterations = round(np.mean(iterations_mean_list))
    max_iterations_percentage = round(iteration_limit_number / iterations_number, 3) * 100
    data = {'max_iteration_percentage': max_iterations_percentage, 'mean_mae': mean_mae,
            'mean_iterations': mean_iterations, 'mean_execution_time (ms)': mean_execution_times}
    dump_json(data, 'tests/statistics.json')


def test_dimension_scaling():
    test_dimensions = [10, 100, 250, 500]
    rank = 1
    eccentricity = 0.5
    active = 1.0
    b = create_b()

    data = {'dimensions': test_dimensions, "rank": rank,
            "eccentricity": eccentricity, "active": active}
    dump_json(data, 'tests/dimension_scaling.json')

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False, seed=seed)
        As = [create_A(n, I) for I in Is]

        constraints = [BoxConstraints(A, b, ineq=True) for A in As]
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums \
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean, std = calculate_mean_std(execution_times)

        new_data = {f'mean_execution_time_dimension_{n} (ms)': mean, f'standard_deviation_dimension_{n} (ms)': std}
        append_to_json_file(new_data, f'tests/dimension_scaling.json')


def test_rank_scaling():
    n = 50
    test_ranks = list(np.arange(0.1, 1.1, 0.1, dtype=float))
    test_ranks = [float(rank) for rank in test_ranks]
    eccentricity = 0.5
    active = 1.0
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    b = create_b()
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    data = {"dimensions": n, "eccentricity": eccentricity, "active": active, 'ranks': test_ranks}
    dump_json(data, 'tests/rank_scaling.json')

    for rank in tqdm(test_ranks):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums \
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean, std = calculate_mean_std(execution_times)

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

    data = {"dimensions": n, "rank": rank, "active": active, 'eccentricities': test_eccentricities}
    dump_json(data, 'tests/eccentricity_scaling.json')

    for eccentricity in tqdm(test_eccentricities):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums \
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean, std = calculate_mean_std(execution_times)

        new_data = {f'mean_execution_time_eccentricity_{eccentricity} (ms)': mean,
                    f'standard_deviation_eccentricity_{eccentricity} (ms)': std}
        append_to_json_file(new_data, f'tests/eccentricity_scaling.json')


def test_active_scaling():
    n = 50
    rank = 1
    eccentricity = 0.5
    test_actives = list(np.arange(0, 1.1, 0.1))
    test_actives = [float(active) for active in test_actives]
    b = create_b()
    Is = create_index_sets(n, uniform=False, seed=seed)
    As = [create_A(n, I) for I in Is]
    constraints = [BoxConstraints(A, b, ineq=True) for A in As]

    data = {"dimensions": n, "rank": rank, "eccentricity": eccentricity, 'actives': test_actives}
    dump_json(data, 'tests/active_scaling.json')

    for active in tqdm(test_actives):
        problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)

        execution_times = []
        for _ in range(100):
            _, execution_time, _, _, _, optimal_minimums, approximated_minimums \
                = solve(problem, constraints, As, n, verbose=0)
            execution_times.append(execution_time)
        mean, std = calculate_mean_std(execution_times)

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
