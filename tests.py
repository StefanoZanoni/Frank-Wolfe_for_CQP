import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.qp import QP
from src.cqp import BCQP
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
    return np.mean(data), np.std(data)


"""def plot_and_save(data, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.semilogy(ylabel='log' in ylabel.lower())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()"""

def plot_and_save(data, xlabel, ylabel, filename, x_data=None, exclude_first=False):
    plt.figure(figsize=(10, 6))
    if x_data is None:
        x_data = np.arange(len(data))
    if exclude_first:
        x_data = x_data[1:]
        data = data[1:]
    plt.plot(x_data, data)
    if 'log' in ylabel.lower():
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

ensure_dir_exists('tests')
ensure_dir_exists('tests/random_tests')
ensure_dir_exists('tests/dimension_scaling')
ensure_dir_exists('tests/rank_scaling')
ensure_dir_exists('tests/eccentricity_scaling')
ensure_dir_exists('tests/active_scaling')

seed = 5
max_iter = 2000


def calculate_position_percentages(positions_list):
    edge_count = positions_list.count('edge')
    inside_count = positions_list.count('inside')
    total_count = len(positions_list)

    if total_count == 0:
        return 0, 0

    edge_percentage = (edge_count / total_count) * 100
    inside_percentage = (inside_count / total_count) * 100

    return edge_percentage, inside_percentage


def append_to_json_file(data, filename):
    with open(filename, 'r') as f:
        file_data = json.load(f)

    file_data.update(data)

    with open(filename, 'w') as f:
        json.dump(file_data, f, indent=2)


def random_test(on_edge: bool = True):
    test_dimensions = np.arange(50, 151, 1)
    if on_edge:
        for n in test_dimensions:
            ensure_dir_exists(f'tests/random_tests/dimension_{n}_edge')
    else:
        for n in test_dimensions:
            ensure_dir_exists(f'tests/random_tests/dimension_{n}_inside')

    execution_times = []
    iterations_list = []
    gap_list = []
    convergence_list = []
    positions_list = []
    one_dimension_count = 0
    total_index_sets = 0
    max_iterations = 0
    total_iterations = 0
    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False)
        A = create_A(n, Is)
        b = create_b(n, len(Is))

        constraints = BoxConstraints(A, b, len(Is), n, ineq=True)

        random_eccentricity = round(np.random.uniform(0, 1), 4)
        random_rank = np.random.uniform(0.01, 1.01)
        random_active = round(np.random.uniform(0, 1), 1)
        problem = QP(n, Is, rank=random_rank, eccentricity=random_eccentricity, active=random_active, c=False)
        bcqp = BCQP(problem, constraints)
        _, execution_time, iterations, all_gaps, all_convergence_rates, positions = \
            solve(bcqp, verbose=0, max_iter=max_iter)

        execution_times.append(execution_time)
        iterations_list.extend(iterations)
        gap_list.extend([gaps[-1] for gaps in all_gaps])
        convergence_list.extend([convergence_rates[-1] for convergence_rates in all_convergence_rates])
        positions_list.extend(positions)
        one_dimension_count += sum(len(I) == 1 for I in Is)
        total_index_sets += len(Is)
        max_iterations += iterations.count(max_iter)
        total_iterations += len(iterations)

        data = {'rank': random_rank,
                'eccentricity': random_eccentricity,
                'active': random_active,
                'execution_time': execution_time,
                'iterations': iterations,
                'positions': positions,
                'gaps': [gaps[-1] for gaps in all_gaps],
                'convergence_rates': [convergence_rates[-1] for convergence_rates in all_convergence_rates]}
        if on_edge:
            dump_json(data, f'tests/random_tests/dimension_{n}_edge/random_results_edge.json')
        else:
            dump_json(data, f'tests/random_tests/dimension_{n}_inside/random_results_inside.json')

        if on_edge:
            for i, gaps in enumerate(all_gaps):
                gap_list.append(gaps[-1])
                plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                              f'tests/random_tests/dimension_{n}_edge/subproblem_{i}_gap_edge.png')
        else:
            for i, gaps in enumerate(all_gaps):
                gap_list.append(gaps[-1])
                plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                              f'tests/random_tests/dimension_{n}_inside/subproblem_{i}_gap_inside.png')

        if on_edge:
            for i, convergence_rates in enumerate(all_convergence_rates):
                plot_and_save(convergence_rates, 'Iteration', 'Convergence rate (log scale)',
                              f'tests/random_tests/dimension_{n}_edge/subproblem_{i}_convergence_rate_edge.png',exclude_first=True)
        else:
            for i, convergence_rates in enumerate(all_convergence_rates):
                plot_and_save(convergence_rates, 'Iteration', 'Convergence rate (log scale)',
                              f'tests/random_tests/dimension_{n}_inside/subproblem_{i}_convergence_rate_inside.png', exclude_first=True)

    mean_time, std_time = calculate_mean_std(execution_times)
    mean_gap, std_gap = calculate_mean_std(gap_list)
    mean_convergence, std_convergence = calculate_mean_std(convergence_list)
    mean_iterations = np.mean(iterations_list)
    edge_percentage, inside_percentage = calculate_position_percentages(positions_list)
    one_dimension_percentage = (one_dimension_count / total_index_sets * 100) if total_index_sets else 0
    max_iterations_percentage = (max_iterations / total_iterations * 100) if total_iterations else 0
    mean_time *= 1000
    std_time *= 1000
    data = {
        f'max_iterations_percentage': max_iterations_percentage,
        'mean_execution_time (ms)': mean_time,
        'standard_deviation_time (ms)': std_time,
        'mean_gap': mean_gap,
        'standard_deviation_gap': std_gap,
        'mean_convergence_rate': mean_convergence,
        'standard_deviation_convergence_rate': std_convergence,
        'mean_iterations': round(mean_iterations),
        'edge_percentage': edge_percentage,
        'inside_percentage': inside_percentage,
        'one_dimension_percentage': one_dimension_percentage
    }
    if on_edge:
        dump_json(data, f'tests/random_tests/random_tests_statistics_edge.json')
    else:
        dump_json(data, f'tests/random_tests/random_tests_statistics_inside.json')


def test_scaling(Is: list[list[int]], constraints: BoxConstraints, n: int, rank: float, eccentricity: float,
                 active: float, test_variable: str) -> None:
    if test_variable == 'rank':
        param_variable = f'rank_{rank}'
    elif test_variable == 'eccentricity':
        param_variable = f'eccentricity_{eccentricity}'
    elif test_variable == 'active':
        param_variable = f'active_{active}'
    elif test_variable == 'dimension':
        param_variable = f'dimension_{n}'

    problem_dir = f'tests/{test_variable}_scaling/{param_variable}'
    ensure_dir_exists(problem_dir)

    execution_times = []
    iterations_list = []
    gap_list = []
    convergence_list = []
    positions_list = []
    one_dimension_count = 0
    total_index_sets = 0
    max_iterations = 0
    total_iterations = 0
    for _ in range(100):
        problem = QP(n, Is, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)
        bcqp = BCQP(problem, constraints)

        _, execution_time, iterations, all_gaps, all_convergence_rates, positions = solve(bcqp,
                                                                                          verbose=0,
                                                                                          max_iter=max_iter)

        # collect general statistics
        execution_times.append(execution_time)
        iterations_list.extend(iterations)
        gap_list.extend([gaps[-1] for gaps in all_gaps])
        convergence_list.extend([convergence_rates[-1] for convergence_rates in all_convergence_rates])
        positions_list.extend(positions)
        one_dimension_count += sum(len(I) == 1 for I in Is)
        total_index_sets += len(Is)
        max_iterations += iterations.count(max_iter)
        total_iterations += len(iterations)

    # Plotting for each subproblem
    for i, gaps in enumerate(all_gaps):
        plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                      f'{problem_dir}/subproblem_{i}_gap.png')
    for i, convergence_rates in enumerate(all_convergence_rates):
        plot_and_save(convergence_rates, 'Iteration', 'Convergence rate (log scale)',
                      f'{problem_dir}/subproblem_{i}_convergence_rate.png', exclude_first=True)

    mean_time, std_time = calculate_mean_std(execution_times)
    mean_gap, std_gap = calculate_mean_std(gap_list)
    mean_convergence, std_convergence = calculate_mean_std(convergence_list)
    mean_iterations = np.mean(iterations_list)
    edge_percentage, inside_percentage = calculate_position_percentages(positions_list)
    one_dimension_percentage = (one_dimension_count / total_index_sets * 100) if Is else 0
    max_iterations_percentage = (max_iterations / total_iterations * 100) if total_iterations else 0
    mean_time *= 1000
    std_time *= 1000

    new_data = {
        f'max_iterations_percentage_{param_variable}': max_iterations_percentage,
        f'mean_execution_time_{param_variable} (ms)': mean_time,
        f'standard_deviation_{param_variable} (ms)': std_time,
        f'mean_gap_{param_variable}': mean_gap,
        f'standard_deviation_gap_{param_variable}': std_gap,
        f'mean_convergence_{param_variable}': mean_convergence,
        f'standard_deviation_convergence_{param_variable}': std_convergence,
        f'mean_iterations_{param_variable}': round(mean_iterations),
        f'edge_percentage_{param_variable}': edge_percentage,
        f'inside_percentage_{param_variable}': inside_percentage,
        f'one_dimension_percentage_{param_variable}': one_dimension_percentage
    }
    dump_json(new_data, f'{problem_dir}/{param_variable}_statistics.json')


def test_dimension_scaling():
    test_dimensions = [10, 100, 250, 500]
    rank = 1
    eccentricity = 0.5
    active = 1

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False)
        A = create_A(n, Is)
        b = create_b(n, len(Is))

        constraints = BoxConstraints(A, b, len(Is), n, ineq=True)
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'dimension')


def test_rank_scaling():
    n = 50
    test_ranks = list(np.arange(0.1, 1.1, 0.1, dtype=float))
    test_ranks = [float(rank) for rank in test_ranks]
    eccentricity = 0.5
    active = 1

    Is = create_index_sets(n, uniform=True, cardinality_K=5)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = BoxConstraints(A, b, len(Is), n, ineq=True)

    for rank in tqdm(test_ranks):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'rank')


def test_eccentricity_scaling():
    n = 50
    rank = 1
    test_eccentricities = list(np.arange(0, 1, 0.1))
    test_eccentricities = [float(eccentricity) for eccentricity in test_eccentricities]
    active = 1

    Is = create_index_sets(n, uniform=True, cardinality_K=5)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = BoxConstraints(A, b, len(Is), n, ineq=True)

    for eccentricity in tqdm(test_eccentricities):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'eccentricity')


def test_active_scaling():
    n = 50
    rank = 1
    eccentricity = 0.5
    test_actives = list(np.arange(0, 1.1, 0.1))
    test_actives = [float(active) for active in test_actives]

    Is = create_index_sets(n, uniform=True, cardinality_K=5)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = BoxConstraints(A, b, len(Is), n, ineq=True)

    for active in tqdm(test_actives):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'active')


def test():
    random_test(on_edge=True)
    random_test(on_edge=False)
    # test_dimension_scaling()
    # test_rank_scaling()
    # test_eccentricity_scaling()
    # test_active_scaling()

    print('All tests done.\n', flush=True)


if __name__ == '__main__':
    test()
