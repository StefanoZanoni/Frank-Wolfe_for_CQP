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
    test_dimensions = np.arange(50, 151, 1)
    for n in test_dimensions:
        ensure_dir_exists(f'tests/dimension_{n}')

    iterations_number = 0
    iteration_limit_number = 0
    iterations_mean_list = []
    execution_times = []
    mean_convergence_list = []
    mean_gaps = []
    edge = 0
    inside = 0
    num_positions = 0
    one_dimension = 0
    num_I = 0
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
        _, execution_time, iterations, all_gaps, all_convergence_rates, _, _, positions = \
            solve(bcqp, verbose=0)

        for position in positions:
            if 'edge' in position:
                edge += 1
            elif 'inside' in position:
                inside += 1
        num_positions += len(positions)
        for I in Is:
            if len(I) == 1:
                one_dimension += 1
        num_I += len(Is)
        execution_times.append(execution_time)
        iterations_number += len(iterations)
        mean_iterations = round(np.mean(iterations))
        iterations_mean_list.append(mean_iterations)
        iteration_limit_number += iterations.count(1000)

        data = {'rank': random_rank, 'eccentricity': random_eccentricity, 'active': random_active,
                'execution_time': execution_time, 'iterations': iterations, 'positions': positions}
        dump_json(data, f'tests/dimension_{n}/random_results.json')

        gap_list = []
        for i, gaps in enumerate(all_gaps):
            gap_list.append(gaps[-1])
            plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                          f'tests/dimension_{n}/subproblem_{i}_gap.png')
        mean_gap = round(np.mean(gap_list), 5)
        mean_gaps.append(mean_gap)
        new_data = {'mean_gap': mean_gap}
        append_to_json_file(new_data, f'tests/dimension_{n}/random_results.json')

        convergences = []
        for i, convergence_rates in enumerate(all_convergence_rates):
            plot_and_save(convergence_rates, 'Iteration', 'Convergence rate',
                          f'tests/dimension_{n}/subproblem_{i}_convergence_rate.png')
            if len(convergence_rates) > 0:
                convergences.append(convergence_rates[-1])
        if len(convergences) > 0:
            mean_convergence_rate = round(np.mean(convergences), 5)
            mean_convergence_list.append(mean_convergence_rate)
            new_data = {'mean_convergence_rate': mean_convergence_rate}
            append_to_json_file(new_data, f'tests/dimension_{n}/random_results.json')

    mean_execution_times = round(np.mean(execution_times) * 1000, 5)
    mean_iterations = round(np.mean(iterations_mean_list))
    max_iterations_percentage = round(iteration_limit_number / iterations_number, 3) * 100
    mean_convergence_rate, std_convergence_rate = calculate_mean_std(mean_convergence_list)
    mean_gap, std_gap = calculate_mean_std(mean_gaps)
    edge_percentage = round(edge / num_positions, 3) * 100
    inside_percentage = round(inside / num_positions, 3) * 100
    one_dimension_percentage = round(one_dimension / num_I, 3) * 100
    data = {'max_iteration_percentage': max_iterations_percentage, 'mean_iterations': mean_iterations,
            'mean_execution_time (ms)': mean_execution_times,
            'mean_convergence_rate': mean_convergence_rate, 'std_convergence_rate': std_convergence_rate,
            'mean_gap': mean_gap, 'std_gap': std_gap, 'edge_percentage': edge_percentage,
            'inside_percentage': inside_percentage, 'one_dimension_percentage': one_dimension_percentage}
    dump_json(data, 'tests/statistics.json')


def test_scaling(Is: list[list[int]], constraints: BoxConstraints, n: int, rank: float, eccentricity: float,
                 active: float, filename: str, json_variable) -> None:
    problem = QP(n, Is, rank=rank, eccentricity=eccentricity, active=active, c=False, seed=seed)
    bcqp = BCQP(problem, constraints)

    execution_times = []
    gap_list = []
    convergence_list = []
    for _ in range(100):
        _, execution_time, _, all_gaps, all_convergence_rates, optimal_minimums, constrained_minimums, positions \
            = solve(bcqp, verbose=0)
        execution_times.append(execution_time)
        gap_list.append(np.mean([gaps[-1] for gaps in all_gaps]))
        convergence_list.append(np.mean([convergence_rates[-1] for convergence_rates in all_convergence_rates]))
    mean_time, std_time = calculate_mean_std(execution_times)
    mean_gap, std_gap = calculate_mean_std(gap_list)
    mean_convergence, std_convergence = calculate_mean_std(convergence_list)
    mean_time *= 1000
    std_time *= 1000

    new_data = {f'mean_execution_time_{json_variable} (ms)': mean_time,
                f'standard_deviation_{json_variable} (ms)': std_time,
                f'mean_gap_{json_variable}': mean_gap,
                f'standard_deviation_gap_{json_variable}': std_gap,
                f'mean_convergence_{json_variable}': mean_convergence,
                f'standard_deviation_convergence_{json_variable}': std_convergence,
                f'positions_{json_variable}': positions}
    append_to_json_file(new_data, filename)


def test_dimension_scaling():
    test_dimensions = [10, 100, 250, 500]
    rank = 1
    eccentricity = 0.5
    active = 1

    data = {'dimensions': test_dimensions, "rank": rank,
            "eccentricity": eccentricity, "active": active}
    dump_json(data, 'tests/dimension_scaling.json')

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=True, cardinality_K=5)
        A = create_A(n, Is)
        b = create_b(n, len(Is))

        constraints = BoxConstraints(A, b, len(Is), n, ineq=True)
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'tests/dimension_scaling.json', n)


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

    data = {"dimensions": n, "eccentricity": eccentricity, "active": active, 'ranks': test_ranks}
    dump_json(data, 'tests/rank_scaling.json')

    for rank in tqdm(test_ranks):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'tests/rank_scaling.json', rank)


def test_eccentricity_scaling():
    n = 50
    rank = 1
    test_eccentricities = list(np.arange(0, 1.1, 0.1))
    test_eccentricities = [float(eccentricity) for eccentricity in test_eccentricities]
    active = 1

    Is = create_index_sets(n, uniform=True, cardinality_K=5)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = BoxConstraints(A, b, len(Is), n, ineq=True)

    data = {"dimensions": n, "rank": rank, "active": active, 'eccentricities': test_eccentricities}
    dump_json(data, 'tests/eccentricity_scaling.json')

    for eccentricity in tqdm(test_eccentricities):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'tests/eccentricity_scaling.json', eccentricity)


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

    data = {"dimensions": n, "rank": rank, "eccentricity": eccentricity, 'actives': test_actives}
    dump_json(data, 'tests/active_scaling.json')

    for active in tqdm(test_actives):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'tests/active_scaling.json', active)


def test():
    # random_test()
    test_dimension_scaling()
    test_rank_scaling()
    test_eccentricity_scaling()
    test_active_scaling()

    print('All tests done.\n', flush=True)


if __name__ == '__main__':
    test()
