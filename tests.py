import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.index_set import create_index_sets
from src.constraints import create_A, create_b, Constraints
from src.qp import QP
from src.cqp import CQP
from src.solver import solve


def ensure_dir_exists(directory):
    """
    Ensures that a directory exists. If the directory does not exist, it is created.

    Parameters:
    - directory (str): The path of the directory to check or create.

    Returns:
    - None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def load_json(filename):
    """
   Loads a JSON file and returns its content.

   Parameters:
   - filename (str): The path to the JSON file to be loaded.

   Returns:
   - dict: The content of the JSON file.
   """

    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(data, filename):
    """
    Dumps data into a JSON file.

    Parameters:
    - data (dict): The data to be dumped into the file.
    - filename (str): The path to the JSON file where the data will be stored.

    Returns:
    - None
    """

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_mean_std(data):
    """
    Calculates the mean and standard deviation of a list of numbers.

    Parameters:
    - data (list): The list of numbers.

    Returns:
    - tuple[float, float]: A tuple containing the mean and standard deviation of the data.
    """
    return np.mean(data), np.std(data)


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


def calculate_position_percentages(positions_list):
    """
    Calculates the percentage of positions that are 'edge' and 'inside' from a list of positions.

    Parameters:
    - positions_list (list[str]): A list of positions, each being either 'edge' or 'inside'.

    Returns:
    - tuple[float, float]: A tuple containing the percentage of 'edge' positions and 'inside' positions, respectively.
    """

    edge_count = positions_list.count('edge')
    inside_count = positions_list.count('inside')
    total_count = len(positions_list)

    if total_count == 0:
        return 0, 0

    edge_percentage = (edge_count / total_count) * 100
    inside_percentage = (inside_count / total_count) * 100

    return edge_percentage, inside_percentage


def append_to_json_file(data, filename):
    """
    Appends data to an existing JSON file. If the file does not exist, it creates a new file with the data.

    Parameters:
    - data (dict): The data to be appended.
    - filename (str): The path to the JSON file.

    Returns:
    - None
    """

    with open(filename, 'r') as f:
        file_data = json.load(f)

    file_data.update(data)

    with open(filename, 'w') as f:
        json.dump(file_data, f, indent=2)


def random_test(on_edge: bool = True, max_iter: int = 2000, seed: int = None):
    """
    Conducts a random test on the constrained quadratic problem with varying dimensions and saves the results.

    Parameters:
    - on_edge (bool, optional): Whether to initialize the starting point on the edge. Defaults to True.
    - max_iter (int, optional): The maximum number of iterations for the solver. Defaults to 2000.
    - seed (int, optional): The seed for the random number generator. If None, the RNG is not seeded. Defaults to None.

    Returns:
    - None
    """

    test_dimensions = np.arange(50, 151, 1)
    if on_edge:
        for n in test_dimensions:
            ensure_dir_exists(f'tests/random_tests/dimension_{n}_edge')
    else:
        for n in test_dimensions:
            ensure_dir_exists(f'tests/random_tests/dimension_{n}_inside')
    if seed:
        np.random.seed(seed)

    execution_times = []
    iterations_list = []
    gap_list = []
    convergence_list = []
    positions_list = []
    one_dimension_count = 0
    total_index_sets = 0
    max_iterations = 0
    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False, seed=seed)
        A = create_A(n, Is)
        b = create_b(n, len(Is))

        constraints = Constraints(A, b, len(Is), n, ineq=True)

        random_eccentricity = np.random.uniform(0, 1)
        random_rank = np.random.uniform(0.01, 1.01)
        random_active = np.random.uniform(0, 1)

        problem = QP(n, Is,
                     rank=random_rank,
                     eccentricity=random_eccentricity,
                     active=random_active,
                     c=False)
        cqp = CQP(problem, constraints)
        _, execution_time, iterations, gaps, convergence_rates, position = solve(cqp,
                                                                                 verbose=0,
                                                                                 max_iter=max_iter)

        execution_times.append(execution_time)
        iterations_list.append(iterations)
        gap_list.append(gaps[-1])
        convergence_list.append(convergence_rates[-1])
        positions_list.append(position)
        one_dimension_count += sum(len(I) == 1 for I in Is)
        total_index_sets += len(Is)
        max_iterations += 1 if iterations == max_iter else 0

        data = {'rank': random_rank,
                'eccentricity': random_eccentricity,
                'active': random_active,
                'execution_time': execution_time,
                'iterations': iterations,
                'position': position,
                'gap': gaps[-1],
                'convergence_rate': convergence_rates[-1]}
        if on_edge:
            dump_json(data, f'tests/random_tests/dimension_{n}_edge/random_results_edge.json')
            plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                          f'tests/random_tests/dimension_{n}_edge/gap_edge.png')
            plot_and_save(convergence_rates, 'Iteration', 'Convergence rate (log scale)',
                          f'tests/random_tests/dimension_{n}_edge/convergence_rate_edge.png')
        else:
            dump_json(data, f'tests/random_tests/dimension_{n}_inside/random_results_inside.json')
            plot_and_save(gaps, 'Iteration', 'Gap (log scale)',
                          f'tests/random_tests/dimension_{n}_inside/gap_inside.png')
            plot_and_save(convergence_rates, 'Iteration', 'Convergence rate (log scale)',
                          f'tests/random_tests/dimension_{n}_inside/convergence_rate_inside.png')

    mean_time, std_time = calculate_mean_std(execution_times)
    mean_gap, std_gap = calculate_mean_std(gap_list)
    mean_convergence, std_convergence = calculate_mean_std(convergence_list)
    mean_iterations = np.mean(iterations_list)
    edge_percentage, inside_percentage = calculate_position_percentages(positions_list)
    one_dimension_percentage = (one_dimension_count / total_index_sets * 100) if total_index_sets else 0
    max_iterations_percentage = (max_iterations / len(test_dimensions) * 100) if len(test_dimensions) else 0
    mean_time *= 1000
    std_time *= 1000
    data = {
        f'max_iterations_percentage (%)': max_iterations_percentage,
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

    if seed:
        np.random.seed(None)


def test_scaling(Is: list[list[int]], constraints: Constraints, n: int, rank: float, eccentricity: float,
                 active: float, test_variable: str, max_iter: int = 2000, seed: int = None) -> None:
    """
    Tests the scaling of the constrained quadratic problem with respect to a specific variable and saves the results.

    Parameters:
    - Is (list[list[int]]): The index sets for the constraints.
    - constraints (Constraints): The constraints of the CQP problem.
    - n (int): The dimension of the problem.
    - rank (float): The rank of the Q matrix.
    - eccentricity (float): The eccentricity of the Q matrix.
    - active (float): The percentage of active constraints.
    - test_variable (str): The variable with respect to which the scaling is tested ('rank', 'eccentricity', 'active',
     or 'dimension').
    - max_iter (int, optional): The maximum number of iterations for the solver. Defaults to 2000.
    - seed (int, optional): The seed for the random number generator. If None, the RNG is not seeded. Defaults to None.

    Returns:
    - None
    """

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

    if seed:
        np.random.seed(seed)

    execution_times = []
    iterations_list = []
    gap_list = []
    convergence_list = []
    positions_list = []
    one_dimension_count = 0
    total_index_sets = 0
    max_iterations = 0
    for _ in range(100):
        problem = QP(n, Is, rank=rank, eccentricity=eccentricity, active=active, c=False)
        cqp = CQP(problem, constraints)

        _, execution_time, iterations, gaps, convergence_rates, position = solve(cqp,
                                                                                 verbose=0,
                                                                                 max_iter=max_iter)

        # collect general statistics
        execution_times.append(execution_time)
        iterations_list.append(iterations)
        gap_list.append(gaps[-1])
        convergence_list.append(convergence_rates[-1])
        positions_list.append(position)
        one_dimension_count += sum(len(I) == 1 for I in Is)
        total_index_sets += len(Is)
        max_iterations += 1 if iterations == max_iter else 0

    mean_time, std_time = calculate_mean_std(execution_times)
    mean_gap, std_gap = calculate_mean_std(gap_list)
    mean_convergence, std_convergence = calculate_mean_std(convergence_list)
    mean_iterations = np.mean(iterations_list)
    edge_percentage, inside_percentage = calculate_position_percentages(positions_list)
    one_dimension_percentage = (one_dimension_count / total_index_sets * 100) if total_index_sets else 0
    max_iterations_percentage = ((max_iterations / 100) * 100)
    mean_time *= 1000
    std_time *= 1000

    new_data = {
        f'max_iterations_percentage_{param_variable} (%)': max_iterations_percentage,
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

    if seed:
        np.random.seed(None)


def test_dimension_scaling(seed: int = None, max_iter: int = 2000):
    test_dimensions = [10, 100, 250, 500, 1000, 1500, 2000, 5000]
    rank = 1
    eccentricity = 0.5
    active = 1

    for n in tqdm(test_dimensions):
        Is = create_index_sets(n, uniform=False, seed=seed)
        A = create_A(n, Is)
        b = create_b(n, len(Is))

        constraints = Constraints(A, b, len(Is), n, ineq=True)
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'dimension',
                     seed=seed,
                     max_iter=max_iter)


def test_rank_scaling(seed: int = None, max_iter: int = 2000):
    n = 50
    test_ranks = list(np.arange(0.1, 1.1, 0.1, dtype=float))
    test_ranks = [float(rank) for rank in test_ranks]
    eccentricity = 0.5
    active = 1

    Is = create_index_sets(n, uniform=True, cardinality_K=5, seed=seed)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = Constraints(A, b, len(Is), n, ineq=True)

    for rank in tqdm(test_ranks):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'rank', seed=seed, max_iter=max_iter)


def test_eccentricity_scaling(seed: int = None, max_iter: int = 2000):
    n = 50
    rank = 1
    test_eccentricities = list(np.arange(0, 1, 0.1))
    test_eccentricities = [float(eccentricity) for eccentricity in test_eccentricities]
    active = 1

    Is = create_index_sets(n, uniform=True, cardinality_K=5, seed=seed)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = Constraints(A, b, len(Is), n, ineq=True)

    for eccentricity in tqdm(test_eccentricities):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'eccentricity',
                     seed=seed,
                     max_iter=max_iter)


def test_active_scaling(seed: int = None, max_iter: int = 2000):
    n = 50
    rank = 1
    eccentricity = 0.5
    test_actives = list(np.arange(0, 1.1, 0.1))
    test_actives = [float(active) for active in test_actives]

    Is = create_index_sets(n, uniform=True, cardinality_K=5, seed=seed)
    A = create_A(n, Is)
    b = create_b(n, len(Is))
    constraints = Constraints(A, b, len(Is), n, ineq=True)

    for active in tqdm(test_actives):
        test_scaling(Is, constraints, n, rank, eccentricity, active, 'active', seed=seed, max_iter=max_iter)


def test():
    random_test(on_edge=True, seed=10011)
    random_test(on_edge=False, seed=10011)
    test_dimension_scaling()
    test_rank_scaling()
    test_eccentricity_scaling()
    test_active_scaling()

    print('All tests done.\n', flush=True)


if __name__ == '__main__':
    test()
