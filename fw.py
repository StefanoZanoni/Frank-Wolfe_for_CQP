import argparse

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.qp import QP
from src.solver import solve


def main():
    parser = argparse.ArgumentParser(description='This is a program to solve optimization problems using the '
                                                 'Frank-Wolfe algorithm.')
    parser.add_argument('--dimensions', '-n',
                        type=int,
                        default=10,
                        help='The dimension of the problem. Default is 10.')
    parser.add_argument('--rank', '-r',
                        type=float,
                        default=1,
                        help='The rank of the matrix Q. Default is 1.')
    parser.add_argument('--eccentricity', '-e',
                        type=float,
                        default=0.9,
                        help='The eccentricity of the matrix Q. Default is 0.9.')
    parser.add_argument('--active', '-a',
                        type=float,
                        default=1.0,
                        help='The active constraints percentage of the problem. Default is 1.0.')
    parser.add_argument('--iterations', '-i',
                        type=int,
                        default=1000,
                        help='The maximum number of iterations. Default is 1000.')
    parser.add_argument('--verbose', '-v',
                        type=int,
                        default=1,
                        help='The verbosity level. Default is 1.')
    args = parser.parse_args()

    n = args.dimensions
    rank = args.rank
    eccentricity = args.eccentricity
    active = args.active
    max_iterations = args.iterations
    verbose = args.verbose
    if verbose != 0:
        verbose = 1

    Is = create_index_sets(n, uniform=False)
    As = [create_A(n, I) for I in Is]
    b = create_b()

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False)

    solution, execution_time, iterations, _, _, optimal_minimums, approximated_minimums = (
        solve(problem, constraints, As, n, max_iter=max_iterations, verbose=verbose))

    print(f"Execution Time: {round(execution_time * 1000, 4)} ms")
    print(f"Iterations for each sub-problem: {iterations}")
    print(f'Founded solution: {solution}')
    print(f'Optimal minimums: {optimal_minimums}')
    print(f'Bounded minimums: {approximated_minimums}')


if __name__ == '__main__':
    main()
