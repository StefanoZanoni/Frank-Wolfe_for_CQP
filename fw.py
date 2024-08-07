import argparse

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import Constraints
from src.qp import QP
from src.cqp import CQP
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
    parser.add_argument('--plot', '-p',
                        action='store_true',
                        help='If present, plots will be generated.')
    parser.add_argument('--directory', '-d',
                        type=str,
                        default='./',
                        help='the directory where to save the plots. Default is current directory.'
                             ' Ignored if --plot or -p is not present')
    parser.add_argument('--pltrange', '-pr',
                        type=tuple[float],
                        default=(0, 1),
                        help='The range of the axis for the plot. Default is (0, 1).'
                             ' Axis_range[0] is the minimum value, axis_range[1] is the maximum value.'
                             ' Ignored if --plot or -p is not present.')
    parser.add_argument('--onedge', '-oe',
                        action='store_true',
                        help='If True, the starting point will be on the edge of the feasible region. Default is True.')
    args = parser.parse_args()

    n = args.dimensions
    rank = args.rank
    eccentricity = args.eccentricity
    active = args.active
    max_iterations = args.iterations
    verbose = args.verbose
    plot = args.plot
    directory = args.directory
    axis_range = args.pltrange
    on_edge = args.onedge
    if verbose != 0:
        verbose = 1

    Is = create_index_sets(n, uniform=False)
    A = create_A(n, Is)
    b = create_b(n, len(Is))

    constraints = Constraints(A, b, len(Is), n, ineq=True)
    problem = QP(n, Is, rank=rank, eccentricity=eccentricity, active=active, c=False)
    cqp = CQP(problem, constraints)

    solution, execution_time, iterations, gaps, convergence_rates, positions = (
        solve(cqp, init_edge=on_edge, max_iter=max_iterations, verbose=verbose, plot=plot,
              axis_range=axis_range, dirname=directory))

    if verbose == 1:
        print(f"Execution Time: {round(execution_time * 1000, 4)} ms")
        print(f'Positions in the feasible region: {positions}')


if __name__ == '__main__':
    main()
