import numpy as np
import argparse
import time

from src.index_set import create_index_sets
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.cqp import BCQP
from src.frank_wolfe import frank_wolfe
from src.qp import QP


# The solve function solves the optimization problem for each constraint
# and returns the optimal solution, execution time, and the number of iterations.
def solve(problem, constraints, As, n) -> tuple[np.ndarray, float, list]:
    """
    Solve the optimization problem for each constraint.

    :param problem: The optimization problem.
    :param constraints: The constraints of the problem.
    :param As: The matrices of constraints.
    :param n: The dimension of the problem.
    :return: The optimal solution, execution time, and the number of iterations.
    """
    x_optimal = np.empty(n)
    iterations = []

    start = time.time()
    for i, c in enumerate(constraints):
        print(f'Sub problem {i}')
        # compute the indexes of k-th I index set
        indexes = As[i][0] != 0
        # initialize the starting point
        x_init = np.zeros(sum(indexes))
        x_init[0] = 1
        # construct the BCQP problem from the actual constraints
        bcqp = BCQP(problem, c)
        # consider only the subproblem relative to the indexes
        bcqp.problem.set_subproblem(indexes)
        # solve the subproblem
        x_i, iteration = frank_wolfe(bcqp, x_init, eps=1e-6, max_iter=1000)
        # merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i
        iterations.append(iteration)
        print('\n')
    end = time.time()

    return x_optimal, end - start, iterations


# The main function parses the command line arguments, creates the problem and constraints, and solves the problem.
def main():
    """
    Parse the command line arguments, create the problem and constraints, and solve the problem.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dimension', type=int, help='problem dimension', nargs='?', default=10)
    n = parser.parse_args().dimension

    Is = create_index_sets(n, 3, uniform=False)
    As = [create_A(n, I) for I in Is]
    b = create_b()

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=n, eccentricity=0.9, active=1, c=False)

    x_optimal, execution_time, iterations = solve(problem, constraints, As, n)
    print(f'\nOptimal solution: {x_optimal}\n')
    print(f'Execution time: {round(execution_time * 1000, 4)}ms\n')
    print(f'Iterations for each subproblem: {iterations}')


if __name__ == '__main__':
    main()
