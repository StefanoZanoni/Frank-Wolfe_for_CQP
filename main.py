import numpy as np
import argparse
import time

from src.index_set import create_index_sets, create_feasible_point
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.cqp import BCQP
from src.frank_wolfe import frank_wolfe
from src.qp import QP


# The solve function solves the optimization problem for each constraint
# and returns the optimal solution, execution time, and the number of iterations.
def solve(problem, constraints, x0, As, n):
    """
    Solve the optimization problem for each constraint.

    :param problem: The optimization problem.
    :param constraints: The constraints of the problem.
    :param x0: The initial point.
    :param As: The matrices of constraints.
    :param n: The dimension of the problem.
    :return: The optimal solution, execution time, and the number of iterations.
    """
    x_optimal = np.zeros(n)
    iterations = []

    start = time.time()
    for i, c in enumerate(constraints):
        print(f'Subproblem {i}')
        # compute the indexes of k-th I index set
        indexes = As[i][0] != 0
        # get the sub x_init relative to the indexes
        x_init = x0[indexes]
        # construct the BCQP problem from the actual constraints
        bcqp = BCQP(problem, c)
        # consider only the subproblem relative to the indexes
        bcqp.problem.set_subproblem(indexes)
        # solve the subproblem
        x_i, iter = frank_wolfe(bcqp, x_init, eps=1e-6, max_iter=100)
        # merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i
        iterations.append(iter)
        print('\n')
    end = time.time()

    return x_optimal, end - start, iterations


# The main function parses the command line arguments, creates the problem and constraints, and solves the problem.
def main():
    """
    Parse the command line arguments, create the problem and constraints, and solve the problem.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dimension', type=int, help='problem dimension')
    n = parser.parse_args().dimension

    Is = create_index_sets(n, 3, uniform=True)
    As = [create_A(n, I) for I in Is]
    b = create_b()

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=n, eccentricity=0.9, active=1, c=False)
    x0 = create_feasible_point(n, Is)

    x_optimal, execution_time, iterations = solve(problem, constraints, x0, As, n)
    print(f'\nOptimal solution: {x_optimal}\n')
    print(f'Execution time: {round(execution_time * 1000, 4)}ms\n')
    print(f'Iterations for each subproblem: {iterations}')


if __name__ == '__main__':
    main()
