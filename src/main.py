import numpy as np
import argparse
import time

from src.index_set import create_index_sets, create_feasible_point
from src.constraints import create_A, create_b
from src.constraints import BoxConstraints
from src.cqp import BCQP
from src.frank_wolfe import frank_wolfe
from src.qp import QP


def solve(problem, constraints, x0, As, n):
    x_optimal = np.zeros(n)

    start = time.time()
    for i, c in enumerate(constraints):
        indexes = As[i][0] != 0
        x_init = x0[indexes]
        bcqp = BCQP(problem, c)
        bcqp.problem.subQ = bcqp.problem.Q[indexes][:, indexes]
        bcqp.problem.subq = bcqp.problem.q[indexes]
        x_i = frank_wolfe(bcqp, x_init)
        x_optimal[indexes] = x_i
    end = time.time()

    return x_optimal, end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dimension', type=int, help='problem dimension')
    n = parser.parse_args().dimension

    Is = create_index_sets(n, 3, uniform=True)
    As = [create_A(n, I) for I in Is]
    b = create_b()

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=n, c=False, seed=1)
    x0 = create_feasible_point(n, Is)

    x_optimal, execution_time = solve(problem, constraints, x0, As, n)
    print(f'Optimal solution: {x_optimal}')
    print(f'Execution time: {execution_time}')


if __name__ == '__main__':
    main()
