import numpy as np
import argparse

from src.index_set import create_index_sets, create_feasible_point
from src.constraints import create_A, create_b
from src.constraints import C
from src.cqp import CQP
from src.frank_wolfe import frank_wolfe
from src.qp import QP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('integer', metavar='N', type=int, help='problem size')
    n = parser.parse_args().integer
    n = 10

    Is = create_index_sets(n, 3, uniform=True)
    As = [create_A(n, I) for I in Is]
    b = create_b()
    constraints = [C(A, b, ineq=True) for A in As]
    problem = QP(n, rank=n, c=False, seed=1)
    x0 = create_feasible_point(10, Is)
    x_optimal = np.zeros(n)
    for i, c in enumerate(constraints):
        indexes = As[i][0] != 0
        x_init = x0[indexes]
        cqp = CQP(problem, c)
        cqp.problem.Q = cqp.problem.Q[indexes][:, indexes]
        cqp.problem.q = cqp.problem.q[indexes]
        x_i = frank_wolfe(cqp, x_init)
        x_optimal[indexes] = x_i
    print(x_optimal)


if __name__ == '__main__':
    main()
