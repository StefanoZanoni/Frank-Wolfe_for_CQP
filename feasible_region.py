import argparse
import numpy as np
from src.index_set import create_index_sets
from src.constraints import create_A, create_b, BoxConstraints
from src.qp import QP
from src.solver import solve
import matplotlib.pyplot as plt

def plot_feasible_region(A, b, box_min, box_max, resolution=100, optimal_solution=None, filename='feasible_region.png'):
    """
    Plot the feasible region for a given set of linear constraints and box constraints.
    
    Parameters:
    A (np.ndarray): Coefficient matrix for the linear constraints.
    b (np.ndarray): Right-hand side vector for the linear constraints.
    box_min (float): Minimum value for the box constraints.
    box_max (float): Maximum value for the box constraints.
    resolution (int): Number of points to sample in each dimension.
    optimal_solution (np.ndarray, optional): The optimal solution point to be plotted. Default is None.
    optimal_minimums (np.ndarray or list of np.ndarray, optional): List of optimal minimum points to be plotted. Default is None.
    filename (str): The filename to save the plot. Default is 'feasible_region.png'.
    """
    dim = A.shape[1]
    
    # Generate points within the box constraints
    grid_points = np.meshgrid(*[np.linspace(box_min, box_max, resolution)] * dim)
    points = np.vstack([point.ravel() for point in grid_points]).T

    # Check which points satisfy the linear constraints
    feasible_points = points[np.all(np.dot(A, points.T).T <= b, axis=1)]
    
    
    if dim == 2:
        plt.scatter(feasible_points[:, 0], feasible_points[:, 1], s=1)
        if optimal_solution is not None:
            plt.scatter(optimal_solution[0], optimal_solution[1], color='red', marker='x')
        plt.xlim(box_min, box_max)
        plt.ylim(box_min, box_max)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Feasible Region')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feasible_points[:, 0], feasible_points[:, 1], feasible_points[:, 2], s=1)
        if optimal_solution is not None:
            ax.scatter(optimal_solution[0], optimal_solution[1], optimal_solution[2], color='red', marker='x')
        
        ax.set_xlim(box_min, box_max)
        ax.set_ylim(box_min, box_max)
        ax.set_zlim(box_min, box_max)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.title('Feasible Region')
    else:
        print("Visualization for dimensions greater than 3 is not supported.")
    
    plt.savefig(filename)
    print(f"Feasible region plot saved as {filename}")


def quadratic_function(x, Q, q): 
    """
    Args:
    x (numpy.ndarray): Input vector.
    Q (numpy.ndarray): Symmetric matrix.
    q (numpy.ndarray): Vector.

    Returns:
        float: Value of the quadratic function.
    """
    quadratic_term = 0.5 * np.dot(x.T, np.dot(Q, x))
    linear_term = np.dot(q.T, x)
    return quadratic_term + linear_term
        

def main():
    parser = argparse.ArgumentParser(description='This is a program to solve optimization problems using the '
                                                 'Frank-Wolfe algorithm.')
    parser.add_argument('--dimensions', '-n',
                        type=int,
                        default=3,
                        help='The dimension of the problem. Default is 10.')
    parser.add_argument('--rank', '-r',
                        type=float,
                        default=1,
                        help='The rank of the matrix Q. Default is 1.')
    parser.add_argument('--eccentricity', '-e',
                        type=float,
                        default=0.99,
                        help='The eccentricity of the matrix Q. Default is 0.9.')
    parser.add_argument('--active', '-a',
                        type=float,
                        default=1,
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

    # Aggregate all constraints for visualization
    A_agg = np.vstack(As)
    b_agg = np.concatenate([b for _ in As])

    constraints = [BoxConstraints(A, b, ineq=True) for A in As]
    problem = QP(n, rank=rank, eccentricity=eccentricity, active=active, c=False)

    solution, execution_time, iterations, _, _, optimal_minimums, approximated_minimums = (
        solve(problem, constraints, As, n, max_iter=max_iterations, verbose=verbose))

    plot_feasible_region(A_agg, b_agg, box_min=0, box_max=1, optimal_solution=solution)
    print(f"Execution Time: {round(execution_time * 1000, 4)} ms")
    print(f"Iterations for each sub-problem: {iterations}")
    print(f'Founded solution: {solution}')
    print(f'Optimal solution: {problem.minimum_point()}')
    print(f'Optimal minimums: {optimal_minimums}')
    print(f'Approximated minimums: {approximated_minimums}')

    
if __name__ == '__main__':
    main()
