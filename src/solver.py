import plotly.graph_objects as go
import numpy as np
import time
import matplotlib.pyplot as plt

from src.frank_wolfe import frank_wolfe
from src.cqp import BCQP
from src.constraints import BoxConstraints


def plot_bcqp(bcqp: BCQP, bounded_minimum_point: np.ndarray, bounded_minimum: float, filename: str,
              axis_range: tuple[float] = (-10, 10)) -> None:
    """
    Plots the BCQP problem's function, constrained values, constrained minimum, and global minimum.

    Parameters:
    - bcqp (BCQP): The box constrained quadratic problem instance.
    - bounded_minimum_point (np.ndarray): The point of the bounded minimum.
    - bounded_minimum (float): The value of the bounded minimum.
    - filename (str): The filename for the output HTML plot.
    - axis_range (tuple[float, float], optional): The range of the axis for plotting. Defaults to (-10, 10).

    Returns:
    - None
    """

    min_value = axis_range[0]
    max_value = axis_range[1]

    if bcqp.problem.get_dim() == 1:
        x_values = np.linspace(min_value, max_value, 400)
        x_values_constrained = np.linspace(0, 1, 3)

        # Calculate the corresponding y values
        y_values = np.array([bcqp.problem.evaluate(np.array([x])) for x in x_values])

        # Calculate the constrained y values
        constraints = np.array([bcqp.constraints.evaluate(np.array([x])) for x in x_values_constrained])
        y_values_constrained = np.array([bcqp.problem.evaluate(np.array([x])) for x in x_values_constrained])
        y_values_constrained = y_values_constrained[constraints]
        x_values_constrained = x_values_constrained[constraints]

        # Create the plot
        fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, name='Function'))

        # Add the constrained values to the plot
        fig.add_trace(go.Scatter(x=x_values_constrained,
                                 y=y_values_constrained,
                                 mode='markers',
                                 name='Constrained Values',
                                 marker=dict(color=np.linspace(0, 1, len(y_values_constrained)), colorscale='YlGnBu')))

        # Add the constrained minimum to the plot
        fig.add_trace(go.Scatter(x=bounded_minimum_point,
                                 y=[bounded_minimum],
                                 mode='markers',
                                 name='Constrained Minimum',
                                 marker=dict(color=[1], colorscale='YlGnBu')))

        # Add the global minimum to the plot
        fig.add_trace(go.Scatter(x=bcqp.problem.minimum_point(),
                                 y=[bcqp.problem.minimum()],
                                 mode='markers',
                                 name='Global Minimum',
                                 marker=dict(color=[1], colorscale='YlOrRd')))

        # Add title and labels
        fig.update_layout(title='Quadratic Function Plot', xaxis_title='x', yaxis_title='f(x)')

    elif bcqp.problem.get_dim() == 2:
        x_values = np.linspace(min_value, max_value, 400)
        y_values = np.linspace(min_value, max_value, 400)
        x_values_constrained = np.linspace(0, 1, 400)
        y_values_constrained = np.linspace(0, 1, 400)

        # Create a meshgrid for x and y values
        X, Y = np.meshgrid(x_values, y_values)
        X_constrained, Y_constrained = np.meshgrid(x_values_constrained, y_values_constrained)

        # Calculate the corresponding z values
        Z = np.array([bcqp.problem.evaluate(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        # Calculate the constrained z values
        constraints = np.array([bcqp.constraints.evaluate(np.array([x, y]))
                                for x, y in zip(np.ravel(X_constrained), np.ravel(Y_constrained))])
        constraints = constraints.reshape(X_constrained.shape)
        Z_constrained = np.array([bcqp.problem.evaluate(np.array([x, y]))
                                  for x, y in zip(np.ravel(X_constrained), np.ravel(Y_constrained))])
        Z_constrained = Z_constrained.reshape(X_constrained.shape)
        Z_constrained = Z_constrained[constraints]
        X_constrained = X_constrained[constraints]
        Y_constrained = Y_constrained[constraints]

        # Create the plot
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, name='Function surface')])

        # Add the constrained values to the plot
        fig.add_trace(go.Scatter3d(x=X_constrained,
                                   y=Y_constrained,
                                   z=Z_constrained,
                                   mode='markers',
                                   name='Constrained Values',
                                   marker=dict(color=np.linspace(0, 1, X_constrained.shape[0]), colorscale='YlGnBu')))

        # Add the constrained minimum to the plot
        fig.add_trace(go.Scatter3d(x=[bounded_minimum_point[0]],
                                   y=[bounded_minimum_point[1]],
                                   z=[bounded_minimum],
                                   mode='markers',
                                   name='Constrained Minimum',
                                   marker=dict(color=[1], colorscale='YlGnBu', size=2)))

        # Add the global minimum to the plot
        fig.add_trace(go.Scatter3d(x=[bcqp.problem.minimum_point()[0]],
                                   y=[bcqp.problem.minimum_point()[1]],
                                   z=[bcqp.problem.minimum()],
                                   mode='markers',
                                   name='Global Minimum',
                                   marker=dict(color=[1], colorscale='YlOrRd', size=2)))

        # Add title and labels
        fig.update_layout(title='Quadratic Function Plot', xaxis_title='x', yaxis_title='f(x)',
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01,
                              bgcolor='LightSteelBlue',
                              bordercolor='Black',
                              borderwidth=2
                          ))

    else:
        print("The dimension of the problem is not supported for plotting.\n")
        return

    fig.write_html(filename + '.html')


def solve(bcqp: BCQP, init_edge: bool = True, max_iter: int = 2000, verbose: int = 1, plot: bool = False,
          dirname: str = './', axis_range: tuple[int] = (-10, 10)):
    """
    Solves the BCQP problem using the Frank-Wolfe algorithm.

    Parameters:
    - bcqp (BCQP): The box constrained quadratic problem instance.
    - init_edge (bool, optional): If True, initializes the starting point at the edge. Defaults to True.
    - max_iter (int, optional): The maximum number of iterations for the Frank-Wolfe algorithm. Defaults to 2000.
    - verbose (int, optional): The verbosity level. If 1, prints detailed logs. Defaults to 1.
    - plot (bool, optional): If True, plots the subproblems. Defaults to False.
    - dirname (str, optional): The directory name to save plots if plotting is enabled. Defaults to './'.
    - axis_range (tuple[int, int], optional): The range of the axis for plotting. Defaults to (-10, 10).

    Returns:
    - A tuple containing the optimal solution, the time taken, iterations, gaps, convergence rates, and positions.
    """

    dim = bcqp.problem.dim
    x_optimal = np.empty(dim)
    iterations = []
    all_gaps = []
    all_convergence_rates = []
    positions = []

    start = time.time()

    for k in range(bcqp.constraints.K):
        if verbose == 1:
            print(f'Sub problem {k}')

        # Compute the indexes of k-th index set I
        indexes = bcqp.constraints.A[k, :] != 0
        sum_indexes = sum(indexes)
        # Initialize the starting point
        if init_edge:
            x_init = np.zeros(sum_indexes)
            x_init[0] = 1
        else:
            x_init = np.full(sum_indexes, 1 / sum_indexes)
        # Consider only the subproblem relative to the indexes
        bcqp.set_subproblem(k, indexes)
        # Solve the subproblem
        x_i, v_i, iteration, gaps, convergence_rates = frank_wolfe(bcqp, x_init,
                                                                   eps=1e-6,
                                                                   max_iter=max_iter,
                                                                   verbose=verbose)
        # Merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i

        iterations.append(iteration)
        all_gaps.append(gaps)
        all_convergence_rates.append(convergence_rates)

        # Check the position of the solution found
        position = bcqp.constraints.check_position(x_i)
        positions.append(position)

        if verbose == 1:
            print('\n')
        if plot:
            filename = dirname + f'plot_bcqp_{k}.png'
            plot_bcqp(bcqp, x_i, v_i, filename, axis_range=axis_range)

    end = time.time()

    return x_optimal, end - start, iterations, all_gaps, all_convergence_rates, positions
