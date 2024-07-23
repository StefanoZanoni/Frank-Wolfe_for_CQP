import plotly.graph_objects as go
import numpy as np
import time
import matplotlib.pyplot as plt

from src.frank_wolfe import frank_wolfe
from src.cqp import CQP
from src.constraints import Constraints


def plot_cqp(cqp: CQP, bounded_minimum_point: np.ndarray, bounded_minimum: float, filename: str,
             axis_range: tuple[float] = (-10, 10)) -> None:
    """
    Plots the CQP problem's function, constrained values, constrained minimum, and global minimum.

    Parameters:
    - cqp (CQP): The constrained quadratic problem instance.
    - bounded_minimum_point (np.ndarray): The point of the bounded minimum.
    - bounded_minimum (float): The value of the bounded minimum.
    - filename (str): The filename for the output HTML plot.
    - axis_range (tuple[float, float], optional): The range of the axis for plotting. Defaults to (-10, 10).

    Returns:
    - None
    """

    min_value = axis_range[0]
    max_value = axis_range[1]

    if cqp.problem.dim == 1:
        x_values = np.linspace(min_value, max_value, 400)
        x_values_constrained = np.linspace(0, 1, 3)

        # Calculate the corresponding y values
        y_values = np.array([cqp.problem.evaluate(np.array([x])) for x in x_values])

        # Calculate the constrained y values
        constraints = np.array([cqp.constraints.evaluate(np.array([x])) for x in x_values_constrained])
        y_values_constrained = np.array([cqp.problem.evaluate(np.array([x])) for x in x_values_constrained])
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
        fig.add_trace(go.Scatter(x=cqp.problem.minimum_point(),
                                 y=[cqp.problem.minimum()],
                                 mode='markers',
                                 name='Global Minimum',
                                 marker=dict(color=[1], colorscale='YlOrRd')))

        # Add title and labels
        fig.update_layout(title='Quadratic Function Plot', xaxis_title='x', yaxis_title='f(x)')

    elif cqp.problem.dim == 2:
        x_values = np.linspace(min_value, max_value, 400)
        y_values = np.linspace(min_value, max_value, 400)
        x_values_constrained = np.linspace(0, 1, 400)
        y_values_constrained = np.linspace(0, 1, 400)

        # Create a meshgrid for x and y values
        X, Y = np.meshgrid(x_values, y_values)
        X_constrained, Y_constrained = np.meshgrid(x_values_constrained, y_values_constrained)

        # Calculate the corresponding z values
        Z = np.array([cqp.problem.evaluate(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        # Calculate the constrained z values
        constraints = np.array([cqp.constraints.evaluate(np.array([x, y]))
                                for x, y in zip(np.ravel(X_constrained), np.ravel(Y_constrained))])
        constraints = constraints.reshape(X_constrained.shape)
        Z_constrained = np.array([cqp.problem.evaluate(np.array([x, y]))
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
        fig.add_trace(go.Scatter3d(x=[cqp.problem.minimum_point()[0]],
                                   y=[cqp.problem.minimum_point()[1]],
                                   z=[cqp.problem.minimum()],
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


def solve(cqp: CQP, init_edge: bool = True, max_iter: int = 2000, verbose: int = 1, plot: bool = False,
          dirname: str = './', axis_range: tuple[int] = (-10, 10)):
    """
    Solves the CQP problem using the Frank-Wolfe algorithm.

    Parameters:
    - cqp (CQP): The constrained quadratic problem instance.
    - init_edge (bool, optional): If True, initializes the starting point at the edge. Defaults to True.
    - max_iter (int, optional): The maximum number of iterations for the Frank-Wolfe algorithm. Defaults to 2000.
    - verbose (int, optional): The verbosity level. If 1, prints detailed logs. Defaults to 1.
    - plot (bool, optional): If True, plots the problem. Defaults to False.
    - dirname (str, optional): The directory name to save plots if plotting is enabled. Defaults to './'.
    - axis_range (tuple[int, int], optional): The range of the axis for plotting. Defaults to (-10, 10).

    Returns:
    - A tuple containing the optimal solution, the time taken, iterations, gaps, convergence rates,
     and the position of the solution found.
    """

    dim = cqp.problem.dim

    x_init = np.zeros(dim)
    # Initialize the starting point
    for k in range(cqp.constraints.K):
        I_k = cqp.constraints.A[k, :] != 0
        indices = np.where(I_k)[0]
        if init_edge:
            x_init[indices[0]] = 1
        else:
            for index in indices:
                x_init[index] = 1 / len(indices)

    # Solve the problem
    start = time.time()
    x, v, iterations, gaps, convergence_rates = frank_wolfe(cqp, x_init,
                                                            eps=1e-6,
                                                            max_iter=max_iter,
                                                            verbose=verbose)
    end = time.time()

    position = cqp.constraints.check_position(x)

    if plot:
        filename = dirname + f'plot_cqp.png'
        plot_cqp(cqp, x, v, filename, axis_range=axis_range)

    return x, end - start, iterations, gaps, convergence_rates, position
