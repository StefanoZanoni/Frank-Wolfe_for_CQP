import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import time

from src.frank_wolfe import frank_wolfe
from src.cqp import BCQP
from src.qp import QP
from src.constraints import BoxConstraints


def plot_bcqp(bcqp: BCQP, bounded_minimum_point: np.ndarray, bounded_minimum: float, filename: str) -> None:

    if bcqp.problem.get_dim() == 1:
        x_values = np.linspace(-10, 10, 400)
        x_values_constrained = np.linspace(0, 1, 400)

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

        # add the constrained minimum to the plot
        fig.add_trace(go.Scatter(x=bounded_minimum_point,
                                 y=[bounded_minimum],
                                 mode='markers',
                                 name='Constrained Minimum',
                                 marker=dict(color=[1], colorscale='YlGnBu')))

        # add the global minimum to the plot
        fig.add_trace(go.Scatter(x=bcqp.problem.minimum_point(),
                                 y=[bcqp.problem.minimum()],
                                 mode='markers',
                                 name='Global Minimum',
                                 marker=dict(color=[1], colorscale='YlOrRd')))

        # Add title and labels
        fig.update_layout(title='Quadratic Function Plot', xaxis_title='x', yaxis_title='f(x)')

    elif bcqp.problem.get_dim() == 2:
        x_values = np.linspace(-40, 40, 400)
        y_values = np.linspace(-40, 40, 400)
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

        # Create the plot
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, name='Function surface')])

        # Add the constrained values to the plot
        fig.add_trace(go.Scatter3d(x=X_constrained,
                                   y=Y_constrained,
                                   z=Z_constrained,
                                   mode='markers',
                                   name='Constrained Values',
                                   marker=dict(color=np.linspace(0, 1, X_constrained.shape[0]), colorscale='YlGnBu')))

        # add the constrained minimum to the plot
        fig.add_trace(go.Scatter3d(x=[bounded_minimum_point[0]],
                                   y=[bounded_minimum_point[1]],
                                   z=[bounded_minimum],
                                   mode='markers',
                                   name='Constrained Minimum',
                                   marker=dict(color=[1], colorscale='YlGnBu', size=3)))

        # add the global minimum to the plot
        fig.add_trace(go.Scatter3d(x=[bcqp.problem.minimum_point()[0]],
                                   y=[bcqp.problem.minimum_point()[1]],
                                   z=[bcqp.problem.minimum()],
                                   mode='markers',
                                   name='Global Minimum',
                                   marker=dict(color=[1], colorscale='YlOrRd', size=3)))

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
        print("The dimension of the problem is not supported for plotting.")
        return

    fig.write_html(filename + '.html')


def solve(problem: QP, constraints: list[BoxConstraints], As: list[np.ndarray], n: int, max_iter: int = 1000,
          verbose: int = 1, plot: bool = False, dirname: str = './') \
        -> tuple[np.ndarray, float, list, list[list], list[list], list[float], list[float]]:
    """
    Solve the optimization problem for each index set.

    :param problem: The optimization problem.
    :param constraints: The constraints of the problem.
    :param As: The matrices of constraints.
    :param n: The dimension of the problem.
    :param max_iter: The maximum number of iterations.
    :param verbose: The verbosity level.
    :param plot: Whether to plot the results.
    :param dirname: The directory where to save the plots.
    :return: The optimal solution, execution time, the number of iterations,
    all the gap histories, all the convergence rate histories, the optimal minimums and the approximated minimums.
    """

    x_optimal = np.empty(n)
    iterations = []
    all_gaps = []
    all_convergence_rates = []
    optimal_minimums = []
    approximated_minimums = []

    start = time.time()
    for k, c in enumerate(constraints):
        if verbose == 1:
            print(f'Sub problem {k}')
        # compute the indexes of k-th index set I
        indexes = As[k][0] != 0
        # initialize the starting point
        x_init = np.zeros(sum(indexes))
        x_init[0] = 1
        # construct the BCQP problem from the actual constraints
        bcqp = BCQP(problem, c)
        # consider only the subproblem relative to the indexes
        bcqp.set_subproblem(indexes)
        # solve the subproblem
        x_i, v_i, iteration, gaps, convergence_rates = frank_wolfe(bcqp, x_init, eps=1e-6, max_iter=max_iter,
                                                                   verbose=verbose)
        # merge the subproblem solution with the optimal solution
        x_optimal[indexes] = x_i
        iterations.append(iteration)
        all_gaps.append(gaps)
        all_convergence_rates.append(convergence_rates)
        optimal_minimums.append(bcqp.problem.minimum())
        approximated_minimums.append(v_i)
        if verbose == 1:
            print('\n')
        if plot:
            filename = dirname + f'plot_bcqp_{k}.png'
            plot_bcqp(bcqp, x_i, v_i, filename)
    end = time.time()

    return x_optimal, end - start, iterations, all_gaps, all_convergence_rates, optimal_minimums, approximated_minimums
