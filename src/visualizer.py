import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from simulate import SimulateData


def plot(data, ylabel, name, bayesian):
    plt.figure()
    plt.plot(data)
    plt.xlabel("Index")
    plt.ylabel(ylabel)

    if bayesian:
        fname = os.path.join("..", "plots", 'bayesian', name)

    else:
        fname = os.path.join("..", "plots", 'maxlike', name)

    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


def plot_mu(chain, num_states, num_iter, max=None, name = "mu_trace_plots"):
    """
    plot the path of mus in MCMC chain
    :param num_iter: number of iterations in the chain
    :param max: maximum number of values to plot
    :return:
    """

    if max is None:
        max = num_iter

    mus = [entry['mu'] for entry in chain]
    mus = mus[-max:]
    mus = np.asmatrix(mus)  # row i corresponds to the mus in the chain at iteration i

    x = np.linspace(1, max, num=max)
    plt.figure()

    for i in range(num_states):
        plt.plot(x, mus[:, i])

    plt.show()

    fname = os.path.join("..", "plots", "bayesian", name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


def KDEplot(data):

    sns.set_style('whitegrid')
    sns.distplot(data, kde=True)
    fname = os.path.join("..", "plots", "KDE")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)
