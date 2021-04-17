import os
import matplotlib.pyplot as plt


def plot(data, xlabel, ylabel, name):
    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fname = os.path.join("", "plots", name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)