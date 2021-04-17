import os
import matplotlib.pyplot as plt


def plot(data, ylabel, name):
    plt.figure()
    plt.plot(data)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    fname = os.path.join("..", "plots", name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)