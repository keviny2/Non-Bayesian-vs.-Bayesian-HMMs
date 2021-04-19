import scipy.stats as stats
import numpy as np
from numba import jit


def normal_pdf(x, mu, sigma=0.5):
    """
    :param x: observation
    :param mu: mean
    :param sigma: standard deviation
    :return: probability
    """
    return stats.norm.pdf(x, mu, sigma)


@jit(nopython=True)
def normal_log_pdf(val, mean, variance):
    if np.any(variance) <= 0:
        raise ValueError("Tried Pass through a variance that is less than or equal to 0 for gene {} at iteration {} ")
    return -0.5*np.log(2*np.pi) - np.log(np.sqrt(variance)) - (0.5/variance)*(val-mean) ** 2

