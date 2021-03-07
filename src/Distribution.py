import scipy.stats as stats


def normal_pdf(x, mu, sigma=0.5):
    """
    :param x: observation
    :param mu: mean
    :param sigma: standard deviation
    :return: probability
    """
    return stats.norm.pdf(x, mu, sigma)

