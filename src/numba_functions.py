import numpy as np
from numba import jit
from Distribution import normal_log_pdf


@jit(nopython=True)
def backward_robust(A, B, observations):
    num_states = A.shape[0]
    num_observed = observations.shape[0]
    beta = np.zeros((num_observed, num_states))

    beta[observations.shape[0] - 1] = np.zeros(num_states)
    for t in range(num_observed - 2, -1, -1):
        for i in range(num_states):
            logbeta = 10000  # this will act as a hacky substitute for None because numba can't take in None 
            for j in range(num_states):
                logbeta = elnsum(logbeta, elnproduct(eln(A[j, i]),
                                                               elnproduct(normal_log_pdf(observations[t+1], B[j, 0], B[j, 1]),
                                                                               beta[t+1, j])))
            beta[t, i] = logbeta

    return beta


@jit(nopython=True)
def eexp(x):
    """

    :param x: x
    :return: exp(x)
    """
    if x == 10000:
        return 0
    else:
        return np.exp(x)


@jit(nopython=True)
def eln(x):
    """

    :param x: x
    :return: ln(x)
    """
    if x < 0:
        print("negative input error")
        return
    if x == 0:
        return np.nan
    elif x > 0:
        return np.log(x)


@jit(nopython=True)
def elnsum(x, y):
    """

    :param x: eln(x)
    :param y: eln(y)
    :return: ln(x + y)
    """
    if x == 10000 or y == 10000:
        if x == 10000:
            return y
        else:
            return x

    else:
        if x > y:
            return x + eln(1 + np.exp(y - x))
        else:
            return y + eln(1 + np.exp(x - y))


@jit(nopython=True)
def elnproduct(x, y):
    """

    :param x: eln(x)
    :param y: eln(y)
    :return: ln(x) + ln(y)
    """
    if x == 10000 or y == 10000:
        return np.nan
    else:
        return x + y


# @jit(nopython=True)
def sample_states_numba(beta: np.ndarray, initial_dist: np.ndarray, observations: np.ndarray,
                        mu: np.ndarray, sigma_invsq: float, A: np.ndarray, num_obs: int) -> np.ndarray:
    """
    numba version of function to sample from state distribution
    :param beta: backward algorithm probability ndarray
    :param initial_dist:
    :param observations:
    :param mu:
    :param sigma_invsq:
    :param A:
    :param num_obs:
    :return:
    """

    # equation (6) in /literature/Bayesian\ Model.pdf
    log_probabilities = np.log(initial_dist) + \
                        normal_log_pdf(observations[0], mu, np.sqrt(1/sigma_invsq)) + \
                        beta[0]

    probabilities = compute_probabilities(log_probabilities)

    # construct new np.array to hold the sampled state path
    new_state_path = np.empty(num_obs)

    new_state_path[0] = np.argmax(np.random.multinomial(1, np.maximum(np.minimum(probabilities / np.sum(probabilities), 1), 0)))  # sample new state

    for i in range(1, num_obs):
        # equation (7) in /literature/Bayesian\ Model.pdf
        log_probabilities = np.log(A[int(new_state_path[i-1]), :]) + \
                                   normal_log_pdf(observations[i], mu, np.sqrt(1/sigma_invsq)) + \
                                   beta[i]

        probabilities = compute_probabilities(log_probabilities)

        new_state_path[i] = np.argmax(np.random.multinomial(1, np.maximum(np.minimum(probabilities / np.sum(probabilities), 1), 0)))  # sample new state

    return new_state_path


@jit(nopython=True)
def compute_probabilities(log_probabilities):
    """
    Given a log-likelihood vector, convert to probability vector using method found in:
    https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability

    :param log_probabilities: log likelihood vector
    :return: probability vector
    """
    probabilities = np.exp(log_probabilities) / np.sum(np.exp(log_probabilities))  # make probs add up to 1

    if np.sum(probabilities) >= 0.99 and np.sum(probabilities) <= 1.01:  # just in case there is some underflow/overflow stuff
        return probabilities

    else:
        max = np.amax(log_probabilities)
        log_probabilities = log_probabilities - max
        valid_probabilities = np.empty(0)
        for i in range(len(log_probabilities)):
            if log_probabilities[i] > -38:
                valid_probabilities = np.append(valid_probabilities, np.exp(log_probabilities[i]))  # append likelihood, not log likelihood
            else:
                valid_probabilities = np.append(valid_probabilities, 0)

        return valid_probabilities / np.sum(valid_probabilities)


@jit(nopython=True)
def simulate_observations(num_obs, initial_state, emission_prob, state_transition):

    observations = np.zeros(num_obs)
    state_path = np.zeros(num_obs)

    curr_state = np.argmax(np.random.multinomial(1, initial_state, 1))

    for i in range(num_obs):
        state_path[i] = curr_state
        observations[i] = np.random.normal(emission_prob[curr_state, 0], emission_prob[curr_state, 1])
        curr_state = np.argmax(np.random.multinomial(1, state_transition[curr_state, :]))

    return observations