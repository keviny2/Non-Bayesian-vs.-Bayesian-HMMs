import pandas as pd
import numpy as np

def forward(observations, A, B, initial):
    num_states = A.shape[0]
    num_observed = observations.shape[0]
    alpha = np.zeros((num_observed, num_states)) # store values for previous alphas
    alpha[0, :] = initial * B[:, observations[0] - 1]

    for t in range(1, num_observed):
        for j in range(num_states):
            alpha[t, j] = alpha[t - 1].dot(A[:, j]) * B[j, observations[t] - 1]

    return alpha


def backward(observations, A, B):
    num_states = A.shape[0]
    num_observed = observations.shape[0]
    beta = np.zeros((num_observed, num_states))

    # setting beta(T) = 1
    beta[observations.shape[0] - 1] = np.ones((num_states))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(num_observed - 2, -1, -1):
        for j in range(num_states):
            beta[t, j] = (beta[t + 1] * B[:, observations[t + 1] - 1]).dot(A[j, :])

    return beta


def baum_welch(observations, A, B, initial, n_iter=100):
    num_states = A.shape[0]
    T = len(observations)

    for n in range(n_iter):
        alpha = forward(observations, A, B, initial)
        beta = backward(observations, A, B)

        xi = np.zeros((num_states, num_states, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, A) * B[:, observations[t + 1] - 1].T, beta[t + 1, :])
            for i in range(num_states):
                numerator = alpha[t, i] * A[i, :] * B[:, observations[t + 1] - 1].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = B.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            B[:, l] = np.sum(gamma[:, observations == l + 1], axis=1)

        B = np.divide(B, denominator.reshape((-1, 1)))

    return {"a": A, "b": B}

if __name__ == '__main__':
    obs = pd.read_csv('../data/dummy_data.csv')['Visible'].values + 1
    pi = np.array([.5, .5])  # initial dist.
    # Transition Probabilities
    A = np.ones((2, 2))
    A = A / np.sum(A, axis=1)

    # Emission Probabilities
    B = np.array(((1, 3, 5), (2, 4, 6)))
    B = B / np.sum(B, axis=1).reshape((-1, 1))

    res = baum_welch(obs, A , B, pi)
    pass