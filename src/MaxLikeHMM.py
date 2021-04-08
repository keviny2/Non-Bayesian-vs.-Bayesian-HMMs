import numpy as np
from Distribution import normal_pdf
from SimulateData import SimulateData

class MaxLikeHMM:

    def __init__(self, observations=None):
        """

        :param observations: vector of observations
        """
        self.observations = observations

    def forward(self, A, B, initial):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        alpha = np.zeros((num_observed, num_states)) # store values for previous alphas
        alpha[0, :] = initial * B[:, self.observations[0] - 1]

        for t in range(1, num_observed):
            for j in range(num_states):
                alpha[t, j] = alpha[t - 1].dot(A[:, j]) * B[j, self.observations[t] - 1]

        return alpha


    def forward_continuous(self, A, B, initial):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        alpha = np.zeros((num_observed, num_states)) # store values for previous alphas
        alpha[0, :] = initial * normal_pdf(self.observations[0], B[:, 0], B[:, 1])

        for t in range(1, num_observed):
            for j in range(num_states):
                alpha[t, j] = alpha[t - 1].dot(A[:, j]) * normal_pdf(self.observations[t], B[j, 0], B[j, 1])

        return alpha


    def backward(self, A, B):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        beta = np.zeros((num_observed, num_states))

        # setting beta(T) = 1
        beta[self.observations.shape[0] - 1] = np.ones((num_states))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(num_observed - 2, -1, -1):
            for j in range(num_states):
                beta[t, j] = (beta[t + 1] * B[:, self.observations[t + 1] - 1]).dot(A[j, :])

        return beta


    def backward_continuous(self, A, B):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        beta = np.zeros((num_observed, num_states))

        # setting beta(T) = 1
        beta[self.observations.shape[0] - 1] = np.ones((num_states))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(num_observed - 2, -1, -1):
            for j in range(num_states):
                beta[t, j] = (beta[t + 1] * normal_pdf(self.observations[t + 1], B[j, 0], B[j, 1])).dot(A[j, :])
        return beta

    def baum_welch(self, A, B, initial, n_iter=100):
        num_states = A.shape[0]
        T = len(self.observations)

        for n in range(n_iter):
            alpha = self.forward(A, B, initial)
            beta = self.backward(A, B)

            xi = np.zeros((num_states, num_states, T - 1))
            for t in range(T - 1):
                denominator = np.dot(np.dot(alpha[t, :].T, A) * B[:, self.observations[t + 1] - 1].T, beta[t + 1, :])
                for i in range(num_states):
                    numerator = alpha[t, i] * A[i, :] * B[:, self.observations[t + 1] - 1].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

            K = B.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                B[:, l] = np.sum(gamma[:, self.observations == l + 1], axis=1)

            B = np.divide(B, denominator.reshape((-1, 1)))

        return {"a": A, "b": B}


    # def baum_welch_continuous(self, A, B, initial, n_iter=100):
    #     """
    #
    #     :param A: state transition matrix
    #     :param B: [[mu1,sigma1],
    #                [mu2,sigma2],
    #                 ....]
    #     :param initial: initial probabilities
    #     :param n_iter: number of iterations
    #     :return: updated state transition and emission matrices
    #     """
    #     num_states = A.shape[0]
    #     T = len(self.observations)
    #
    #     # BUG: after running for many iterations either:
    #     #  1. the matrix becomes symmetric (shouldn't do this)
    #     #  2. nan values show up
    #     for n in range(n_iter):
    #         alpha = self.forward_continuous(A, B, initial)
    #         beta = self.backward_continuous(A, B)
    #
    #         xi = np.zeros((num_states, num_states, T - 1))
    #         for t in range(T - 1):
    #             denominator = np.dot(np.dot(alpha[t, :].T, A) * np.array([normal_pdf(self.observations[t + 1], B[state, 0], B[state, 1]) for state in range(num_states)]).T, beta[t + 1, :])
    #             for i in range(num_states):
    #                 numerator = alpha[t, i] * A[i, :] * np.array([normal_pdf(self.observations[t + 1], B[state, 0], B[state, 1]) for state in range(num_states)]).T * beta[t + 1, :].T
    #                 xi[i, :, t] = numerator / denominator
    #
    #         gamma = np.sum(xi, axis=1)
    #         A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
    #
    #         # Add additional T'th element in gamma
    #         gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
    #
    #         # formulas from U of T lecture slides:
    #         # http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture11.pdf
    #         K = B.shape[0]
    #         for l in range(K):
    #             denominator = np.sum(gamma[l, :])
    #             new_mu = np.dot(gamma[l, :], self.observations)
    #             new_sigma = np.dot(gamma[l, :], np.square(self.observations - B[l][1]))
    #             B[l, :] = [new_mu, new_sigma] / denominator
    #
    #
    #     return {"a": A, "b": B}

    def eexp(self, x):
        if x == None:
            return 0
        else:
            return np.exp(x)

    def eln(self,x):
        try:
            if x == 0:
                return None
            elif x > 0:
                return np.log(x)
        except ValueError:
            print("negative input error")
            raise ValueError


    def elnsum(self, x, y):
        if x== None or y == None:
            if x == None:
                return y
            else:
                return x

        else:
            if x > y:
                return x + self.eln(1+np.exp(y - x))
            else:
                return y + self.eln(1+np.exp(x - y))

    def elnproduct(self, x, y):
        if x == None or y == None:
            return None
        else:
            return x + y

    def forward_robust(self, A, B, initial):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        alpha = np.zeros((num_observed, num_states))  # store values for previous alphas
        for i in range(num_states):
            alpha[0, i] = self.elnproduct(self.eln(initial[i]), self.eln(normal_pdf(self.observations[0], B[i, 0], B[i, 1])))

        for t in range(1, num_observed):
            for j in range(num_states):
                logalpha = None
                for i in range(num_states):
                    logalpha = self.elnsum(logalpha, self.elnproduct(alpha[t-1, i], self.eln(A[i, j])))
                alpha[t, j] = self.elnproduct(logalpha, self.eln(normal_pdf(self.observations[t], B[j, 0], B[j, 1])))

        self.alpha = alpha
        return alpha

    def backward_robust(self, A, B):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        beta = np.zeros((num_observed, num_states))

        beta[self.observations.shape[0] - 1] = np.zeros(num_states)
        for t in range(num_observed - 2, -1, -1):
            for i in range(num_states):
                logbeta = None
                for j in range(num_states):
                    logbeta = self.elnsum(logbeta, self.elnproduct(self.eln(A[j, i]),
                                                                   self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])), beta[t+1, j])))
                beta[t, i] = logbeta


        self.beta = beta
        return beta


    def gamma_robust(self, A):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]

        gamma = np.zeros((num_observed, num_states))
        for t in range(num_observed):
            normalizer = None
            for i in range(num_states):
                gamma[t, i] = self.elnproduct(self.alpha[t, i], self.beta[t, i])
                normalizer = self.elnsum(normalizer, gamma[t,i])

            for i in range(num_observed):
                gamma[t, i] = self.elnproduct(gamma[t, i], -normalizer)

        return gamma


    def xi_robust(self, A, B, initial):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]

        xi = np.zeros((num_states, num_states, num_observed-1))
        for t in range(num_observed-1):
            normalizer = None
            for i in range(num_states):
                for j in range(num_states):
                    xi[j, i, t] = self.elnproduct(self.alpha[t, i], self.elnproduct(self.eln(A[j, i]),
                                                                                    self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])),
                                                                                                    self.beta[t+1, j])))
                    normalizer = self.elnsum(normalizer, xi[j, i, t])
            for i in range(num_states):
                for j in range(num_states):
                    xi[j, i, t] = self.elnproduct(xi[j, i, t], -normalizer)

        return xi

    def baum_welch_alternative(self, A, B, initial):

        num_states = np.shape(A)[0]
        T = len(self.observations)
        converge = False
        loglik_prev = - np.inf

        while not converge:
            alpha = self.forward_continuous(A, B, initial)
            beta = self.backward_continuous(A, B)

            gamma = np.zeros((T, num_states))
            for i in range(num_states):
                gamma[i] = alpha[i] * beta[i] / sum(alpha[i] * beta[i])

            xi = np.zeros((num_states, num_states, T-1))
            for i in range(T-1):
                summation = 0
                for j in range(num_states):
                    for k in range(num_states):
                        xi[k, j, i] = alpha[i, j] * beta[i+1, k] * A[k, j] * normal_pdf(self.observations[i+1], B[k, 0], B[k, 1])
                        summation += xi[k, j, i]
                xi[:, :, i] = xi[:, :, i] / summation

            initial = gamma[1]

            for i in range(num_states):
                for j in range(num_states):
                    A[i, j] = np.sum(xi, axis = 0)[i, j] / np.sum(np.sum(xi, axis = 0), axis = 0)

            for i in range(num_states):
                B[i, 0] = gamma[:, i] @ self.observations / np.sum(gamma[:, i])
                B[i, 1] = gamma[:, i] @ (self.observations - B[i, 0])**2 / np.sum(gamma[:, i])

            loglik_new = initial @ np.log(initial)

            for i in range(T-1):
                for j in range(num_states):
                    for k in range(num_states):
                        loglik_new += xi[j, k, i] * np.log(A[j, k])

            for i in range(len(self.observations)):
                for j in range(num_states):
                    loglik_new += gamma[i, j] * np.log(normal_pdf(self.observations[i], B[j, 0], B[j, 1]))

            if (np.abs(loglik_new - loglik_prev) < 1e-6):
                converge = True
            else:
                loglik_prev = loglik_new

        return A, B


    def viterbi(self, pi, transition, emission, obs):
        hidden = np.shape(emission)[0]
        d = np.shape(obs)[0]

        # init blank path
        path = np.zeros(d, dtype = int)
        #  highest probability of any path that reaches state i
        prob = np.zeros((hidden, d))
        # the state with the highest probability
        state = np.zeros((hidden, d))

         # init delta and phi
        prob[:, 0] = pi * emission[:, obs[0]]
        state[:, 0] = 0

        for i in range(1, d, 1):
            for j in range(hidden):

                prob[j,i] = np.max(prob[:, i-1] * transition[:, j] * emission[j, obs[i]])
                state[j,i] = np.argmax(prob[:, i-1] * transition[:, j] * emission[j, obs[i]])

        path[d-1] = np.argmax(prob[:, d-1])
        for i in range(d-2, -1, -1):
            path[i] = state[path[i+1], [i+1]]

        return path, prob, state

    def viterbi_continuous(self, pi, transition, emission, obs):
        hidden = np.shape(emission)[0]
        d = np.shape(obs)[0]

        # init blank path
        path = np.zeros(d, dtype = int)
        #  highest probability of any path that reaches state i
        prob = np.zeros((hidden, d))
        # the state with the highest probability
        state = np.zeros((hidden, d))

         # init delta and phi
        prob[:, 0] = pi * emission[:, obs[0]]
        state[:, 0] = 0

        for i in range(1, d, 1):
            for j in range(hidden):

                prob[j,i] = np.max(prob[:, i-1] * transition[:, j] * normal_pdf(obs[d], emission[j, 0], emission[j, 1]))
                state[j,i] = np.argmax(prob[:, i-1] * transition[:, j] * normal_pdf(obs[d], emission[j, 0], emission[j, 1]))

        path[d-1] = np.argmax(prob[:, d-1])
        for i in range(d-2, -1, -1):
            path[i] = state[path[i+1], [i+1]]

        return path, prob, state

if __name__ == '__main__':

    # simulate = SimulateData()
    # observations, state_path, A, B, initial = simulate.simulate_data(continuous=True)
    # initial = [0.5, 0.5]
    # HMM = MaxLikeHMM(observations)
    # res = HMM.baum_welch_continuous(A, B, initial)
    # res

    pass