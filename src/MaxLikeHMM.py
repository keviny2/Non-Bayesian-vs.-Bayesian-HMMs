import numpy as np
from Distribution import normal_pdf, normal_log_pdf
from SimulateData import SimulateData

class MaxLikeHMM:

    def __init__(self, observations=None):
        """

        :param observations: vector of observations
        """

        # TODO: clean up this file?
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

    # TODO: implement Baum Welch with robust forward backward implementations?
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
        """

        :param x: x
        :return: exp(x)
        """
        if x == None:
            return 0
        else:
            return np.exp(x)


    def eln(self, x):
        """

        :param x: x
        :return: ln(x)
        """
        try:
            if x == 0:
                return None
            elif x > 0:
                return np.log(x)
        except ValueError:
            print("negative input error")
            raise ValueError


    def elnsum(self, x, y):  # ln(x+y)
        if x == None or y == None:
            if x == None:
                return y
            else:
                return x

        else:
            if x > y:
                return x + self.eln(1 + np.exp(y - x))
            else:
                return y + self.eln(1 + np.exp(x - y))


    def elnproduct(self, x, y):  # ln(x) + ln(y)
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

        return alpha


    def backward_robust(self, A, B):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]
        beta = np.zeros((num_observed, num_states))

        for t in range(num_observed - 2, -1, -1):
            for i in range(num_states):
                logbeta = None
                for j in range(num_states):

                    logbeta = self.elnsum(logbeta, self.elnproduct(self.eln(A[i, j]),
                                                                   self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])), beta[t+1, j])))
                beta[t, i] = logbeta
    
        return beta


    def gamma_robust(self, A, alpha, beta):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]

        gamma = np.zeros((num_observed, num_states))
        for t in range(num_observed):
            normalizer = None
            for i in range(num_states):

                gamma[t, i] = self.elnproduct(alpha[t, i],beta[t, i])

                normalizer = self.elnsum(normalizer, gamma[t,i])

            for i in range(num_states):
                gamma[t, i] = self.elnproduct(gamma[t, i], -normalizer)

        print(gamma)

        return gamma


    def xi_robust(self, A, B, alpha, beta):
        num_states = A.shape[0]
        num_observed = self.observations.shape[0]

        xi = np.zeros((num_observed-1, num_states, num_states))

        for t in range(num_observed-1):
            normalizer = None
            for i in range(num_states):
                for j in range(num_states):
                    xi[t, i, j] = self.elnproduct(alpha[t, i], self.elnproduct(self.eln(A[i, j]),
                                                                                    self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])),
                                                                                                    beta[t+1, j])))
                    normalizer = self.elnsum(normalizer, xi[t, i, j])

            for i in range(num_states):
                for j in range(num_states):
                    xi[t, i, j] = self.elnproduct(xi[t, i, j], -normalizer)
        return xi

    def baum_welch_robust(self, A, B, initial):
        num_states = np.shape(A)[0]
        T = len(self.observations)
        converge = False
        loglik_prev = - np.inf

        while not converge:
            alpha = self.forward_robust(A, B, initial)
            beta = self.backward_robust(A, B)

            gamma = self.gamma_robust(A, alpha, beta)
            xi = self.xi_robust(A, B, alpha, beta)


            for i in range(num_states):
                initial[i] = self.eexp(gamma[0, i])


            for i in range(num_states):
                for j in range(num_states):
                    numerator = None
                    denominator = None
                    for t in range(T-1):
                        numerator = self.elnsum(numerator, xi[t, i, j])
                        denominator = self.elnsum(denominator, gamma[t, i])
                    A[i, j] = self.eexp(self.elnproduct(numerator, -denominator))

            for i in range(num_states):
                numerator = None
                denominator = None
                for t in range(T):
                    numerator = self.elnsum(numerator, self.elnproduct(gamma[t, i], self.eln(self.observations[t])))
                    denominator = self.elnsum(denominator, gamma[t, i])
                B[i, 0] = self.eexp(self.elnproduct(numerator, -denominator))


                numerator = None
                for t in range(T):
                    numerator = self.elnsum(numerator, self.elnproduct(gamma[t, i], self.eln((self.observations[t] - B[i, 0])**2)))
                B[i, 1] = np.sqrt(self.eexp(self.elnproduct(numerator, -denominator)))

            loglik_new = 0
            for i in range(num_states):
                loglik_new += initial[i] * np.log(normal_pdf(self.observations[0], B[i, 0], B[i, 1]))

            for i in range(T - 1):
                for j in range(num_states):
                    for k in range(num_states):
                        loglik_new += self.eexp(xi[i, j, k]) * np.log(A[j, k])

            for i in range(len(self.observations)):
                for j in range(num_states):
                    loglik_new += self.eexp(gamma[i, j]) * np.log(normal_pdf(self.observations[i], B[j, 0], B[j, 1]))

            if (np.abs(loglik_new - loglik_prev) < 1e-4):
                converge = True
            else:
                loglik_prev = loglik_new

            print(A)
            print(B)
            print(initial)
            print(loglik_new)

        return A, B, initial

    def viterbi_robust(self, initial, A, B):
        """

        :param initial:
        :param A:
        :param B:
        :return:
        """
        num_states = np.shape(A)[0]
        T = len(self.observations)
        path = np.zeros(T, dtype = int)
        prob = np.zeros((T, num_states))
        state = np.zeros((T, num_states))

        for i in range(num_states):
            prob[0, i] = self.elnproduct(self.eln(initial[i]), self.eln(normal_pdf(self.observations[0], B[i, 0], B[i, 1])))
        state[0] = 0

        for i in range(1, T):
            for j in range(num_states):
                list = np.zeros(3)
                for k in range(num_states):
                    list[k] = self.elnproduct(prob[i-1, k], self.elnproduct(self.eln(A[k, j]),
                                                                                      self.eln(normal_pdf(self.observations[i], B[j, 0], B[j, 1]))))

                prob[i, j] = np.max(list)
                state[i, j] = np.argmax(list)


        path[T-1] = np.argmax(prob[T-1])
        for i in range(T-2, -1, -1):
            path[i] = state[i+1, path[i+1]]

        return path, prob, state

if __name__ == '__main__':

    # simulate = SimulateData()
    # observations, state_path, A, B, initial = simulate.simulate_data(continuous=True)
    # initial = [0.5, 0.5]
    # HMM = MaxLikeHMM(observations)
    # res = HMM.baum_welch_continuous(A, B, initial)
    # res

    pass