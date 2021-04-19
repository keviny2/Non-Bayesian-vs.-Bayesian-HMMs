import numpy as np

from distribution import normal_pdf
from hmm import HMM

class MaxLikeHMM(HMM):

    def __init__(self, observations = None, num_states = 6):

        self.observations = observations
        self.num_states = num_states
        self.num_observations = len(self.observations)

    def initial_parameters(self):

        tran_matrix = np.ones((self.num_states, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_states):
                if i == j:
                    tran_matrix[i, j] *= 0.8
                else:
                    tran_matrix[i, j] *= 0.04

        emis_matrix = np.array([[-5, 7],
                                [2, 7],
                                [9, 7],
                                [16, 7],
                                [23, 7],
                                [30, 7]])

        initial = np.ones(self.num_states)
        for i in range(self.num_observations):
            if self.observations[i] >= np.min(self.observations) and self.observations[i] < -1.5:
                initial[0] += 1
            elif self.observations[i] >= 1.5 and self.observations[i] < 5.5:
                initial[1] += 1
            elif self.observations[i] >= 5.5 and self.observations[i] < 12.5:
                initial[2] += 1
            elif self.observations[i] >= 12.5 and self.observations[i] < 19.5:
                initial[3] += 1
            elif self.observations[i] >= 19.5 and self.observations[i] < 26.5:
                initial[4] += 1
            elif self.observations[i] >= 26.5 and self.observations[i] <= np.max(self.observations):
                initial[5] += 1
        initial = initial / 1000

        return tran_matrix, emis_matrix, initial


    def forward_robust(self, A, B, initial):

        alpha = np.zeros((self.num_observations, self.num_states))  # store values for previous alphas
        for i in range(self.num_states):
            alpha[0, i] = self.elnproduct(self.eln(initial[i]), self.eln(normal_pdf(self.observations[0], B[i, 0], B[i, 1])))

        for t in range(1, self.num_observations):
            for j in range(self.num_states):
                logalpha = None
                for i in range(self.num_states):
                    logalpha = self.elnsum(logalpha, self.elnproduct(alpha[t-1, i], self.eln(A[i, j])))
                alpha[t, j] = self.elnproduct(logalpha, self.eln(normal_pdf(self.observations[t], B[j, 0], B[j, 1])))

        return alpha


    def backward_robust(self, A, B):

        beta = np.zeros((self.num_observations, self.num_states))

        for t in range(self.num_observations - 2, -1, -1):
            for i in range(self.num_states):
                logbeta = None
                for j in range(self.num_states):

                    logbeta = self.elnsum(logbeta, self.elnproduct(self.eln(A[i, j]),
                                                                   self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])), beta[t+1, j])))
                beta[t, i] = logbeta
    
        return beta


    def gamma_robust(self, alpha, beta):

        gamma = np.zeros((self.num_observations, self.num_states))
        for t in range(self.num_observations):
            normalizer = None
            for i in range(self.num_states):

                gamma[t, i] = self.elnproduct(alpha[t, i],beta[t, i])

                normalizer = self.elnsum(normalizer, gamma[t,i])

            for i in range(self.num_states):
                gamma[t, i] = self.elnproduct(gamma[t, i], -normalizer)

        return gamma


    def xi_robust(self, A, B, alpha, beta):

        xi = np.zeros((self.num_observations-1, self.num_states, self.num_states))
        for t in range(self.num_observations-1):
            normalizer = None
            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[t, i, j] = self.elnproduct(alpha[t, i], self.elnproduct(self.eln(A[i, j]),
                                                                                    self.elnproduct(self.eln(normal_pdf(self.observations[t+1], B[j, 0], B[j, 1])),
                                                                                                    beta[t+1, j])))
                    normalizer = self.elnsum(normalizer, xi[t, i, j])

            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[t, i, j] = self.elnproduct(xi[t, i, j], -normalizer)
        return xi

    def baum_welch_robust(self, A, B, initial):

        converge = False
        loglik_prev = - np.inf

        print("=" * 20, 'Beginning Baum-Welch!', '=' * 20)
        iter = 1
        while not converge:
            alpha = self.forward_robust(A, B, initial)
            beta = self.backward_robust(A, B)

            gamma = self.gamma_robust(alpha, beta)
            xi = self.xi_robust(A, B, alpha, beta)


            for i in range(self.num_states):
                initial[i] = self.eexp(gamma[0, i])


            for i in range(self.num_states):
                for j in range(self.num_states):
                    numerator = None
                    denominator = None
                    for t in range(self.num_observations - 1):
                        numerator = self.elnsum(numerator, xi[t, i, j])
                        denominator = self.elnsum(denominator, gamma[t, i])
                    A[i, j] = self.eexp(self.elnproduct(numerator, -denominator))

            for i in range(self.num_states):
                numerator = None
                denominator = None
                for t in range(self.num_observations):
                    numerator = self.elnsum(numerator, self.elnproduct(gamma[t, i], self.eln(self.observations[t])))
                    denominator = self.elnsum(denominator, gamma[t, i])
                B[i, 0] = self.eexp(self.elnproduct(numerator, -denominator))


                numerator = None
                for t in range(self.num_observations):
                    numerator = self.elnsum(numerator, self.elnproduct(gamma[t, i], self.eln((self.observations[t] - B[i, 0])**2)))
                B[i, 1] = np.sqrt(self.eexp(self.elnproduct(numerator, -denominator)))

            loglik_new = 0
            for i in range(self.num_states):
                loglik_new += initial[i] * np.log(normal_pdf(self.observations[0], B[i, 0], B[i, 1]))

            for i in range(self.num_observations - 1):
                for j in range(self.num_states):
                    for k in range(self.num_states):
                        loglik_new += self.eexp(xi[i, j, k]) * np.log(A[j, k])

            for i in range(self.num_observations):
                for j in range(self.num_states):
                    loglik_new += self.eexp(gamma[i, j]) * np.log(normal_pdf(self.observations[i], B[j, 0], B[j, 1]))

            if (np.abs(loglik_new - loglik_prev) < 1e-4 or loglik_new < loglik_prev):
                converge = True
            else:
                loglik_prev = loglik_new

            print('Iteration:', iter)
            print('Log-likelihood:', loglik_new)
            iter += 1

        return A, B, initial
