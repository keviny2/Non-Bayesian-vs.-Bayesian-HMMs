import numpy as np
from Distribution import normal_pdf
from SimulateData import SimulateData
import matplotlib.pyplot as plt
import os

class MaxLikeHMM:

    def __init__(self, observations = None, num_states = 6):

        self.observations = observations
        self.num_states = num_states
        self.num_observations = len(self.observations)

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


        return A, B, initial

    def viterbi_robust(self, data, initial, A, B):
        nrow = len(data)

        path = np.zeros(nrow, dtype = int)
        prob = np.zeros((nrow, self.num_states))
        state = np.zeros((nrow, self.num_states))

        for i in range(self.num_states):
            prob[0, i] = self.elnproduct(self.eln(initial[i]), self.eln(normal_pdf(data[0], B[i, 0], B[i, 1])))
        state[0] = 0

        for i in range(1, nrow):
            for j in range(self.num_states):
                list = np.zeros(6)
                for k in range(self.num_states):
                    list[k] = self.elnproduct(prob[i-1, k], self.elnproduct(self.eln(A[k, j]),
                                                                                      self.eln(normal_pdf(data[i], B[j, 0], B[j, 1]))))

                prob[i, j] = np.max(list)
                state[i, j] = np.argmax(list)


        path[nrow - 1] = np.argmax(prob[nrow - 1])
        for i in range(nrow - 2, -1, -1):
            path[i] = state[i+1, path[i+1]]

        return path, prob, state

    def plot(self, data, ylabel, name):
        plt.figure()
        plt.plot(data)
        plt.xlabel("Index")
        plt.ylabel(ylabel)
        fname = os.path.join("..", "plots", name)
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
