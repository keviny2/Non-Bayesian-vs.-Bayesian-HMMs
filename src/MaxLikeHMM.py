import numpy as np
from Distribution import normal_pdf
from SimulateData import SimulateData
import matplotlib.pyplot as plt
import os

class MaxLikeHMM:

    def __init__(self, observations=None):
        """

        :param observations: vector of observations
        """

        # TODO: clean up this file?
        self.observations = observations

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

            if (np.abs(loglik_new - loglik_prev) < 1e-4 or loglik_new < loglik_prev):
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
                list = np.zeros(6)
                for k in range(num_states):
                    list[k] = self.elnproduct(prob[i-1, k], self.elnproduct(self.eln(A[k, j]),
                                                                                      self.eln(normal_pdf(self.observations[i], B[j, 0], B[j, 1]))))

                prob[i, j] = np.max(list)
                state[i, j] = np.argmax(list)


        path[T-1] = np.argmax(prob[T-1])
        for i in range(T-2, -1, -1):
            path[i] = state[i+1, path[i+1]]

        return path, prob, state

#if __name__ == '__main__':

    # data = SimulateData()
    # obs, state, A, B, init = data.simulate_continuous()
    #
    # plt.figure()
    # plt.plot(obs[:1000])
    # plt.xlabel("Index")
    # plt.ylabel("Simulated Observations")
    # fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "original_observations")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)
    #
    # plt.figure()
    # plt.plot(state[:1000])
    # plt.xlabel("Index")
    # plt.ylabel("Simulated Hidden States")
    # fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "original_states")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)
    #
    # model = MaxLikeHMM(observations=obs[:1000])
    # tran_matrix = np.ones((6,6))
    # for i in range(6):
    #     for j in range(6):
    #         if i == j:
    #             tran_matrix[i, j] *= 0.8
    #         else:
    #             tran_matrix[i, j] *= 0.04
    #
    # emis_matrix = np.array([[-5, 7],
    #                         [2, 7],
    #                         [9, 7],
    #                         [16, 7],
    #                         [23, 7],
    #                         [30, 7]])
    #
    # initial = np.ones(6)
    # for i in range(len(obs[:1000])):
    #     if obs[i] >= np.min(obs) and obs[i] < -1.5:
    #         initial[0] += 1
    #     elif obs[i] >= 1.5 and obs[i] < 5.5:
    #         initial[1] += 1
    #     elif obs[i] >= 5.5 and obs[i] < 12.5:
    #         initial[2] += 1
    #     elif obs[i] >= 12.5 and obs[i] < 19.5:
    #         initial[3] += 1
    #     elif obs[i] >= 19.5 and obs[i] < 26.5:
    #         initial[4] += 1
    #     elif obs[i] >= 26.5 and obs[i] <= np.max(obs):
    #         initial[5] += 1
    # initial = initial / 1000
    #
    # sim_tran, sim_emis, init = model.baum_welch_robust(tran_matrix, emis_matrix, initial)
    # path, _, _ = model.viterbi_robust(init, sim_tran, sim_emis)
    # print(path)
    # print(state[:1000])
    # print(sum(path == state[:1000]))  #915  747
    #
    #
    # plt.figure()
    # plt.plot(path)
    # plt.xlabel("Index")
    # plt.ylabel("Estimated Hidden States")
    # fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "viterbi_path")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)
    #
    # # sim_tran = np.array([[0.76331548, 0.06240083, 0.04594133, 0.0815152 , 0.00626544,
    # #     0.04056173],
    # #    [0.04275861, 0.76968643, 0.0809278 , 0.01683517, 0.03937972,
    # #     0.05041227],
    # #    [0.02533083, 0.01630721, 0.90764861, 0.03607666, 0.00469835,
    # #     0.00993834],
    # #    [0.05453972, 0.01527269, 0.03202524, 0.78106736, 0.06948663,
    # #     0.04760835],
    # #    [0.0096078 , 0.03852523, 0.03776866, 0.0301672 , 0.85254404,
    # #     0.03138707],
    # #    [0.01064084, 0.0309499 , 0.00764781, 0.00965317, 0.00979895,
    # #     0.93130932]])
    # # sim_emis = np.array([[ 0,  1],
    # #    [ 4,  2],
    # #    [ 9,  2],
    # #    [13,  2],
    # #    [20,  2],
    # #    [25,  2]])
    # #
    # # init = np.array([3.46758931e-19, 6.22713998e-03, 9.77398372e-01, 1.63744876e-02,
    # #    5.31935101e-38, 2.64969723e-86])
    #
    # model_test = MaxLikeHMM(observations=obs[1000:])
    # test_path, _, _ = model_test.viterbi_robust(init, sim_tran, sim_emis)
    # print(test_path)
    # print(state[1000:])
    # print(sum(test_path == state[1000:]))  #187  157
    #
    # plt.figure()
    # plt.plot(state[1000:])
    # plt.xlabel("Index")
    # plt.ylabel("Estimated Hidden States")
    # fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "test_state")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)
    #
    # plt.figure()
    # plt.plot(test_path)
    # plt.xlabel("Index")
    # plt.ylabel("Estimated Hidden States")
    # fname = os.path.join("/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project", "plots", "viterbi_test_path")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)
    #


