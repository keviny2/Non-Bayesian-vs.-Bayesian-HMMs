import numpy as np
from numba_functions import simulate_observations

class SimulateData:

    def __init__(self):
        pass


    def generate_random_state_transition_matrix(self, nrow, ncol):
        """
        simulate random transition matrix
        :param nrow:
        :param ncol:
        :return:
        """
        x = np.random.random((nrow, ncol))

        rsum = None
        csum = None

        while (np.any(rsum != 1)) | (np.any(csum != 1)):
            x /= x.sum(0)
            x = x / x.sum(1)[:, np.newaxis]
            rsum = x.sum(1)
            csum = x.sum(0)

        return x

    def generate_random_emission_matrix(self, nrow, ncol):
        """
        simulate random emission parameter matrix
        :param nrow:
        :param ncol:
        :return:
        """
        initial = np.random.normal(1, 0.5)
        emission_matrix = np.zeros((nrow, ncol))
        for row in range(nrow):
            emission_matrix[row, :] = [initial*(row+1), 0.5]

        return emission_matrix


    def generate_random_state_path(self, num_states, num_obs):
        """
        simulate random state path
        :param num_states:
        :param num_obs:
        :return:
        """
        return np.random.choice(np.arange(num_states), num_obs)

    def generate_num(self):
        return np.random.uniform(0, 1)

    def generate_state(self, v, num):
        for i in range(6):
            if num < sum(v[:i+1]):
                a = i
                return a

    def generate_obs(self, state):
        if state == 0:
            a = np.random.normal(0, 2)
        elif state == 1:
            a = np.random.normal(5, 2)
        elif state == 2:
            a = np.random.normal(10, 2)
        elif state == 3:
            a = np.random.normal(15, 2)
        elif state == 4:
            a = np.random.normal(20, 2)
        else:
            a = np.random.normal(25, 2)
        return a


    def simulate_continuous(self, num_obs=1200):

        """
        sherry's process to simulate data
        :param num_obs:
        :return:
        """
        A = np.array(np.array([[0.8, 0.04, 0.05, 0.04, 0.03, 0.04],
                               [0.03, 0.85, 0.03, 0.04, 0.02, 0.03],
                               [0.02, 0.02, 0.9, 0.02, 0.02, 0.02],
                               [0.04, 0.03, 0.09, 0.75, 0.04, 0.05],
                               [0.02, 0.04, 0.03, 0.02, 0.87, 0.02],
                               [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]]))

        B = np.array(np.array([[0, 2],
                              [5, 2],
                              [10, 2],
                              [15, 2],
                              [20, 2],
                              [25, 2]]))

        converge = False
        init = np.ones(A.shape[0]) / A.shape[0]
        while not converge:
            update = self.marginal(A, init)
            if ((update != init).all()):
                init = update
            else:
                converge = True
        print(init)

        state = np.zeros(num_obs)
        obs = np.zeros(num_obs)
        state[0] = self.generate_state(init, self.generate_num())
        obs[0] = self.generate_obs(state[0])

        for i in np.arange(1, num_obs):
            tran = A[int(state[i - 1])]
            state[i] = self.generate_state(tran, self.generate_num())
            obs[i] = self.generate_obs(state[i])

        return obs, state, A, B, init


    def marginal(self, A, init):
        return np.dot(A.T, init)
