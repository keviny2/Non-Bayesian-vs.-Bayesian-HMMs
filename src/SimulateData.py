import numpy as np
from numba_functions import simulate_observations

class SimulateData:

    def __init__(self):
        pass

    def simulate_data(self, state_transition=None, emission_prob=None, initial_state=None, num_obs=100, sherry=False):
        """
        simulate data for inference

        :param state_transition: stochastic matrix of transition probabilities
        :param emission_prob: matrix holding parameters of normal distribution corresponding to each state
        :param initial_state: np array of initial state probabilities
        :param num_obs: integer number representing number of observations
        :param sherry: boolean indicating which simulation procedure to perform
        (sherry==TRUE corresponds to Sherry's procedure from CS540)

        :return: simulated observations, state path, state transition matrix, emission parameters, initial distribution
        """
        if sherry:
            return self.simulate_continuous_sherry(num_obs)
        else:
            return self.simulate_continuous(state_transition, emission_prob, initial_state, num_obs)


    def simulate_continuous(self, state_transition, emission_prob, initial_state, num_obs):
        """
        simulate data for HMM with discrete hidden space and continuous observation space

        :param state_transition: stochastic matrix of transition probabilities
        :param emission_prob: matrix holding parameters of normal distribution corresponding to each state
        :param initial_state: np array of initial state probabilities
        :param num_obs: integer number representing number of observations
        :return:
        """
        if state_transition is None:
            state_transition = np.array([[0.8, 0.1, 0.1],
                                         [0.05, 0.85, 0.1],
                                         [0.05, 0.05, 0.9]])

        if emission_prob is None:
            # variances need to be the same
            emission_prob = np.array([[0, 0.5],
                                      [1, 0.5],
                                      [2, 0.5]])

        if initial_state is None:
            initial_state = np.array([1/3, 1/3, 1/3])

        observations = simulate_observations(num_obs, initial_state, emission_prob, state_transition)

        # generate random initialization
        A = self.generate_random_state_transition_matrix(state_transition.shape[0], state_transition.shape[1])
        B = self.generate_random_emission_matrix(emission_prob.shape[0], emission_prob.shape[1])
        state_path_sim = self.generate_random_state_path(A.shape[0], num_obs)

        initial = np.zeros(state_transition.shape[0])
        initial[0] = 1  # begin at the first state

        return observations, state_path_sim, A, B, initial

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


    def simulate_continuous_sherry(self, num_obs=1000):
        """
        sherry's process to simulate data
        :param num_obs:
        :return:
        """
        A = np.array([[0.6, 0.3, 0.1],
                      [0.1, 0.8, 0.1],
                      [0.1, 0.3, 0.6]])

        B = np.array([[-2, 1],
                     [0, 1],
                     [2, 1]])

        converge = False
        init = np.array([1 / 3, 1 / 3, 1 / 3])
        while not converge:
            update = self.marginal(A, init)
            if ((update != init).all()):
                init = update
            else:
                converge = True
        init = np.array([0.2, 0.6, 0.2])

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

    def generate_num(self):
        return np.random.uniform(0, 1)

    def generate_state(self, v, num):
        n = len(v)
        for i in range(3):
            if num < sum(v[:i+1]):
                a = i
                return a

    def generate_obs(self, state):
        if state == 0:
            a = np.random.normal(-2, 1)
        elif state == 1:
            a = np.random.normal(0, 1)
        else:
            a = np.random.normal(2, 1)
        return a