import numpy as np
import pandas as pd

class SimulateData:

    def __init__(self):
        pass

    def simulate_data(self, state_transition=None, emission_prob=None, initial_state=None, num_obs=100, continuous=True):

        if continuous:
            return self.simulate_continuous(state_transition, emission_prob, initial_state, num_obs)

        else:
            return self.simulate_discrete()


    def simulate_continuous(self, state_transition, emission_prob, initial_state, num_obs):
        if state_transition is None:
            state_transition = np.array([[0.3, 0.7],[0.6, 0.4]])

        if emission_prob is None:
            emission_prob = np.array([[1, 0.5],[2, 0.5]])

        if initial_state is None:
            initial_state = np.array([1, 0, 0])

        observations = np.zeros(num_obs)
        curr_state = np.argmax(np.random.multinomial(1, initial_state, 1))
        for i in range(num_obs):
            observations[i] = np.random.normal(emission_prob[curr_state, 0], emission_prob[curr_state, 1])
            curr_state = np.argmax(np.random.multinomial(1, state_transition[curr_state, :]))

        # generate random initialization
        A = self.generate_state_transition_matrix(state_transition.shape[0], state_transition.shape[1])
        B = self.generate_emision_matrix(emission_prob.shape[0], emission_prob.shape[1])
        initial = np.zeros(state_transition.shape[0])
        initial[0] = 1
        return observations, A, B, initial

    def simulate_discrete(self):

        # load data and create HMM object
        obs = pd.read_csv('../data/dummy_data.csv')['Visible'].values + 1

        pi = np.array([.5, .5])  # initial dist.
        # Transition Probabilities   {A,B} are the states
        # A = [[p(A|A), p(B|A)],
        #      [p(A|B), p(B|B)]]
        A = np.ones((2, 2))
        A = A / np.sum(A, axis=1)

        # Emission Probabilities    {1,2,3} are the emissions
        # B = [[p(1|A), p(2|A), p(3|A)],
        #      [p(1|B), p(2|B), p(3|B)]]
        B = np.array(((1, 3, 5), (2, 4, 6)))
        B = B / np.sum(B, axis=1).reshape((-1, 1))

        return obs, A, B, pi

    def generate_state_transition_matrix(self, nrow, ncol):
        x = np.random.random((nrow, ncol))

        rsum = None
        csum = None

        while (np.any(rsum != 1)) | (np.any(csum != 1)):
            x /= x.sum(0)
            x = x / x.sum(1)[:, np.newaxis]
            rsum = x.sum(1)
            csum = x.sum(0)

        return x

    def generate_emision_matrix(self, nrow, ncol):
        initial = np.random.normal(1, 0.5)
        emission_matrix = np.zeros((nrow, ncol))
        for row in range(nrow):
            emission_matrix[row, :] = [initial*(row+1), 0.5]

        return emission_matrix