from SimulateData import SimulateData
from MaxLikeHMM import MaxLikeHMM
from BayesianHMM import BayesianHMM
import numpy as np

class crossValidation:

    def __init__(self, data = SimulateData(), num_states = 6):
        self.data = data
        self.num_states = num_states

    def training_rate(self, Maxlike, Bayesian):

        self.obs, self.state, _, _, _ = self.data.simulate_continuous()
        model = MaxLikeHMM(self.obs[:1000])

        if Maxlike == True:
            tran_matrix = np.ones((self.num_states, self.num_states))
            model.tran_matrix = np.ones((6,6))
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

            initial = np.ones(6)
            for i in range(len(self.obs[:1000])):
                if self.obs[i] >= np.min(self.obs) and self.obs[i] < -1.5:
                    initial[0] += 1
                elif self.obs[i] >= 1.5 and self.obs[i] < 5.5:
                    initial[1] += 1
                elif self.obs[i] >= 5.5 and self.obs[i] < 12.5:
                    initial[2] += 1
                elif self.obs[i] >= 12.5 and self.obs[i] < 19.5:
                    initial[3] += 1
                elif self.obs[i] >= 19.5 and self.obs[i] < 26.5:
                    initial[4] += 1
                elif self.obs[i] >= 26.5 and self.obs[i] <= np.max(self.obs):
                    initial[5] += 1
            initial = initial / 1000

            sim_tran, sim_emis, sim_init = model.baum_welch_robust(tran_matrix, emis_matrix, initial)
            path, _, _ = model.viterbi_robust(sim_init, sim_tran, sim_emis)
            rate = np.sum(path == self.state[:1000])/1000

            self.sim_tran = sim_tran
            self.sim_emis = sim_emis
            self.sim_init = sim_init

            return rate

        elif Bayesian == True:

            self.observations, self.state_path, A, B, initial = self.data.simulate_continuous()
            HMM = BayesianHMM(observations=self.observations[:1000], state_path=self.state_path, num_states = self.num_states)
            HMM.generate_priors()

            num_iter = int(1e3)
            HMM.sample_parameters(num_iter=num_iter, num_burnin=int(1e2))
            path = HMM.sample_states()
            rate = np.sum(path == self.state_path[:1000])/1000
            return rate


    def test_rate(self, Maxlike, BayesianLike):

        if Maxlike == True:
            model_test = MaxLikeHMM(observations=self.obs[1000:])
            test_path, _, _ = model_test.viterbi_robust(self.sim_init, self.sim_tran, self.sim_emis)
            rate = np.sum(test_path == self.state[1000:])/200 # 187  157
            return rate

        elif BayesianLike == True:
            model_test = BayesianHMM(observations=self.observations[1000:])
            test_path = model_test.sample_states()
            rate = np.sum(test_path == self.state_path[:1000])/200
            return rate