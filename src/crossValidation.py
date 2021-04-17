from SimulateData import SimulateData
from MaxLikeHMM import MaxLikeHMM
from BayesianHMM import BayesianHMM
import numpy as np
from Plot import plot

class crossValidation:

    def __init__(self, data = SimulateData(), num_states = 6):
        self.data = data
        self.num_states = num_states

    def training_rate(self, Maxlike, Bayesian):

        if Maxlike == True:

            self.obs, self.state, _, _, _ = self.data.simulate_continuous(num_obs=1200)
            self.train_obs = self.obs[:1000]
            self.test_obs = self.obs[1000:]
            self.train_state = self.state[:1000]
            self.test_state = self.state[1000:]

            self.model = MaxLikeHMM(observations = self.train_obs)
            tran_matrix, emis_matrix, initial = self.model.initial_parameters()

            sim_tran, sim_emis, sim_init = self.model.baum_welch_robust(tran_matrix, emis_matrix, initial)
            path, _, _ = self.model.viterbi_robust(self.train_obs, sim_init, sim_tran, sim_emis)
            rate = np.sum(path == self.train_state)/len(self.train_state)

            self.sim_tran = sim_tran
            self.sim_emis = sim_emis
            self.sim_init = sim_init

            plot(self.train_obs, ylabel = "Simulated Observations", name = "Max_Original_Observations")
            plot(self.train_state, ylabel = "Simulated Hidden States", name = "Max_Original_States")
            plot(path, ylabel = "Estimated Hidden States", name = "Max_Viterbi_Path")

            return rate

        elif Bayesian == True:

            self.observations, self.state_path, A, B, initial = self.data.simulate_continuous(num_obs = 1.2e4)
            self.train_obs = self.obs[:1e4]
            self.test_obs = self.obs[1e4:]
            self.train_state = self.state[:1e4]
            self.test_state = self.state[1e4:]

            HMM = BayesianHMM(observations=self.train_obs, state_path=self.train_state, num_states = self.num_states)
            HMM.generate_priors()

            num_iter = int(1e3)
            HMM.sample_parameters(num_iter=num_iter, num_burnin=int(1e2))
            path = HMM.sample_states()
            rate = np.sum(path == self.train_state)/len(self.train_state)

            return rate


    def test_rate(self, Maxlike, BayesianLike):

        if Maxlike == True:

            test_path, _, _ = self.model.viterbi_robust(self.test_obs,self.sim_init, self.sim_tran, self.sim_emis)
            rate = np.sum(test_path == self.state[1000:])/200 # 187  157

            plot(self.test_state, ylabel="Simulated Hidden States", name="Max_Test_States")
            plot(test_path, ylabel="Estimated Hidden States", name="Max_Viterbi_Test_Path")

            return rate

        elif BayesianLike == True:
            model_test = BayesianHMM(observations=self.observations[1e4:])
            test_path = model_test.sample_states()
            rate = np.sum(test_path == self.state_path[:1e4])/2e3
            return rate