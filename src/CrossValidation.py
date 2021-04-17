from SimulateData import SimulateData
from MaxLikeHMM import MaxLikeHMM
from BayesianHMM import BayesianHMM
import numpy as np

class crossValidation:

    def __init__(self, data = SimulateData(), num_states = 6):
        self.data = data
        self.num_states = num_states

    def training_rate(self, MaxLike, BayesianLike):

        if MaxLike == True:

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

            self.model.plot(self.train_obs, ylabel = "Simulated Observations", name = "Max_Original_Observations")
            self.model.plot(self.train_state, ylabel = "Simulated Hidden States", name = "Max_Original_States")
            self.model.plot(path, ylabel = "Estimated Hidden States", name = "Max_Viterbi_Path")

            return rate

        elif BayesianLike == True:

            self.obs, self.state, A, B, initial = self.data.simulate_continuous(num_obs = 12000)
            self.train_obs = self.obs[:10000]
            self.test_obs = self.obs[10000:]
            self.train_state = self.state[:10000]
            self.test_state = self.state[10000:]

            '''
            first 10000 observations and hidden states work as training set
            remaining 2000 observations and hidden states work as test set
            '''

            self.HMM = BayesianHMM(observations=self.train_obs, state_path=self.train_state, num_states = self.num_states)
            self.HMM.generate_priors()

            self.num_iter = int(100)
            self.HMM.sample_parameters(num_iter=self.num_iter, num_burnin=int(10))

            path = self.HMM.state_path

            '''
            Does line 53 return the predicted path for training set? not pretty sure ;-;
            '''
            rate = np.sum(path == self.train_state)/len(self.train_state)

            '''
            Edit the plot_results function from BayesianHMM!! Instead of showing them, the plots are saved in plots folder
            '''
            self.HMM.plot_results(num_iter=self.num_iter, max=None, name="Bayes_Original_Observations")
            self.HMM.plot_results(num_iter=self.num_iter, max=None, name="Bayes_Original_States")
            self.HMM.plot_results(num_iter=self.num_iter, max=None,  name="Bayes_Predicted_Path")

            return rate

    def test_rate(self, MaxLike, BayesianLike):

        if MaxLike == True:

            test_path, _, _ = self.model.viterbi_robust(self.test_obs,self.sim_init, self.sim_tran, self.sim_emis)
            rate = np.sum(test_path == self.state[1000:])/200 # 187  157

            self.model.plot(self.test_state, ylabel="Simulated Hidden States", name="Max_Test_States")
            self.model.plot(test_path, ylabel="Estimated Hidden States", name="Max_Viterbi_Test_Path")

            return rate

        elif BayesianLike == True:

            '''
            find the test_path here!!
            '''
            test_path = self.HMM.state_path
            rate = np.sum(test_path == self.state_path[10000:])/2000


            self.HMM.plot_results(num_iter= self.num_iter, max=None, name="Bayes_Test_States")
            self.HMM.plot_results(num_iter= self.num_iter, max=None, name="Bayes_Predicted_Test_Path")

            return rate