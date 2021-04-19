from SimulateData import SimulateData
from MaxLikeHMM import MaxLikeHMM
from BayesianHMM import BayesianHMM
import numpy as np
import matplotlib.pyplot as plt
import os

class CrossValidation:

    def __init__(self, bayesian=False, num_states = 6):
        self.data = SimulateData()
        self.num_states = num_states

        # NOTE: keeping values you use multiple times is good for class variable
        self.bayesian = bayesian

    # NOTE: num_obs and num_test should go in init!
    def train(self, num_obs=1200, num_test=200):

        # NOTE: initializing class parameters outside of init
        obs, self.state_path = self.data.simulate_continuous(num_obs=num_obs)
        
        index = num_obs-num_test
        train_obs = obs[:index]
        self.test_obs = obs[index:]
        train_state_path = self.state_path[:index]
        self.test_state_path = self.state_path[index:]


        if self.bayesian:

            '''
            first 10000 observations and hidden states work as training set
            remaining 2000 observations and hidden states work as test set
            '''

            self.HMM = BayesianHMM(observations=train_obs, state_path=train_state_path, num_states = self.num_states)
            self.HMM.generate_priors()

            self.num_iter = int(100)
            self.HMM.sample_parameters(num_iter=self.num_iter, num_burnin=int(100))

            path = self.HMM.state_path

            '''
            Does line 53 return the predicted path for training set? not pretty sure ;-;
            '''
            rate = np.sum(path == train_state_path)/len(train_state_path)

            '''
            Edit the plot_results function from BayesianHMM!! Instead of showing them, the plots are saved in plots folder
            '''
            self.plot(train_obs, ylabel = "Simulated Observations", name = "Bayes_Original_Observations")
            self.plot(train_state_path, ylabel = "Simulated Hidden States", name = "Bayes_Original_States")
            self.plot(path, ylabel = "Estimated Hidden States", name = "Bayes_Viterbi_Path")

            return rate

        else:

            self.model = MaxLikeHMM(observations = train_obs)
            tran_matrix, emis_matrix, initial = self.model.initial_parameters()

            sim_tran, sim_emis, sim_init = self.model.baum_welch_robust(tran_matrix, emis_matrix, initial)
            path, _, _ = self.model.viterbi_robust(train_obs, sim_init, sim_tran, sim_emis)
            rate = np.sum(path == train_state_path)/len(train_state_path)

            self.sim_tran = sim_tran
            self.sim_emis = sim_emis
            self.sim_init = sim_init

            self.plot(train_obs, ylabel = "Simulated Observations", name = "Max_Original_Observations")
            self.plot(train_state_path, ylabel = "Simulated Hidden States", name = "Max_Original_States")
            self.plot(path, ylabel = "Estimated Hidden States", name = "Max_Viterbi_Path")

            return rate

    def test(self):

        if self.bayesian:

            B = [[mu, np.sqrt(1/self.HMM.sigma_invsq)] for mu in self.HMM.mu]
            test_path, _, _ = self.HMM.viterbi_robust(self.test_obs, self.HMM.initial_dist, self.HMM.A, np.array(B))
            rate = np.sum(test_path == self.state_path[1000:])/200


            self.plot(self.test_state_path, ylabel="Simulated Hidden States", name="Bayes_Test_States")
            self.plot(test_path, ylabel="Estimated Hidden States", name="Bayes_Viterbi_Test_Path")

            return rate

        else:

            test_path, _, _ = self.model.viterbi_robust(self.test_obs, self.sim_init, self.sim_tran, self.sim_emis)
            rate = np.sum(test_path == self.state_path[1000:])/200  # 187  157

            self.plot(self.test_state_path, ylabel="Simulated Hidden States", name="Max_Test_States")
            self.plot(test_path, ylabel="Estimated Hidden States", name="Max_Viterbi_Test_Path")

            return rate


    def plot(self, data, ylabel, name):
        plt.figure()
        plt.plot(data)
        plt.xlabel("Index")
        plt.ylabel(ylabel)
        fname = os.path.join("..", "plots", name)
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)