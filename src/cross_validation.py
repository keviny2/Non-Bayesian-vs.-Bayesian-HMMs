import numpy as np

from simulate import SimulateData
from maxlike_hmm import MaxLikeHMM
from bayesian_hmm import BayesianHMM
from visualizer import plot, plot_mu

class CrossValidation:

    def __init__(self, bayesian=False, num_states = 6):
        self.data = SimulateData()
        self.num_states = num_states
        self.bayesian = bayesian
        self.model = None  # will either be a Bayesian or a MaxLike model


    def train(self, num_training=1200, num_test=200, num_iter=10000, num_burnin=1000):
        """

        :param num_training: size of the training set for cv
        :param num_test: size of the test set for cv
        :param num_iter: for BayesianHMM
        :param num_burnin: for BayesianHMM
        :return:
        """

        obs, state_path = self.data.simulate_continuous(num_obs=num_training)

        # TODO: (SHERRY) KDE plot

        index = num_training - num_test
        training_set = obs[:index]
        test_set = obs[index:]
        training_state_path = state_path[:index]
        test_state_path = state_path[index:]


        if self.bayesian:

            self.model = BayesianHMM(observations=training_set, state_path=training_state_path, num_states = self.num_states)
            self.model.generate_priors()

            self.model.sample_parameters(num_iter=num_iter, num_burnin=num_burnin)

            B = [[mu, np.sqrt(1/self.model.sigma_invsq)] for mu in self.model.mu]

            # use viterbi algo because Gibbs sampler not trained on test data
            # BUG: training rate is lower than test rate
            path, _, _ = self.model.viterbi_robust(training_set, self.model.initial_dist, self.model.A, np.array(B))
            # path = self.model.state_path
            rate = np.sum(path == training_state_path)/len(training_state_path)

            plot(training_set, ylabel = "Simulated Observations", name = "Bayes_Original_Observations", bayesian=True)
            plot(training_state_path, ylabel = "Simulated Hidden States", name = "Bayes_Original_States",bayesian=True)
            plot(path, ylabel = "Estimated Hidden States", name = "Bayes_Viterbi_Path", bayesian=True)
            plot_mu(chain=self.model.chain, num_states=self.model.num_states, num_iter=num_iter)


        else:

            self.model = MaxLikeHMM(observations = training_set)
            tran_matrix, emis_matrix, initial = self.model.initial_parameters()

            sim_tran, sim_emis, sim_init = self.model.baum_welch_robust(tran_matrix, emis_matrix, initial)
            path, _, _ = self.model.viterbi_robust(training_set, sim_init, sim_tran, sim_emis)
            rate = np.sum(path == training_state_path)/len(training_state_path)

            self.sim_tran = sim_tran
            self.sim_emis = sim_emis
            self.sim_init = sim_init

            plot(path, ylabel = "Estimated Hidden States", name = "Max_Viterbi_Path",bayesian=False)

        
        print("Training Rate:", rate)
        return state_path, test_set, test_state_path

    def test(self, state_path, test_set, test_state_path):

        if self.bayesian:

            B = [[mu, np.sqrt(1/self.model.sigma_invsq)] for mu in self.model.mu]

            # use viterbi algo because Gibbs sampler not trained on test data
            test_path, _, _ = self.model.viterbi_robust(test_set, self.model.initial_dist, self.model.A, np.array(B))
            rate = np.sum(test_path == state_path[1000:])/200


            plot(test_state_path, ylabel="Simulated Hidden States", name="Bayes_Test_States", bayesian=True)
            plot(test_path, ylabel="Estimated Hidden States", name="Bayes_Viterbi_Test_Path", bayesian=True)


        else:

            test_path, _, _ = self.model.viterbi_robust(test_set, self.sim_init, self.sim_tran, self.sim_emis)
            rate = np.sum(test_path == state_path[1000:])/200  # 187  157

            plot(test_state_path, ylabel="Simulated Hidden States", name="Max_Test_States", bayesian=False)
            plot(test_path, ylabel="Estimated Hidden States", name="Max_Viterbi_Test_Path", bayesian=False)


        print("Test Rate:", rate)


