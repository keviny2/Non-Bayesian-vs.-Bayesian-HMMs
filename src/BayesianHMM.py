import numpy as np

class BayesianHMM:

    def __init__(self, observations=None, states=None, num_states=6):
        """

        :param observations: observation vector
        :param states: state sequence
        :param num_states: number of states
        """
        self.observations = np.array(observations)
        self.states = np.array(states)
        self.num_obs = len(self.observations)
        self.num_states = num_states

        # model parameters initially set to None
        self.initial_dist = None
        self.A = None
        self.mu = None
        self.sigma_invsq = None

        # model hyperparameters initially set to None
        self.xi = None
        self.kappa = None
        self.beta = None


    def generate_priors(self):
        """
        generate priors according to the paper and set class variables
        :return:
        """
        dir_param = np.ones(self.num_states)
        initial_dist = np.random.dirichlet(dir_param)

        # state transition matrix
        A = np.random.dirichlet(dir_param, self.num_states)

        # generate mus from prior dist.
        max = self.observations.max()
        min = self.observations.min()
        R = max - min

        xi = (min + max) / 2
        kappa = 1 / R**2

        mu = np.random.normal(loc=xi, scale=1/kappa, size=self.num_states)

        # generate sigmas from prior dist.
        beta = np.random.gamma(shape=0.2, scale=10/R**2)
        sigma_invsq = np.random.gamma(shape=2, scale=beta, size=self.num_states)

        # set hmm model parameter class variables
        self.initial_dist = initial_dist
        self.A = A
        self.mu = mu
        self.sigma_invsq = sigma_invsq

        # set hmm model hyperparameter class variables
        self.xi = xi
        self.kappa = kappa
        self.beta = beta


    def sample_parameters(self):
        """
        inference step
        :return:
        """
        self.sample_mu()
        self.sample_sigma_invsq()
        self.sample_beta()
        self.sample_A()
        self.sample_initial_dist()
        self.sample_states()


    def sample_mu(self):
        pass

    def sample_sigma_invsq(self):
        pass

    def sample_beta(self):
        pass

    def sample_A(self):
        pass

    def sample_initial_dist(self):
        pass

    def sample_states(self):
        pass


if __name__ == '__main__':
    # TODO: work on simulating data and initializing state path
    HMM = BayesianHMM([1, 2, 3])
    HMM.generate_priors()